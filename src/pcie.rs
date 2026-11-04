// Copyright (c) 2026 Kata Containers contributors
//
// SPDX-License-Identifier: Apache-2.0

//! PCI Express configuration space and switch topology, for the
//! peer-to-peer path between a passthrough device and its neighbours.
//!
//! The kernel computes IOMMU groups from the ACS isolation bits at
//! enumeration and, with an IOMMU on, sets those bits itself on every port
//! that has them.  What it leaves alone is Direct Translated P2P, the ACS
//! bit that lets a request a NIC has already translated through ATS cross a
//! switch to a GPU without a round trip through the root complex (NVIDIA
//! HGX Shared NVSwitch guide 19.0, sections 2.6 and 2.7).  Both capabilities
//! are read and written here through the function's `config` file.  A
//! write, and any read past the first 64 bytes, needs root.

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use crate::{attr_hex, context, failed, normalize_bdf, Sysfs};

/// The extended capability IDs come from section 7.6.2 of the PCI Express
/// base specification.
pub const EXT_CAP_ACS: u16 = 0x000d;
pub const EXT_CAP_ATS: u16 = 0x000f;

/// The extended capability list starts right above conventional config
/// space and the whole space is 4 KiB.
const EXT_CAP_START: u64 = 0x100;
const EXT_CONFIG_LEN: u64 = 4096;

/// The ACS registers sit at these offsets from the capability, per section
/// 7.7.11.
pub const ACS_CAP: u64 = 0x4;
pub const ACS_CTRL: u64 = 0x6;
pub const ACS_SOURCE_VALIDATION: u16 = 1 << 0;
pub const ACS_TRANSLATION_BLOCKING: u16 = 1 << 1;
pub const ACS_REQUEST_REDIRECT: u16 = 1 << 2;
pub const ACS_COMPLETION_REDIRECT: u16 = 1 << 3;
pub const ACS_UPSTREAM_FORWARDING: u16 = 1 << 4;
pub const ACS_EGRESS_CONTROL: u16 = 1 << 5;
pub const ACS_DIRECT_TRANSLATED: u16 = 1 << 6;

/// The ATS control register sits at this offset from the capability, per
/// section 10.5.1.3.
pub const ATS_CTRL: u64 = 0x6;
pub const ATS_ENABLE: u16 = 1 << 15;

/// A PCI-to-PCI bridge is a root port or a switch's upstream or downstream
/// port.
const CLASS_PCI_BRIDGE: u32 = 0x0604;

/// A function's configuration space is reached through its sysfs `config`
/// file rather than a mapped bus.  The kernel serialises a user's accesses
/// with its own, and the file is there for every function the bus lists.
#[derive(Debug)]
pub struct Config {
    address: String,
    path: PathBuf,
}

/// An ACS capability is kept as it was read.  The offset travels with the
/// registers so that [`Config::set_acs_ctrl`] writes where the read found
/// them instead of walking the list again.  The capability register comes
/// along because what a port can do bounds what a caller may turn on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acs {
    pub offset: u64,
    pub cap: u16,
    pub ctrl: u16,
}

impl Acs {
    pub fn supports(&self, bits: u16) -> bool {
        self.cap & bits == bits
    }

    pub fn enabled(&self, bits: u16) -> bool {
        self.ctrl & bits == bits
    }
}

/// An ATS capability is read and never written.  A device's firmware decides
/// whether it is there at all, and the IOMMU driver sets the enable bit when
/// it attaches the device to a domain.  A caller only learns whether the
/// device's requests arrive translated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ats {
    pub offset: u64,
    pub ctrl: u16,
}

impl Ats {
    pub fn enabled(&self) -> bool {
        self.ctrl & ATS_ENABLE != 0
    }
}

impl Config {
    pub fn open(sysfs: &Sysfs, address: &str) -> io::Result<Self> {
        let dir = sysfs.device(address).ok_or_else(|| {
            failed(
                io::ErrorKind::InvalidInput,
                format!("{address:?} is not a PCI address"),
            )
        })?;
        let path = dir.join("config");
        if !path.is_file() {
            return Err(failed(
                io::ErrorKind::NotFound,
                format!(
                    "no PCI device {address} under {}",
                    sysfs.devices().display()
                ),
            ));
        }
        let address = dir
            .file_name()
            .expect("a device directory")
            .to_string_lossy()
            .into_owned();

        Ok(Self { address, path })
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    fn read(&self, offset: u64, bytes: &mut [u8]) -> io::Result<()> {
        OpenOptions::new()
            .read(true)
            .open(&self.path)
            .and_then(|file| file.read_exact_at(bytes, offset))
            .map_err(|err| {
                context(
                    err,
                    format!(
                        "{}: read config space at {offset:#x} (past 64 bytes needs root)",
                        self.address
                    ),
                )
            })
    }

    pub fn read16(&self, offset: u64) -> io::Result<u16> {
        let mut bytes = [0u8; 2];
        self.read(offset, &mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    pub fn read32(&self, offset: u64) -> io::Result<u32> {
        let mut bytes = [0u8; 4];
        self.read(offset, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn write16(&self, offset: u64, value: u16) -> io::Result<()> {
        OpenOptions::new()
            .write(true)
            .open(&self.path)
            .and_then(|file| file.write_all_at(&value.to_le_bytes(), offset))
            .map_err(|err| {
                context(
                    err,
                    format!(
                        "{}: write config space at {offset:#x} (needs root)",
                        self.address
                    ),
                )
            })
    }

    /// A conventional PCI function's `config` is 256 bytes and has no list
    /// above it.  Sysfs sizes a PCI Express function's at 4 KiB whoever reads
    /// it.
    fn has_extended_space(&self) -> io::Result<bool> {
        let len = fs::metadata(&self.path)
            .map_err(|err| context(err, format!("{}: stat config space", self.address)))?
            .len();
        Ok(len >= EXT_CONFIG_LEN)
    }

    /// Returns the offset of extended capability `id`, or `None` when the
    /// function does not have it.
    pub fn ext_capability(&self, id: u16) -> io::Result<Option<u64>> {
        if !self.has_extended_space()? {
            return Ok(None);
        }

        let mut offset = EXT_CAP_START;
        // A well-formed list visits each dword at most once.
        for _ in 0..(EXT_CONFIG_LEN - EXT_CAP_START) / 4 {
            let header = self.read32(offset)?;
            // No list at all reads as zero, and a function answering nothing
            // reads as all ones.
            if header == 0 || header == u32::MAX {
                return Ok(None);
            }
            if header as u16 == id {
                return Ok(Some(offset));
            }
            let next = u64::from(header >> 20) & !0x3;
            if next < EXT_CAP_START {
                return Ok(None);
            }
            offset = next;
        }

        Err(failed(
            io::ErrorKind::InvalidData,
            format!(
                "{}: the extended capability list does not terminate",
                self.address
            ),
        ))
    }

    pub fn acs(&self) -> io::Result<Option<Acs>> {
        let Some(offset) = self.ext_capability(EXT_CAP_ACS)? else {
            return Ok(None);
        };

        Ok(Some(Acs {
            offset,
            cap: self.read16(offset + ACS_CAP)?,
            ctrl: self.read16(offset + ACS_CTRL)?,
        }))
    }

    /// The kernel reads this register back and ORs its own isolation bits
    /// into it on every `pci_restore_state`, so a bit set here outlives a
    /// reset of the port.
    pub fn set_acs_ctrl(&self, acs: &Acs, ctrl: u16) -> io::Result<()> {
        self.write16(acs.offset + ACS_CTRL, ctrl)
    }

    pub fn ats(&self) -> io::Result<Option<Ats>> {
        let Some(offset) = self.ext_capability(EXT_CAP_ATS)? else {
            return Ok(None);
        };

        Ok(Some(Ats {
            offset,
            ctrl: self.read16(offset + ATS_CTRL)?,
        }))
    }
}

/// The kernel nests a function's directory under its bridge's, and
/// `bus/pci/devices` links to it.  The real path says who hangs off whom.
fn canonical(sysfs: &Sysfs, address: &str) -> io::Result<PathBuf> {
    let dir = sysfs.device(address).ok_or_else(|| {
        failed(
            io::ErrorKind::InvalidInput,
            format!("{address:?} is not a PCI address"),
        )
    })?;
    fs::canonicalize(&dir).map_err(|err| context(err, format!("resolve {}", dir.display())))
}

/// Returns the bridge a function hangs off.  A root port gets `None`,
/// because its parent is the host bridge, which the bus does not list as a
/// function.
pub fn parent(sysfs: &Sysfs, address: &str) -> io::Result<Option<String>> {
    let real = canonical(sysfs, address)?;

    Ok(real
        .parent()
        .and_then(|dir| dir.file_name())
        .and_then(|name| normalize_bdf(&name.to_string_lossy())))
}

/// Returns the functions directly below a bridge.  They are sorted because
/// `read_dir` returns them in no order a caller can rely on.
pub fn children(sysfs: &Sysfs, address: &str) -> io::Result<Vec<String>> {
    let real = canonical(sysfs, address)?;
    let entries =
        fs::read_dir(&real).map_err(|err| context(err, format!("list {}", real.display())))?;

    let mut children = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| context(err, format!("list {}", real.display())))?;
        if let Some(bdf) = normalize_bdf(&entry.file_name().to_string_lossy()) {
            children.push(bdf);
        }
    }
    children.sort();

    Ok(children)
}

pub fn is_bridge(sysfs: &Sysfs, address: &str) -> bool {
    sysfs
        .device(address)
        .and_then(|dir| attr_hex(&dir, "class").ok())
        .is_some_and(|class| class >> 8 == CLASS_PCI_BRIDGE)
}

/// Returns the ACS-capable downstream ports of every switch between a
/// function and the root.  Those ports are where a peer-to-peer request to
/// or from the function enters and leaves each switch, and so where Direct
/// Translated P2P decides whether it crosses the switch or climbs to the
/// root complex.  Root ports are not switch ports and are left out.  Reading
/// ACS needs root.
pub fn switch_ports(sysfs: &Sysfs, address: &str) -> io::Result<Vec<String>> {
    let mut current = normalize_bdf(address).ok_or_else(|| {
        failed(
            io::ErrorKind::InvalidInput,
            format!("{address:?} is not a PCI address"),
        )
    })?;

    let mut ports = BTreeSet::new();
    while let Some(up) = parent(sysfs, &current)? {
        // The bridges below a bridge are switch ports.  ACS keeps the
        // downstream ones.  The PCIe specification defines the capability for
        // Root Ports, Switch Downstream Ports and multi-function devices, so
        // a nested switch's upstream port reads none and drops out, as
        // 18:00.0 does on the DGX between 16:01.0 and 19:00.0.
        for sibling in children(sysfs, &up)? {
            if is_bridge(sysfs, &sibling) && Config::open(sysfs, &sibling)?.acs()?.is_some() {
                ports.insert(sibling);
            }
        }
        current = up;
    }

    Ok(ports.into_iter().collect())
}

/// Returns what hangs off a port that is not a further switch.  Those are
/// the devices a peer-to-peer request through the port is to or from.
pub fn endpoints(sysfs: &Sysfs, port: &str) -> io::Result<Vec<String>> {
    Ok(children(sysfs, port)?
        .into_iter()
        .filter(|child| !is_bridge(sysfs, child))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs::{self, Fake};
    use rstest::{fixture, rstest};

    const BRIDGE: (u16, u16, u32) = (0x15b3, 0x197c, 0x060400);
    const GPU: (u16, u16, u32) = (0x10de, 0x303f, 0x030200);
    const NIC: (u16, u16, u32) = (0x15b3, 0x1025, 0x020000);

    /// A switch port advertises every isolation bit and Direct Translated
    /// P2P.
    const PORT_CAP: u16 = 0x7f;
    /// The kernel enables these bits on such a port at enumeration.
    const KERNEL_CTRL: u16 = ACS_SOURCE_VALIDATION
        | ACS_REQUEST_REDIRECT
        | ACS_COMPLETION_REDIRECT
        | ACS_UPSTREAM_FORWARDING;

    fn acs_body(cap: u16, ctrl: u16) -> Vec<u8> {
        let mut body = cap.to_le_bytes().to_vec();
        body.extend(ctrl.to_le_bytes());
        body
    }

    fn add(fake: &Fake, parent: Option<&str>, address: &str, identity: (u16, u16, u32)) {
        let (vendor, device, class) = identity;
        match parent {
            Some(parent) => fake.add_pci_device_under(parent, address, vendor, device, class),
            None => fake.add_pci_device(address, vendor, device, class, None),
        }
    }

    /// The DGX layout has a root port, a switch whose two downstream ports
    /// carry a NIC and a second switch, and a GPU below that one.  Only the
    /// downstream ports have ACS.
    ///
    /// ```text
    /// 14:02.0 (root port)
    /// └── 15:00.0 (upstream)
    ///     ├── 16:00.0 (downstream, ACS) ── 17:00.0 NIC
    ///     └── 16:01.0 (downstream, ACS) ── 18:00.0 (upstream)
    ///                                      └── 19:00.0 (downstream, ACS) ── 1a:00.0 GPU
    /// ```
    #[fixture]
    fn switch() -> Fake {
        let fake = testfs::fake();
        add(&fake, None, "0000:14:02.0", BRIDGE);
        add(&fake, Some("0000:14:02.0"), "0000:15:00.0", BRIDGE);
        add(&fake, Some("0000:15:00.0"), "0000:16:00.0", BRIDGE);
        add(&fake, Some("0000:15:00.0"), "0000:16:01.0", BRIDGE);
        add(&fake, Some("0000:16:00.0"), "0000:17:00.0", NIC);
        add(&fake, Some("0000:16:01.0"), "0000:18:00.0", BRIDGE);
        add(&fake, Some("0000:18:00.0"), "0000:19:00.0", BRIDGE);
        add(&fake, Some("0000:19:00.0"), "0000:1a:00.0", GPU);
        // Root and upstream ports have a configuration space too, just no ACS.
        for port in ["0000:14:02.0", "0000:15:00.0", "0000:18:00.0"] {
            fake.set_ext_capabilities(port, &[]);
        }
        for port in ["0000:16:00.0", "0000:16:01.0", "0000:19:00.0"] {
            fake.set_ext_capabilities(
                port,
                &[
                    (0x0001, &[0u8; 4]),
                    (EXT_CAP_ACS, &acs_body(PORT_CAP, KERNEL_CTRL)),
                ],
            );
        }
        fake
    }

    #[rstest]
    #[case::endpoint("0000:1a:00.0", Some("0000:19:00.0"))]
    #[case::downstream_port("0000:16:00.0", Some("0000:15:00.0"))]
    #[case::root_port("0000:14:02.0", None)]
    fn a_functions_parent_is_the_bridge_it_hangs_off(
        switch: Fake,
        #[case] address: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(parent(&switch.sysfs, address).unwrap().as_deref(), expected);
    }

    #[rstest]
    fn a_bridges_children_come_in_address_order(switch: Fake) {
        assert_eq!(
            children(&switch.sysfs, "0000:15:00.0").unwrap(),
            ["0000:16:00.0", "0000:16:01.0"]
        );
        assert!(children(&switch.sysfs, "0000:1a:00.0").unwrap().is_empty());
    }

    #[rstest]
    fn a_bridge_is_known_by_its_class(switch: Fake) {
        assert!(is_bridge(&switch.sysfs, "0000:16:00.0"));
        assert!(!is_bridge(&switch.sysfs, "0000:1a:00.0"));
        assert!(!is_bridge(&switch.sysfs, "0000:ff:00.0"));
        assert!(!is_bridge(&switch.sysfs, "not an address"));
    }

    /// The NIC's port is on the GPU's path too, since a request between
    /// them enters the switch there.  The second switch is the GPU's alone.
    #[rstest]
    #[case::gpu("0000:1a:00.0", &["0000:16:00.0", "0000:16:01.0", "0000:19:00.0"])]
    #[case::nic("0000:17:00.0", &["0000:16:00.0", "0000:16:01.0"])]
    fn the_switch_ports_are_the_acs_capable_downstream_ports_on_the_path(
        switch: Fake,
        #[case] address: &str,
        #[case] expected: &[&str],
    ) {
        assert_eq!(switch_ports(&switch.sysfs, address).unwrap(), expected);
    }

    #[rstest]
    fn a_function_on_a_root_port_has_no_switch_ports(switch: Fake) {
        add(&switch, Some("0000:14:02.0"), "0000:1b:00.0", GPU);

        assert!(switch_ports(&switch.sysfs, "0000:1b:00.0")
            .unwrap()
            .is_empty());
        assert!(switch_ports(&switch.sysfs, "0000:14:02.0")
            .unwrap()
            .is_empty());
    }

    #[rstest]
    fn switch_ports_take_either_spelling_of_the_address(switch: Fake) {
        assert_eq!(
            switch_ports(&switch.sysfs, "1a:00.0").unwrap(),
            switch_ports(&switch.sysfs, "0000:1a:00.0").unwrap()
        );
        assert_eq!(
            switch_ports(&switch.sysfs, "nope").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[rstest]
    fn endpoints_are_the_functions_below_a_port_that_are_not_bridges(switch: Fake) {
        assert_eq!(
            endpoints(&switch.sysfs, "0000:16:00.0").unwrap(),
            ["0000:17:00.0"]
        );
        assert!(endpoints(&switch.sysfs, "0000:16:01.0").unwrap().is_empty());
    }

    #[rstest]
    fn a_missing_function_cannot_be_walked(switch: Fake) {
        assert_eq!(
            parent(&switch.sysfs, "0000:ff:00.0").unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(
            children(&switch.sysfs, "nope").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[rstest]
    fn acs_is_read_where_the_list_puts_it(switch: Fake) {
        let config = Config::open(&switch.sysfs, "16:00.0").unwrap();

        assert_eq!(config.address(), "0000:16:00.0");
        let acs = config.acs().unwrap().unwrap();
        assert_eq!(acs.offset, 0x108, "after one 8-byte capability");
        assert!(acs.supports(ACS_DIRECT_TRANSLATED));
        assert!(acs.enabled(KERNEL_CTRL));
        assert!(!acs.enabled(ACS_DIRECT_TRANSLATED));
    }

    #[rstest]
    fn set_acs_ctrl_writes_the_control_register_only(switch: Fake) {
        let config = Config::open(&switch.sysfs, "0000:16:00.0").unwrap();
        let acs = config.acs().unwrap().unwrap();

        config
            .set_acs_ctrl(&acs, acs.ctrl | ACS_DIRECT_TRANSLATED)
            .unwrap();

        let after = config.acs().unwrap().unwrap();
        assert_eq!(after.cap, acs.cap);
        assert_eq!(after.ctrl, KERNEL_CTRL | ACS_DIRECT_TRANSLATED);
        assert!(after.enabled(ACS_DIRECT_TRANSLATED));
    }

    #[rstest]
    #[case::exposed_and_enabled(ATS_ENABLE, true)]
    #[case::exposed_and_off(0, false)]
    fn ats_is_read_from_the_capability(switch: Fake, #[case] ctrl: u16, #[case] enabled: bool) {
        switch.set_ext_capabilities(
            "0000:17:00.0",
            &[(EXT_CAP_ATS, &[0, 0, ctrl as u8, (ctrl >> 8) as u8])],
        );

        let ats = Config::open(&switch.sysfs, "0000:17:00.0")
            .unwrap()
            .ats()
            .unwrap()
            .unwrap();

        assert_eq!(ats.offset, 0x100);
        assert_eq!(ats.enabled(), enabled);
    }

    /// A NIC whose firmware keeps ATS off exposes no capability for it.
    #[rstest]
    fn a_capability_the_function_lacks_is_none(switch: Fake) {
        let port = Config::open(&switch.sysfs, "0000:16:00.0").unwrap();
        assert!(port.ats().unwrap().is_none());

        switch.set_ext_capabilities("0000:17:00.0", &[]);
        let nic = Config::open(&switch.sysfs, "0000:17:00.0").unwrap();
        assert!(nic.ats().unwrap().is_none());
        assert!(nic.acs().unwrap().is_none());
    }

    /// A function answering nothing reads as all ones.
    #[rstest]
    fn a_space_that_reads_all_ones_has_no_capabilities(switch: Fake) {
        std::fs::write(switch.device("0000:16:00.0").join("config"), [0xffu8; 4096]).unwrap();

        let config = Config::open(&switch.sysfs, "0000:16:00.0").unwrap();

        assert!(config.acs().unwrap().is_none());
    }

    #[rstest]
    fn a_next_pointer_into_conventional_space_ends_the_list(switch: Fake) {
        let mut config = vec![0u8; 4096];
        let header: u32 = 0x0001 | (1 << 16) | (0x40 << 20);
        config[0x100..0x104].copy_from_slice(&header.to_le_bytes());
        std::fs::write(switch.device("0000:16:00.0").join("config"), config).unwrap();

        assert!(Config::open(&switch.sysfs, "0000:16:00.0")
            .unwrap()
            .acs()
            .unwrap()
            .is_none());
    }

    #[rstest]
    fn a_list_that_loops_is_refused(switch: Fake) {
        let mut config = vec![0u8; 4096];
        let header: u32 = 0x0001 | (1 << 16) | (0x100 << 20);
        config[0x100..0x104].copy_from_slice(&header.to_le_bytes());
        std::fs::write(switch.device("0000:16:00.0").join("config"), config).unwrap();

        let err = Config::open(&switch.sysfs, "0000:16:00.0")
            .unwrap()
            .acs()
            .unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// A conventional function has 256 bytes and nothing above them.
    #[rstest]
    fn a_conventional_function_has_no_extended_space(switch: Fake) {
        std::fs::write(switch.device("0000:17:00.0").join("config"), [0u8; 256]).unwrap();

        let config = Config::open(&switch.sysfs, "0000:17:00.0").unwrap();

        assert!(config.ext_capability(EXT_CAP_ATS).unwrap().is_none());
    }

    /// An unprivileged reader gets 64 bytes of a space sysfs still sizes at
    /// 4 KiB, so the error has to say so rather than report no capability.
    #[rstest]
    fn a_read_the_kernel_cuts_short_names_the_privilege() {
        let fake = testfs::fake();
        fake.add_mappable_device("0000:65:00.0", 0x10de, 0x2330, 0x030200, 0x1000);
        let config = Config::open(&fake.sysfs, "0000:65:00.0").unwrap();

        let err = config.read16(EXT_CAP_START).unwrap_err().to_string();

        assert!(err.contains("needs root"), "{err}");
        assert!(err.contains("0000:65:00.0"), "{err}");
    }

    #[rstest]
    fn open_refuses_what_is_not_there(switch: Fake) {
        assert_eq!(
            Config::open(&switch.sysfs, "../etc").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            Config::open(&switch.sysfs, "0000:ff:00.0")
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        // This function is added without its config file, as one sysfs only
        // half shows would be.
        add(&switch, None, "0000:1c:00.0", GPU);
        assert_eq!(
            Config::open(&switch.sysfs, "0000:1c:00.0")
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
    }

    #[rstest]
    fn writes_need_a_writable_file(switch: Fake) {
        let config = Config::open(&switch.sysfs, "0000:16:00.0").unwrap();
        let path = switch.device("0000:16:00.0").join("config");
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&path, permissions).unwrap();
        if std::fs::OpenOptions::new().write(true).open(&path).is_ok() {
            return; // root ignores the mode bits
        }

        let err = config.write16(0x10e, 0x5d).unwrap_err().to_string();

        assert!(err.contains("needs root"), "{err}");
    }
}
