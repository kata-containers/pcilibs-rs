// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! Init must identify NVLink hardware before choosing drivers and services.
//! Keep discovery independent of `cc` so callers need no firmware access.
//!
//! Hardware rules: NVIDIA HGX integration guide, release 19.0, §2.5.2:
//! <https://docs.nvidia.com/hgx-platforms/shared-nvswitch-gpu-passthrough-virtualization-integration-guide.pdf>
//!
//! ```no_run
//! use pcilibs_rs::{nvlink, Sysfs};
//!
//! # fn main() -> std::io::Result<()> {
//! let sysfs = Sysfs::default();
//! let topology = nvlink::discover(&sysfs)?;
//! let gpu_count = nvlink::discover_gpus(&sysfs)?.len();
//! // Port GUIDs exist only after the RDMA drivers register their devices.
//! let ports = nvlink::discover_management_ports(&sysfs)?;
//! let guid = ports.first().map(|port| format!("0x{:016x}", port.guid));
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read};
use std::net::Ipv6Addr;
use std::path::Path;

use crate::platform::linux::{pci_functions, PciFunction};
use crate::{attr_hex, context, failed, normalize_bdf, Sysfs};

// Preserve the original discovery API for existing callers.
pub use crate::platform::{discover_gpus, discover_topology as discover, Topology};

const NVIDIA: u32 = 0x10de;
const MELLANOX: u32 = 0x15b3;
const IS_SM_DISABLED: u32 = 1 << 10;
// PCI VPD has a 15-bit byte address. Bound allocation even for a fake sysfs file.
const MAX_VPD: u64 = 1 << 15;
const VPD_END: u8 = 0x78;
const VPD_READ_ONLY: u8 = 0x90;
const VPD_READ_WRITE: u8 = 0x91;

#[derive(Debug, PartialEq, Eq)]
pub struct ManagementFunction {
    pub bdf: String,
    /// Own `SMDL=SW_MNG` marker; keep separate from firmware-controlled SM capability.
    pub sw_mng: bool,
}

/// Keep PCI identity with the GUID to avoid selecting an unrelated NIC.
#[derive(Debug, PartialEq, Eq)]
pub struct ManagementPort {
    pub pci_bdf: String,
    pub ib_device: String,
    pub port: u32,
    /// Low 64 bits of GID 0. Format as `format!("0x{:016x}", port.guid)` for FM.
    pub guid: u64,
}

fn is_switch(function: &PciFunction) -> bool {
    function.vendor == NVIDIA && function.class >> 8 == 0x0680
}

/// H100/H200 expose switches on PCI; ConnectX-managed switches need VPD discovery.
/// Return candidates without BAR access so this works before driver loading.
pub fn discover_switches(sysfs: &Sysfs) -> io::Result<Vec<String>> {
    Ok(pci_functions(sysfs)?
        .into_iter()
        .filter(is_switch)
        .map(|device| device.bdf)
        .collect())
}

pub(crate) fn discover_fabric(
    sysfs: &Sysfs,
    functions: &[PciFunction],
) -> io::Result<(Vec<String>, Vec<ManagementFunction>)> {
    let mut switches = Vec::new();
    let mut mellanox = Vec::new();
    let mut management_slots = BTreeSet::new();
    for function in functions {
        if is_switch(function) {
            switches.push(function.bdf.clone());
        } else if function.vendor == MELLANOX {
            let device_path = sysfs.devices().join(&function.bdf);
            // A shared slot does not make an SR-IOV VF a management PF.
            match fs::symlink_metadata(device_path.join("physfn")) {
                Ok(_) => continue,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(context(e, device_path.display())),
            }
            let path = device_path.join("vpd");
            let sw_mng = read_sw_mng(&path)?;
            if sw_mng {
                management_slots.insert(slot(&function.bdf).to_owned());
            }
            mellanox.push(ManagementFunction {
                bdf: function.bdf.clone(),
                sw_mng,
            });
        }
    }
    let management_functions = mellanox
        .into_iter()
        .filter(|function| management_slots.contains(slot(&function.bdf)))
        .collect();
    Ok((switches, management_functions))
}

// Normalization guarantees the function separator.
fn slot(bdf: &str) -> &str {
    bdf.rsplit_once('.').unwrap().0
}

fn invalid(what: &str) -> io::Error {
    failed(io::ErrorKind::InvalidData, what.to_owned())
}

fn read_sw_mng(path: &Path) -> io::Result<bool> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(context(e, path.display())),
    };
    let mut data = Vec::new();
    file.take(MAX_VPD + 1)
        .read_to_end(&mut data)
        .map_err(|e| context(e, path.display()))?;
    if data.len() as u64 > MAX_VPD {
        return Err(context(
            invalid("VPD exceeds PCI address space"),
            path.display(),
        ));
    }
    vpd_sw_mng(&data).map_err(|e| context(e, path.display()))
}

// Raw substring matching can mistake serial numbers for a management role.
fn vpd_sw_mng(mut data: &[u8]) -> io::Result<bool> {
    let mut found = false;
    while let Some((&tag, rest)) = data.split_first() {
        if tag == VPD_END {
            return Ok(found);
        }
        let (length, payload) = if tag & 0x80 != 0 {
            let length = rest
                .get(..2)
                .ok_or_else(|| invalid("truncated VPD resource length"))?;
            (
                u16::from_le_bytes([length[0], length[1]]) as usize,
                &rest[2..],
            )
        } else {
            ((tag & 7) as usize, rest)
        };
        let resource = payload
            .get(..length)
            .ok_or_else(|| invalid("truncated VPD resource"))?;
        if matches!(tag, VPD_READ_ONLY | VPD_READ_WRITE) {
            let mut keywords = resource;
            while !keywords.is_empty() {
                let header = keywords
                    .get(..3)
                    .ok_or_else(|| invalid("truncated VPD keyword"))?;
                let end = 3 + header[2] as usize;
                let value = keywords
                    .get(3..end)
                    .ok_or_else(|| invalid("truncated VPD value"))?;
                if &header[..2] == b"VA" {
                    found |= value
                        .split(|b| *b == b':')
                        .any(|field| field == b"SMDL=SW_MNG");
                }
                keywords = &keywords[end..];
            }
        }
        data = &payload[length..];
    }
    Err(invalid("VPD has no end resource"))
}

/// PCI role and SM capability keep unrelated NICs out of FM/NVLSM selection.
/// Sort by BDF, numeric port, then IB name for deterministic selection.
/// Requires registered IB devices: missing trees/read/parse failures are errors;
/// an existing tree with no eligible ports returns an empty vector.
pub fn discover_management_ports(sysfs: &Sysfs) -> io::Result<Vec<ManagementPort>> {
    let topology = discover(sysfs)?;
    let root = sysfs.infiniband();
    let entries = fs::read_dir(&root).map_err(|e| context(e, root.display()))?;
    let mut ports = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let device = match fs::read_link(path.join("device")) {
            Ok(device) => device,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(context(e, path.display())),
        };
        let Some(bdf) = device
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(normalize_bdf)
        else {
            continue;
        };
        if !topology.management_functions.iter().any(|f| f.bdf == bdf) {
            continue;
        }
        let port_root = path.join("ports");
        for port in fs::read_dir(&port_root).map_err(|e| context(e, port_root.display()))? {
            let port = port?;
            let Some(number) = port
                .file_name()
                .to_str()
                .and_then(|n| n.parse::<u32>().ok())
                .filter(|n| *n > 0)
            else {
                continue;
            };
            if let Some(guid) = management_guid(&port.path())? {
                ports.push(ManagementPort {
                    pci_bdf: bdf.clone(),
                    ib_device: entry.file_name().to_string_lossy().into_owned(),
                    port: number,
                    guid,
                });
            }
        }
    }
    ports.sort_by(|a, b| {
        (&a.pci_bdf, a.port, &a.ib_device).cmp(&(&b.pci_bdf, b.port, &b.ib_device))
    });
    Ok(ports)
}

fn management_guid(path: &Path) -> io::Result<Option<u64>> {
    let layer = path.join("link_layer");
    if fs::read_to_string(&layer)
        .map_err(|e| context(e, layer.display()))?
        .trim()
        != "InfiniBand"
    {
        return Ok(None);
    }
    if attr_hex(path, "cap_mask")? & IS_SM_DISABLED != 0 {
        return Ok(None);
    }
    let gid_path = path.join("gids/0");
    let text = fs::read_to_string(&gid_path).map_err(|e| context(e, gid_path.display()))?;
    let gid: Ipv6Addr = text
        .trim()
        .parse()
        .map_err(|_| context(invalid("invalid GID"), gid_path.display()))?;
    let guid = u128::from(gid) as u64;
    if guid == 0 {
        return Err(context(invalid("zero port GUID"), gid_path.display()));
    }
    Ok(Some(guid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs::{self, Fake};
    use rstest::rstest;

    const SW_MNG: &[u8] = b"MLX:MN=MLNX:CSKU=V2:UUID=V3:PCI=V0:SMDL=SW_MNG:MODL=C7010Z";

    fn vpd(keyword: &[u8; 2], value: &[u8]) -> Vec<u8> {
        let length = (value.len() + 3) as u16;
        let mut data = vec![0x90];
        data.extend(length.to_le_bytes());
        data.extend(keyword);
        data.push(value.len() as u8);
        data.extend(value);
        data.push(0x78);
        data
    }

    fn mlx(fake: &Fake, bdf: &str, marker: bool) {
        fake.add_pci_device(bdf, 0x15b3, 0x1021, 0x020700, None);
        if marker {
            fs::write(fake.device(bdf).join("vpd"), vpd(b"VA", SW_MNG)).unwrap();
        }
    }

    fn port(fake: &Fake, ib: &str, number: u32, mask: &str, gid: &str) {
        let path = fake
            .sysfs
            .infiniband()
            .join(ib)
            .join("ports")
            .join(number.to_string());
        fs::create_dir_all(path.join("gids")).unwrap();
        fs::write(path.join("cap_mask"), mask).unwrap();
        fs::write(path.join("link_layer"), "InfiniBand\n").unwrap();
        fs::write(path.join("gids/0"), gid).unwrap();
    }

    #[test]
    fn h100_baseboard_discovers_four_switches_and_eight_gpus_without_drivers() {
        let fake = testfs::fake();
        for bus in 3..7 {
            fake.add_pci_device(
                &format!("0000:{bus:02x}:00.0"),
                0x10de,
                0x22a3,
                0x068000,
                None,
            );
        }
        for bus in 0x40..0x48 {
            fake.add_pci_device(
                &format!("0000:{bus:02x}:00.0"),
                0x10de,
                0x2330,
                0x030200,
                None,
            );
        }
        fake.add_pci_device("0000:01:00.0", 0x8086, 1, 0x068000, None);
        fake.add_pci_device("0000:02:00.0", 0x10de, 1, 0x040300, None);
        fs::create_dir(fake.sysfs.devices().join("not-a-bdf")).unwrap();
        let topology = discover(&fake.sysfs).unwrap();
        assert_eq!(
            topology.switches,
            [
                "0000:03:00.0",
                "0000:04:00.0",
                "0000:05:00.0",
                "0000:06:00.0"
            ]
        );
        assert_eq!(topology.gpus.len(), 8);
        assert!(topology.management_functions.is_empty());
        assert_eq!(discover_gpus(&fake.sysfs).unwrap(), topology.gpus);
        assert_eq!(discover_switches(&fake.sysfs).unwrap(), topology.switches);
    }

    #[rstest]
    #[case::service_vm(0)]
    #[case::full_passthrough(8)]
    fn management_pfs_are_not_a_count_of_switches(#[case] gpu_count: u32) {
        let fake = testfs::fake();
        for i in 0..gpu_count {
            fake.add_pci_device(
                &format!("0000:{:02x}:00.0", 0x40 + i),
                0x10de,
                0x2901,
                0x030200,
                None,
            );
        }
        // Only the two LPFs carry SW_MNG, but all four present PFs belong.
        for function in [3, 1, 2, 0] {
            mlx(&fake, &format!("0000:05:00.{function}"), function < 2);
        }
        mlx(&fake, "0000:05:00.5", true);
        std::os::unix::fs::symlink(
            fake.device("0000:05:00.0"),
            fake.device("0000:05:00.5").join("physfn"),
        )
        .unwrap();
        mlx(&fake, "0000:06:00.0", false);
        mlx(&fake, "0001:05:00.0", false);
        fake.add_pci_device("0000:05:00.4", 0x8086, 1, 0x020000, None);
        let topology = discover(&fake.sysfs).unwrap();
        assert_eq!(topology.gpus.len(), gpu_count as usize);
        assert!(topology.switches.is_empty());
        assert_eq!(
            topology
                .management_functions
                .iter()
                .map(|f| (f.bdf.as_str(), f.sw_mng))
                .collect::<Vec<_>>(),
            [
                ("0000:05:00.0", true),
                ("0000:05:00.1", true),
                ("0000:05:00.2", false),
                ("0000:05:00.3", false)
            ]
        );
    }

    #[test]
    fn no_fixed_function_count_or_device_id_is_required() {
        let fake = testfs::fake();
        // VM assignment and newer ConnectX IDs must not hide a documented VPD role.
        fake.add_pci_device("0000:05:00.3", 0x15b3, 0xffff, 0x020700, None);
        fs::write(fake.device("0000:05:00.3").join("vpd"), vpd(b"VA", SW_MNG)).unwrap();
        let topology = discover(&fake.sysfs).unwrap();
        assert_eq!(topology.management_functions.len(), 1);
        assert_eq!(topology.management_functions[0].bdf, "0000:05:00.3");
    }

    #[test]
    fn gpu_enumeration_includes_pre_cc_and_vga_devices_without_reading_vpd() {
        let fake = testfs::fake();
        fake.add_pci_device("0000:04:00.0", 0x10de, 0x20b0, 0x030200, None);
        fake.add_pci_device("0000:03:00.0", 0x10de, 0x2b85, 0x030000, None);
        mlx(&fake, "0000:02:00.0", false);
        fs::write(fake.device("0000:02:00.0").join("vpd"), b"malformed").unwrap();
        assert_eq!(
            discover_gpus(&fake.sysfs).unwrap(),
            ["0000:03:00.0", "0000:04:00.0"]
        );
    }

    #[test]
    fn missing_trees_and_bad_identity_are_errors_not_empty_topology() {
        let fake = testfs::fake();
        assert_eq!(discover(&fake.sysfs).unwrap(), Topology::default());
        fake.add_pci_device("0000:04:00.0", 0x10de, 0x2330, 0x030200, None);
        fs::write(fake.device("0000:04:00.0").join("class"), "garbage").unwrap();
        assert_eq!(
            discover(&fake.sysfs).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        fs::remove_dir_all(fake.sysfs.devices()).unwrap();
        assert_eq!(
            discover(&fake.sysfs).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
    }

    #[rstest]
    #[case::documented(b"VA", SW_MNG, true)]
    #[case::last_field(b"VA", b"MLX:SMDL=SW_MNG", true)]
    #[case::substring(b"VA", b"MLX:SMDL=SW_MNG_OTHER", false)]
    #[case::wrong_key(b"VA", b"MLX:OTHER_SMDL=SW_MNG", false)]
    #[case::bare_marker(b"VA", b"SW_MNG", false)]
    #[case::serial_number(b"SN", b"SMDL=SW_MNG", false)]
    fn vpd_requires_the_exact_vendor_field(
        #[case] key: &[u8; 2],
        #[case] value: &[u8],
        #[case] expected: bool,
    ) {
        assert_eq!(vpd_sw_mng(&vpd(key, value)).unwrap(), expected);
    }

    #[test]
    fn vpd_skips_identifier_and_small_resources_and_reads_multiple_keywords() {
        let mut data = vec![0x82, 11, 0];
        data.extend(b"SMDL=SW_MNG");
        data.extend([0x09, 0]);
        // Keep scanning after the serial number.
        data.extend(vpd(b"SN", b"123").into_iter().take(9));
        data.extend(vpd(b"VA", SW_MNG));
        assert!(vpd_sw_mng(&data).unwrap());
        data[0] = 0x91; // identifier bytes are not a valid keyword resource
        assert!(vpd_sw_mng(&data).is_err());
    }

    #[rstest]
    #[case::no_end(vec![])]
    #[case::short_length(vec![0x90, 1])]
    #[case::short_resource(vec![0x90, 10, 0, 0])]
    #[case::short_keyword(vec![0x90, 1, 0, b'V', 0x78])]
    #[case::short_value(vec![0x90, 3, 0, b'V', b'A', 8, 0x78])]
    fn corrupt_vpd_is_an_error(#[case] data: Vec<u8>) {
        assert_eq!(
            vpd_sw_mng(&data).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn unreadable_oversized_and_corrupt_vpd_do_not_hide_management_devices() {
        let fake = testfs::fake();
        mlx(&fake, "0000:05:00.0", false);
        let path = fake.device("0000:05:00.0").join("vpd");
        fs::create_dir(&path).unwrap();
        assert!(discover(&fake.sysfs).is_err());
        fs::remove_dir(&path).unwrap();
        for bytes in [vec![0; MAX_VPD as usize + 1], vec![0x90]] {
            fs::write(&path, bytes).unwrap();
            let error = discover(&fake.sysfs).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains("05:00.0/vpd"));
        }
    }

    #[test]
    fn ports_are_filtered_by_pci_role_and_capability_and_sorted_numerically() {
        let fake = testfs::fake();
        for (bdf, marker) in [
            ("0000:05:00.0", true),
            ("0000:05:00.2", false),
            ("0000:01:00.0", false),
        ] {
            mlx(&fake, bdf, marker);
        }
        fake.add_infiniband("mlx5_0", "0000:01:00.0", "1: CA", ""); // Must not win first-port selection.
        fake.add_infiniband("mlx5_1", "0000:05:00.2", "1: CA", "");
        fake.add_infiniband("mlx5_9", "0000:05:00.0", "1: CA", "");
        port(&fake, "mlx5_0", 1, "0x0", "fe80::9999");
        port(&fake, "mlx5_1", 1, "0x400", "malformed"); // disabled: never read GID
        port(
            &fake,
            "mlx5_9",
            10,
            "0xa750e84a",
            "fe80:0000:0000:0000:9c63:c003:00e5:6b5c",
        );
        port(&fake, "mlx5_9", 2, "0xa750e848", "fe80::1:2");
        port(&fake, "mlx5_9", 3, "0", "malformed");
        fs::write(
            fake.sysfs.infiniband().join("mlx5_9/ports/3/link_layer"),
            "Ethernet",
        )
        .unwrap();
        fs::create_dir(fake.sysfs.infiniband().join("mlx5_9/ports/not-a-port")).unwrap();
        fs::create_dir(fake.sysfs.infiniband().join("no-pci-link")).unwrap();
        let ports = discover_management_ports(&fake.sysfs).unwrap();
        assert_eq!(
            ports.iter().map(|p| (p.port, p.guid)).collect::<Vec<_>>(),
            [(2, 0x10002), (10, 0x9c63c00300e56b5c)]
        );
        assert!(ports
            .iter()
            .all(|p| p.pci_bdf == "0000:05:00.0" && p.ib_device == "mlx5_9"));
        assert_eq!(format!("0x{:016x}", ports[0].guid), "0x0000000000010002");
        // Firmware capability, not function number, decides SM eligibility.
        port(&fake, "mlx5_1", 1, "0", "fe80::2");
        assert_eq!(discover_management_ports(&fake.sysfs).unwrap().len(), 3);
    }

    #[rstest]
    #[case::bad_mask("invalid", "fe80::1", "cap_mask")]
    #[case::overflow_mask("0x100000000", "fe80::1", "cap_mask")]
    #[case::bad_gid("0", "fe80::nothex", "gids/0")]
    #[case::zero_guid("0", "fe80::", "gids/0")]
    fn malformed_management_ports_are_errors(
        #[case] mask: &str,
        #[case] gid: &str,
        #[case] file: &str,
    ) {
        let fake = testfs::fake();
        mlx(&fake, "0000:05:00.0", true);
        fake.add_infiniband("mlx5_0", "0000:05:00.0", "1: CA", "");
        port(&fake, "mlx5_0", 1, mask, gid);
        let error = discover_management_ports(&fake.sysfs).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains(file), "{error}");
    }

    #[test]
    fn missing_ib_tree_is_distinct_from_no_eligible_ports() {
        let fake = testfs::fake();
        assert_eq!(
            discover_management_ports(&fake.sysfs).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        fs::create_dir_all(fake.sysfs.infiniband()).unwrap();
        assert!(discover_management_ports(&fake.sysfs).unwrap().is_empty());
    }

    #[rstest]
    #[case::layer("link_layer")]
    #[case::mask("cap_mask")]
    #[case::gid("gids/0")]
    fn missing_port_attributes_are_not_silently_skipped(#[case] file: &str) {
        let fake = testfs::fake();
        mlx(&fake, "0000:05:00.0", true);
        fake.add_infiniband("mlx5_0", "0000:05:00.0", "1: CA", "");
        port(&fake, "mlx5_0", 1, "0", "fe80::1");
        fs::remove_file(fake.sysfs.infiniband().join("mlx5_0/ports/1").join(file)).unwrap();
        assert_eq!(
            discover_management_ports(&fake.sysfs).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
    }
}
