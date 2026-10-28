// Copyright (c) Kata Containers contributors
//
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use crate::{attr_hex, normalize_bdf, Sysfs};

/// Root of the IOMMUFD character device tree; cdevs live at
/// `<IOMMUFD_VFIO_DIR>/devices/vfioN`.  Devfs rather than sysfs, so it stays a
/// path of its own.  A kernel contract, not configuration.
pub const IOMMUFD_VFIO_DIR: &str = "/dev/vfio";

/// One IOMMUFD character device: the file at `<vfio_dir>/devices/vfioN`,
/// together with the PCI function behind it, read from sysfs.
#[derive(Clone, Debug)]
pub struct IommufdDev {
    /// Kernel-assigned index: the `N` in `/dev/vfio/devices/vfioN`.
    pub num: u32,
    /// Absolute path to the character device.
    pub path: PathBuf,
    /// Canonical address of the PCI function, e.g. `0000:65:00.0`: the
    /// target of `/sys/class/vfio-dev/vfioN/device`, and the name a
    /// sysfs-wide scan such as [`crate::nvlink::discover`] knows it by.
    pub bdf: String,
    /// PCI vendor ID (e.g. `0x10de` for NVIDIA).
    pub vendor: u16,
    /// PCI device ID (e.g. `0x2330` for an H100 SXM5 80GB).
    pub device: u16,
    /// Full 24-bit PCI class code (class byte | subclass byte | prog-if byte).
    pub class: u32,
}

impl IommufdDev {
    /// Base class and subclass — the upper 16 bits of the 24-bit class code,
    /// e.g. `0x0302` for a 3D controller.  This is the granularity device
    /// classification usually wants; the low byte (prog-if) varies by
    /// programming interface, not device kind.
    pub fn class_prefix(&self) -> u16 {
        (self.class >> 8) as u16
    }

    /// The device's name in the PCI ID database (compile-time `pci-ids`
    /// crate), e.g. "GH100 [H100 SXM5 80GB]" for 10de:2330.  None if the
    /// database does not know the (vendor, device) pair.
    pub fn device_name(&self) -> Option<&'static str> {
        crate::pci_ids::Device::from_vid_pid(self.vendor, self.device).map(|d| d.name())
    }
}

/// PCI class codes (base + subclass, i.e. `class >> 8`) that must NOT be
/// passed through to a guest in a legacy IOMMU group.
///
/// These devices appear in the same IOMMU group as a GPU or NIC but cannot
/// be passed through themselves:
///
/// | Code   | Class                |
/// |--------|----------------------|
/// | 0x0600 | Host Bridge          |
/// | 0x0604 | PCI-to-PCI Bridge    |
/// | 0x0403 | Audio device         |
///
/// NVSwitches (`0x0680`, Other Bridge) are intentionally absent: they are
/// passthrough-capable and the device plugin binds them to `vfio-pci` on
/// purpose.
const NON_PASSTHROUGH_CLASSES: &[u16] = &[0x0600, 0x0604, 0x0403];

/// Returns `true` if a device with the given 24-bit PCI class code can be
/// passed through to a guest.
///
/// The check is based on the base+subclass (upper 16 bits of the 24-bit
/// class code).  Bridges and audio companions that share an IOMMU group
/// with a GPU are not passthrough-capable; everything else is assumed to be.
///
/// # Example
///
/// ```
/// use pcilibs_rs::is_passthrough_capable_class;
///
/// assert!(is_passthrough_capable_class(0x030200));  // 3D controller (GPU)
/// assert!(is_passthrough_capable_class(0x068000));  // NVSwitch (Other Bridge)
/// assert!(!is_passthrough_capable_class(0x060000)); // Host Bridge
/// assert!(!is_passthrough_capable_class(0x040300)); // Audio device
/// ```
pub fn is_passthrough_capable_class(class: u32) -> bool {
    !NON_PASSTHROUGH_CLASSES.contains(&((class >> 8) as u16))
}

/// The PCI function behind `vfio<num>`, through
/// `<sysfs>/class/vfio-dev/vfio<num>/device`.  `None` unless that link
/// names a PCI function whose identity parses: a cdev over a platform or
/// mediated device has nothing this type describes.
fn read_dev(num: u32, path: PathBuf, sysfs: &Sysfs) -> Option<IommufdDev> {
    let device = sysfs.vfio_dev(&format!("vfio{num}")).join("device");
    let bdf = normalize_bdf(fs::read_link(&device).ok()?.file_name()?.to_str()?)?;
    let id = |name| u16::try_from(attr_hex(&device, name).ok()?).ok();
    Some(IommufdDev {
        num,
        path,
        bdf,
        vendor: id("vendor")?,
        device: id("device")?,
        class: attr_hex(&device, "class").ok()?,
    })
}

/// Look up a single IOMMUFD character device by its kernel name (e.g. `"vfio5"`).
///
/// Returns `None` if the cdev or its sysfs entry is absent or any field
/// cannot be parsed.
///
/// This is the single-device counterpart to [`enumerate_iommufd`]: use it
/// when the caller already knows which cdev it wants (e.g. from a CDI spec)
/// and does not need to scan the full `/dev/vfio/devices/` directory.
pub fn lookup_iommufd_dev(name: &str, vfio_dir: &Path, sysfs: &Sysfs) -> Option<IommufdDev> {
    let num = name.strip_prefix("vfio")?.parse::<u32>().ok()?;
    let path = vfio_dir.join("devices").join(name);
    if !path.exists() {
        return None;
    }
    read_dev(num, path, sysfs)
}

/// Enumerate all IOMMUFD character devices under `<vfio_dir>/devices/` and
/// resolve their PCI function from sysfs.  Entries whose sysfs files are
/// absent or unparseable are silently skipped.  Result is sorted by device
/// number.
pub fn enumerate_iommufd(vfio_dir: &Path, sysfs: &Sysfs) -> Vec<IommufdDev> {
    let Ok(rd) = fs::read_dir(vfio_dir.join("devices")) else {
        return vec![];
    };
    let mut devs: Vec<IommufdDev> = rd
        .flatten()
        .filter_map(|e| {
            let num = e
                .file_name()
                .to_str()?
                .strip_prefix("vfio")?
                .parse::<u32>()
                .ok()?;
            read_dev(num, e.path(), sysfs)
        })
        .collect();
    devs.sort_by_key(|d| d.num);
    devs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs::{add, add_at};
    use tempfile::TempDir;

    #[test]
    fn enumerates_and_sorts_by_num() {
        let root = TempDir::new().unwrap();
        add(root.path(), 42, 0x10de, 0x2330, 0x030200);
        add(root.path(), 7, 0x10de, 0x22a3, 0x068000);
        add(root.path(), 3, 0x15b3, 0x101e, 0x020000);

        let devs = enumerate_iommufd(root.path(), &Sysfs::new(root.path()));
        assert_eq!(
            devs.iter().map(|d| d.num).collect::<Vec<_>>(),
            vec![3, 7, 42]
        );
        assert_eq!(devs[1].vendor, 0x10de);
        assert_eq!(devs[1].device, 0x22a3);
        assert_eq!(devs[1].class, 0x068000);
        assert!(devs[2].path.ends_with("devices/vfio42"));
        assert_eq!(devs[2].bdf, "0000:2a:00.0");
    }

    /// The address is the link target's name, spelled the way sysfs does.
    #[test]
    fn bdf_is_the_function_the_vfio_dev_links_to() {
        let root = TempDir::new().unwrap();
        add_at(root.path(), 0, "05:00.1", 0x15b3, 0x1021, 0x020700);

        let devs = enumerate_iommufd(root.path(), &Sysfs::new(root.path()));

        assert_eq!(devs[0].bdf, "0000:05:00.1");
    }

    /// Only a PCI function has the identity this type carries: a cdev over
    /// anything else, or over a `device` that is not a link at all, is not
    /// one.
    #[test]
    fn cdev_without_a_pci_function_behind_it_is_skipped() {
        let root = TempDir::new().unwrap();
        let sysfs = Sysfs::new(root.path());
        let devices = root.path().join("devices");
        fs::create_dir_all(&devices).unwrap();
        for (n, target) in [(0, Some("../../../fff00000.platform")), (1, None)] {
            fs::write(devices.join(format!("vfio{n}")), b"").unwrap();
            let device = sysfs.vfio_dev(&format!("vfio{n}")).join("device");
            match target {
                Some(target) => {
                    fs::create_dir_all(device.parent().unwrap()).unwrap();
                    std::os::unix::fs::symlink(target, &device).unwrap();
                }
                None => {
                    fs::create_dir_all(&device).unwrap();
                    fs::write(device.join("vendor"), "0x10de\n").unwrap();
                    fs::write(device.join("device"), "0x2330\n").unwrap();
                    fs::write(device.join("class"), "0x030200\n").unwrap();
                }
            }
        }

        assert!(enumerate_iommufd(root.path(), &sysfs).is_empty());
        assert!(lookup_iommufd_dev("vfio1", root.path(), &sysfs).is_none());
    }

    #[test]
    fn device_name_resolves_from_pci_ids() {
        let root = TempDir::new().unwrap();
        add(root.path(), 0, 0x10de, 0x2330, 0x030200);
        // A device id the database cannot know.
        add(root.path(), 1, 0x10de, 0xdead, 0x030200);

        let devs = enumerate_iommufd(root.path(), &Sysfs::new(root.path()));
        let h100 = devs[0].device_name().expect("10de:2330 must be known");
        assert!(h100.contains("GH100"), "unexpected name: {}", h100);
        assert_eq!(devs[1].device_name(), None);
    }

    #[test]
    fn class_prefix_drops_the_prog_if_byte() {
        let root = TempDir::new().unwrap();
        add(root.path(), 0, 0x10de, 0x2330, 0x030200);

        let devs = enumerate_iommufd(root.path(), &Sysfs::new(root.path()));

        assert_eq!(devs[0].class_prefix(), 0x0302);
    }

    #[test]
    fn missing_sysfs_entry_skipped() {
        let root = TempDir::new().unwrap();
        let devices = root.path().join("devices");
        fs::create_dir_all(&devices).unwrap();
        fs::write(devices.join("vfio0"), b"").unwrap();
        // no sysfs entry — filter_map returns None

        assert!(enumerate_iommufd(root.path(), &Sysfs::new(root.path())).is_empty());
    }

    #[test]
    fn missing_devices_dir_returns_empty() {
        let root = TempDir::new().unwrap();
        assert!(enumerate_iommufd(root.path(), &Sysfs::new(root.path())).is_empty());
    }

    #[test]
    fn passthrough_capable_gpu_and_nvswitch() {
        assert!(is_passthrough_capable_class(0x030200)); // 3D controller (GPU)
        assert!(is_passthrough_capable_class(0x030000)); // VGA
        assert!(is_passthrough_capable_class(0x020000)); // Network controller
        assert!(is_passthrough_capable_class(0x068000)); // NVSwitch (Other Bridge)
    }

    #[test]
    fn passthrough_not_capable_bridges_and_audio() {
        assert!(!is_passthrough_capable_class(0x060000)); // Host Bridge
        assert!(!is_passthrough_capable_class(0x060400)); // PCI-to-PCI Bridge
        assert!(!is_passthrough_capable_class(0x040300)); // Audio device

        // prog-if byte in class should not affect the result
        assert!(!is_passthrough_capable_class(0x060001)); // Host Bridge, non-zero prog-if
    }

    #[test]
    fn lookup_returns_dev_for_known_name() {
        let root = TempDir::new().unwrap();
        add(root.path(), 5, 0x10de, 0x22a3, 0x068000);

        let dev = lookup_iommufd_dev("vfio5", root.path(), &Sysfs::new(root.path()))
            .expect("should find vfio5");
        assert_eq!(dev.num, 5);
        assert_eq!(dev.vendor, 0x10de);
        assert_eq!(dev.device, 0x22a3);
        assert_eq!(dev.class, 0x068000);
        assert!(dev.path.ends_with("devices/vfio5"));
        assert_eq!(dev.bdf, "0000:05:00.0");
    }

    #[test]
    fn lookup_returns_none_for_missing_sysfs() {
        let root = TempDir::new().unwrap();

        assert!(lookup_iommufd_dev("vfio99", root.path(), &Sysfs::new(root.path())).is_none());
    }

    #[test]
    fn lookup_returns_none_when_cdev_absent_but_sysfs_present() {
        let root = TempDir::new().unwrap();
        add(root.path(), 5, 0x10de, 0x22a3, 0x068000);
        fs::remove_file(root.path().join("devices/vfio5")).unwrap();

        assert!(lookup_iommufd_dev("vfio5", root.path(), &Sysfs::new(root.path())).is_none());
    }

    #[test]
    fn lookup_returns_none_for_bad_name() {
        let root = TempDir::new().unwrap();

        assert!(lookup_iommufd_dev("notavfio", root.path(), &Sysfs::new(root.path())).is_none());
    }
}
