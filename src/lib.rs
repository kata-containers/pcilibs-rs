// Copyright (c) Ant Group
//
// SPDX-License-Identifier: Apache-2.0
//

mod iommufd;
mod pci_ids;
mod pci_manager;
mod sysfs;
#[cfg(any(test, feature = "testfs"))]
pub mod testfs;
pub mod vfio;

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use nix::sys::stat::{self, SFlag};

pub use iommufd::{
    enumerate_iommufd, is_passthrough_capable_class, lookup_iommufd_dev, IommufdDev,
    IOMMUFD_VFIO_DIR,
};
pub use pci_manager::{is_pcie_device, PCIDevice, PCIDeviceManager};
pub use sysfs::{Sysfs, SYSFS};

pub(crate) fn failed(kind: std::io::ErrorKind, message: String) -> std::io::Error {
    std::io::Error::new(kind, message)
}

/// Keeps the kind, so a caller can still tell a missing device from a
/// permission problem, and prepends what was being attempted.
pub(crate) fn context(err: std::io::Error, what: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(err.kind(), format!("{what}: {err}"))
}

/// The PCI domain sysfs always spells out, and callers often omit.
pub const PCI_DEV_DOMAIN: &str = "0000";

/// `65:00.0` and `0000:65:00.0` name the same device; sysfs only answers to
/// the second.
///
/// Rebuilt from the parsed numbers rather than patched up, because the result
/// is joined onto a sysfs root: none of the caller's string may reach a path.
pub fn normalize_bdf(bdf: &str) -> Option<String> {
    let fields: Vec<&str> = bdf.split(':').collect();
    let (domain, bus, slot) = match fields[..] {
        [bus, slot] => (PCI_DEV_DOMAIN, bus, slot),
        [domain, bus, slot] => (domain, bus, slot),
        _ => return None,
    };
    let (device, function) = slot.split_once('.')?;

    let domain = u16::from_str_radix(domain, 16).ok()?;
    let bus = u8::from_str_radix(bus, 16).ok()?;
    let device = u8::from_str_radix(device, 16).ok()?;
    let function = u8::from_str_radix(function, 16).ok()?;

    Some(format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}"))
}

/// Device driver for vfio-pci guest kernel driver.
pub const DRIVER_VFIO_PCI_GK_TYPE: &str = "vfio-pci-gk";
/// Device driver for vfio-pci.
pub const DRIVER_VFIO_PCI_TYPE: &str = "vfio-pci";
/// Device driver for vfio-ap hotplug.
pub const DRIVER_VFIO_AP_TYPE: &str = "vfio-ap";
/// Device driver for vfio-ap coldplug.
pub const DRIVER_VFIO_AP_COLD_TYPE: &str = "vfio-ap-cold";

pub fn is_vfio_device_type(device_type: &str) -> bool {
    matches!(
        device_type,
        DRIVER_VFIO_PCI_TYPE
            | DRIVER_VFIO_PCI_GK_TYPE
            | DRIVER_VFIO_AP_TYPE
            | DRIVER_VFIO_AP_COLD_TYPE
    )
}

/// Root of the InfiniBand character device tree.  Devfs rather than sysfs,
/// so it stays a path of its own.
pub const INFINIBAND_DEV_DIR: &str = "/dev/infiniband";

fn device_kind(mode: u32) -> &'static str {
    let kind = SFlag::from_bits_truncate(mode) & SFlag::S_IFMT;
    if kind == SFlag::S_IFCHR {
        "char"
    } else if kind == SFlag::S_IFBLK {
        "block"
    } else {
        "other"
    }
}

/// One-line summary of every InfiniBand device the guest kernel currently
/// exposes, plus every device node under `dev_dir` and the PCI BDF backing
/// each IB device.
///
/// Pure sysfs / devfs reads — no agent-specific dependencies.
/// Used as a diagnostic context string in log calls, so it never fails: a
/// missing tree is itself the diagnosis.
pub fn snapshot_infiniband(dev_dir: &Path, sysfs: &Sysfs) -> String {
    let mut ib_parts: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(sysfs.infiniband()) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            let pci_bdf = fs::read_link(path.join("device"))
                .ok()
                .and_then(|t| t.file_name().map(|n| n.to_string_lossy().into_owned()))
                .unwrap_or_else(|| "<none>".to_string());
            let node_type = fs::read_to_string(path.join("node_type"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let fw = fs::read_to_string(path.join("fw_ver"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            ib_parts.push(format!(
                "{name}=[bdf={pci_bdf},node_type={node_type:?},fw={fw}]"
            ));
        }
    }

    let mut verbs_parts: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(sysfs.infiniband_verbs()) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("uverbs") {
                continue;
            }
            let path = entry.path();
            let ibdev = fs::read_to_string(path.join("ibdev"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let dev = fs::read_to_string(path.join("dev"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            verbs_parts.push(format!("{name}=[ibdev={ibdev},dev={dev}]"));
        }
    }

    let mut chardev_parts: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(dev_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let kind = device_kind(metadata.mode());
            let rdev = metadata.rdev();
            let major = stat::major(rdev);
            let minor = stat::minor(rdev);
            chardev_parts.push(format!("{name}=[{kind},{major}:{minor}]"));
        }
    }

    format!(
        "ib_devices=[{}] uverbs=[{}] chardevs=[{}]",
        ib_parts.join(", "),
        verbs_parts.join(", "),
        chardev_parts.join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::already_qualified("0000:65:00.0", "0000:65:00.0")]
    #[case::domain_omitted("65:00.0", "0000:65:00.0")]
    #[case::non_zero_domain("0009:01:00.0", "0009:01:00.0")]
    #[case::upper_case("0009:01:00.0", "0009:01:00.0")]
    #[case::unpadded("9:1:0.0", "0009:01:00.0")]
    fn normalize_bdf_canonicalises_an_address(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(normalize_bdf(input).unwrap(), expected);
    }

    #[rstest::rstest]
    #[case::traversal("../../../etc/shadow")]
    #[case::traversal_shaped_like_an_address("0000:../:00.0")]
    #[case::absolute("/etc/shadow")]
    #[case::separator_in_a_field("0000:65:00.0/../..")]
    #[case::too_few_fields("65")]
    #[case::too_many_fields("0000:0000:65:00.0")]
    #[case::no_function("0000:65:00")]
    #[case::not_hex("zzzz:65:00.0")]
    #[case::empty("")]
    fn normalize_bdf_refuses_what_is_not_an_address(#[case] input: &str) {
        assert!(normalize_bdf(input).is_none(), "{input:?} was accepted");
    }

    #[test]
    fn test_is_vfio_device_type() {
        assert!(is_vfio_device_type(DRIVER_VFIO_PCI_TYPE));
        assert!(is_vfio_device_type(DRIVER_VFIO_PCI_GK_TYPE));
        assert!(is_vfio_device_type(DRIVER_VFIO_AP_TYPE));
        assert!(is_vfio_device_type(DRIVER_VFIO_AP_COLD_TYPE));
        assert!(!is_vfio_device_type("virtio-pci"));
    }

    /// A test cannot mknod a device node, so the mode bits stand in for what
    /// `stat` would have answered.
    #[rstest::rstest]
    #[case::character_device(SFlag::S_IFCHR, "char")]
    #[case::block_device(SFlag::S_IFBLK, "block")]
    #[case::regular_file(SFlag::S_IFREG, "other")]
    #[case::directory(SFlag::S_IFDIR, "other")]
    #[case::fifo(SFlag::S_IFIFO, "other")]
    fn a_device_node_is_named_by_its_type(#[case] kind: SFlag, #[case] expected: &str) {
        assert_eq!(device_kind(kind.bits() | 0o600), expected);
    }

    /// Log context, so no RDMA has to read as "nothing here" rather than
    /// fail the caller.
    #[test]
    fn a_snapshot_of_a_guest_without_rdma_lists_nothing() {
        let fake = testfs::fake();

        assert_eq!(
            snapshot_infiniband(Path::new("/no/such/dev/infiniband"), &fake.sysfs),
            "ib_devices=[] uverbs=[] chardevs=[]"
        );
    }

    #[test]
    fn a_snapshot_names_a_device_after_the_pci_function_behind_it() {
        let fake = testfs::fake();
        fake.add_device("0000:03:00.0", Some("mlx5_core"));
        fake.add_infiniband("mlx5_0", "03:00.0", "1: CA", "28.43.2026");
        fake.add_infiniband_verbs("uverbs0", "mlx5_0", "231:192");

        let snapshot = snapshot_infiniband(Path::new("/no/such/dev/infiniband"), &fake.sysfs);

        assert!(
            snapshot.contains(r#"mlx5_0=[bdf=0000:03:00.0,node_type="1: CA",fw=28.43.2026]"#),
            "{snapshot}"
        );
        assert!(
            snapshot.contains("uverbs0=[ibdev=mlx5_0,dev=231:192]"),
            "{snapshot}"
        );
    }

    /// A device whose attributes will not read is the diagnosis, so it gets
    /// reported rather than dropped.
    #[test]
    fn a_snapshot_lists_a_device_it_could_not_read() {
        let fake = testfs::fake();
        fs::create_dir_all(fake.sysfs.infiniband().join("mlx5_0")).unwrap();
        fs::create_dir_all(fake.sysfs.infiniband_verbs().join("uverbs0")).unwrap();

        let snapshot = snapshot_infiniband(Path::new("/no/such/dev/infiniband"), &fake.sysfs);

        assert!(
            snapshot.contains(r#"mlx5_0=[bdf=<none>,node_type="",fw=]"#),
            "{snapshot}"
        );
        assert!(snapshot.contains("uverbs0=[ibdev=,dev=]"), "{snapshot}");
    }

    /// `infiniband_verbs` holds an `abi_version` file beside the devices,
    /// which would otherwise be reported as a device with nothing readable.
    #[test]
    fn a_snapshot_skips_what_is_not_a_verbs_device() {
        let fake = testfs::fake();
        let verbs = fake.sysfs.infiniband_verbs();
        fs::create_dir_all(&verbs).unwrap();
        fs::write(verbs.join("abi_version"), "6\n").unwrap();

        let snapshot = snapshot_infiniband(Path::new("/no/such/dev/infiniband"), &fake.sysfs);

        assert_eq!(snapshot, "ib_devices=[] uverbs=[] chardevs=[]");
    }

    /// Against the real `/dev`, since a test cannot create a device node:
    /// `/dev/null` is a character device at 1:3 on every Linux kernel.
    #[test]
    fn a_snapshot_reports_a_device_node_by_major_and_minor() {
        let fake = testfs::fake();

        let snapshot = snapshot_infiniband(Path::new("/dev"), &fake.sysfs);

        assert!(snapshot.contains("null=[char,1:3]"), "{snapshot}");
    }

    #[test]
    fn a_snapshot_reports_what_is_not_a_device_node() {
        let fake = testfs::fake();
        let dev_dir = fake.root().join("dev/infiniband");
        fs::create_dir_all(&dev_dir).unwrap();
        fs::write(dev_dir.join("uverbs0"), "").unwrap();

        let snapshot = snapshot_infiniband(&dev_dir, &fake.sysfs);

        assert!(
            snapshot.contains("chardevs=[uverbs0=[other,0:0]]"),
            "{snapshot}"
        );
    }
}
