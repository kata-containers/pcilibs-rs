// Copyright (c) 2026 Kata Containers contributors
//
// SPDX-License-Identifier: Apache-2.0

//! Fake kernel trees for tests, in a temp directory.
//!
//! Behind the `testfs` feature so a consumer can build the same fixtures its
//! dependency's own tests use, for dev-dependencies only: this writes fake
//! sysfs trees and belongs nowhere near production code.

use std::fs;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use crate::Sysfs;

/// A sysfs tree, kept alive by the handle: dropping it removes the tree.
pub struct Fake {
    pub sysfs: Sysfs,
    root: TempDir,
}

/// The directories the kernel always has, so a test only has to add what it
/// is about.
pub fn fake() -> Fake {
    let root = tempfile::tempdir().unwrap();
    let sysfs = Sysfs::new(root.path());

    fs::create_dir_all(sysfs.devices()).unwrap();
    fs::create_dir_all(sysfs.bus_pci().join("drivers")).unwrap();
    fs::create_dir_all(sysfs.iommu_groups()).unwrap();
    fs::write(sysfs.drivers_probe(), "").unwrap();

    Fake { sysfs, root }
}

impl Fake {
    pub fn root(&self) -> &Path {
        self.root.path()
    }

    pub fn device(&self, address: &str) -> PathBuf {
        self.sysfs.device(address).expect("a PCI address")
    }

    pub fn add_device(&self, address: &str, driver: Option<&str>) {
        let path = self.device(address);
        fs::create_dir_all(&path).unwrap();

        // The kernel links both ways, and names the device by its canonical
        // address on both sides.
        if let Some(driver) = driver {
            let name = path.file_name().expect("a device directory");
            let driver_dir = self.add_driver(driver);
            std::os::unix::fs::symlink(&driver_dir, path.join("driver")).unwrap();
            std::os::unix::fs::symlink(&path, driver_dir.join(name)).unwrap();
        }
    }

    pub fn add_pci_device(
        &self,
        address: &str,
        vendor: u16,
        device: u16,
        class: u32,
        driver: Option<&str>,
    ) {
        self.add_device(address, driver);

        let path = self.device(address);
        fs::write(path.join("vendor"), format!("{vendor:#06x}\n")).unwrap();
        fs::write(path.join("device"), format!("{device:#06x}\n")).unwrap();
        fs::write(path.join("class"), format!("{class:#08x}\n")).unwrap();
        fs::write(path.join("numa_node"), "0\n").unwrap();
    }

    /// Everything [`crate::PciDev`] reads to open a function: identity,
    /// config space, a runtime-PM policy that needs no waking, `reset`, and
    /// a `resource0`.  The BAR0 is sparse, the registers a caller reaches
    /// for being megabytes apart.
    pub fn add_mappable_device(
        &self,
        address: &str,
        vendor: u16,
        device: u16,
        class: u32,
        bar0_len: u64,
    ) {
        self.add_pci_device(address, vendor, device, class, None);

        let path = self.device(address);
        fs::File::create(path.join("resource0"))
            .and_then(|bar0| bar0.set_len(bar0_len))
            .unwrap();
        fs::write(path.join("config"), [0u8; 64]).unwrap();
        fs::write(path.join("reset"), "").unwrap();
        fs::create_dir(path.join("power")).unwrap();
        fs::write(path.join("power/control"), "auto\n").unwrap();
        fs::write(path.join("power/runtime_status"), "active\n").unwrap();
    }

    pub fn set_register(&self, address: &str, offset: u64, value: u32) {
        fs::OpenOptions::new()
            .write(true)
            .open(self.device(address).join("resource0"))
            .and_then(|bar0| bar0.write_all_at(&value.to_le_bytes(), offset))
            .unwrap();
    }

    pub fn add_driver(&self, name: &str) -> PathBuf {
        let path = self.driver(name);
        fs::create_dir_all(&path).unwrap();
        path
    }

    pub fn driver(&self, name: &str) -> PathBuf {
        self.sysfs.driver(name).expect("a driver name")
    }

    /// A loaded module, as `/sys/module/<module>/drivers/pci:<driver>`.
    pub fn add_module(&self, module: &str, driver: &str) {
        let path = self
            .sysfs
            .module(module)
            .expect("a module name")
            .join("drivers")
            .join(format!("pci:{driver}"));
        fs::create_dir_all(path).unwrap();
    }

    pub fn add_iommu_group(&self, group: u32) -> PathBuf {
        let path = self.sysfs.iommu_groups().join(group.to_string());
        fs::create_dir_all(&path).unwrap();
        path
    }

    pub fn set_iommu_group(&self, address: &str, group: u32) {
        let path = self.add_iommu_group(group);
        std::os::unix::fs::symlink(path, self.device(address).join("iommu_group")).unwrap();
    }

    pub fn add_infiniband(&self, name: &str, address: &str, node_type: &str, fw_ver: &str) {
        let path = self.sysfs.infiniband().join(name);
        fs::create_dir_all(&path).unwrap();
        std::os::unix::fs::symlink(self.device(address), path.join("device")).unwrap();
        fs::write(path.join("node_type"), format!("{node_type}\n")).unwrap();
        fs::write(path.join("fw_ver"), format!("{fw_ver}\n")).unwrap();
    }

    /// `dev` is the cdev's `<major>:<minor>`, as sysfs prints it.
    pub fn add_infiniband_verbs(&self, name: &str, ibdev: &str, dev: &str) {
        let path = self.sysfs.infiniband_verbs().join(name);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("ibdev"), format!("{ibdev}\n")).unwrap();
        fs::write(path.join("dev"), format!("{dev}\n")).unwrap();
    }

    pub fn driver_override(&self, address: &str) -> String {
        read(self.device(address).join("driver_override"))
    }

    pub fn drivers_probe(&self) -> String {
        read(self.sysfs.drivers_probe())
    }

    pub fn driver_unbind(&self, driver: &str) -> String {
        read(self.driver(driver).join("unbind"))
    }
}

fn read(path: PathBuf) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

/// Add one fake cdev `vfio<n>` with the given sysfs `vendor`, `device`, and
/// `class` contents (as sysfs prints them, e.g. "0x10de", "0x2330",
/// "0x030200").
///
/// `root` is both the `/dev/vfio` and the sysfs root: the cdev lands at
/// `<root>/devices/vfio<n>` and its identity under `<root>/class/vfio-dev/`,
/// so a caller passes `root` and `Sysfs::new(root)` to the same tree.
pub fn add(root: &Path, n: u32, vendor: &str, device: &str, class: &str) {
    let devices = root.join("devices");
    fs::create_dir_all(&devices).unwrap();
    fs::write(devices.join(format!("vfio{n}")), b"").unwrap();
    let dev_dir = Sysfs::new(root)
        .vfio_dev(&format!("vfio{n}"))
        .join("device");
    fs::create_dir_all(&dev_dir).unwrap();
    fs::write(dev_dir.join("vendor"), format!("{vendor}\n")).unwrap();
    fs::write(dev_dir.join("device"), format!("{device}\n")).unwrap();
    fs::write(dev_dir.join("class"), format!("{class}\n")).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sysfs spells the device the same on both sides of the link.
    #[test]
    fn links_a_device_by_its_canonical_address() {
        let fake = fake();
        fake.add_device("65:00.0", Some("vfio-pci"));

        let driver = fake.driver("vfio-pci");
        assert!(driver.join("0000:65:00.0").is_symlink());
        assert!(!driver.join("65:00.0").exists());
    }
}
