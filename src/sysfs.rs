// Copyright (c) 2026 Kata Containers contributors
//
// SPDX-License-Identifier: Apache-2.0

//! Sysfs paths, off an injectable mount point so tests can point them at a
//! temp tree rather than the running kernel.

use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

use crate::normalize_bdf;

pub const SYSFS: &str = "/sys";

/// One ordinary component, so joining it cannot leave the root.
fn is_component(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(one)) if one == OsStr::new(name))
        && components.next().is_none()
}

#[derive(Clone, Debug)]
pub struct Sysfs {
    root: PathBuf,
}

impl Default for Sysfs {
    fn default() -> Self {
        Self::new(Path::new(SYSFS))
    }
}

impl Sysfs {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }

    pub fn bus_pci(&self) -> PathBuf {
        self.root.join("bus/pci")
    }

    pub fn devices(&self) -> PathBuf {
        self.bus_pci().join("devices")
    }

    /// `None` unless the address is one, since this is where it becomes a
    /// path.
    pub fn device(&self, address: &str) -> Option<PathBuf> {
        Some(self.devices().join(normalize_bdf(address)?))
    }

    /// Preferred over a driver's own `bind`, which ignores `driver_override`.
    pub fn drivers_probe(&self) -> PathBuf {
        self.bus_pci().join("drivers_probe")
    }

    /// Absent unless the driver is registered, i.e. its module is loaded.
    ///
    /// `None` unless the name is one, since this is where it becomes a path.
    pub fn driver(&self, name: &str) -> Option<PathBuf> {
        is_component(name).then(|| self.bus_pci().join("drivers").join(name))
    }

    /// Where a loaded module lists the drivers it registers.  Takes either
    /// spelling: modprobe answers to `vfio-pci`, sysfs only to `vfio_pci`.
    ///
    /// `None` unless the name is one, since this is where it becomes a path.
    pub fn module(&self, name: &str) -> Option<PathBuf> {
        let name = name.replace('-', "_");
        is_component(&name).then(|| self.root.join("module").join(name))
    }

    /// Empty when the IOMMU is off.  Beats the kernel command line, where
    /// the option is architecture-specific and often implicit.
    pub fn iommu_groups(&self) -> PathBuf {
        self.root.join("kernel/iommu_groups")
    }

    fn class(&self, name: &str) -> PathBuf {
        self.root.join("class").join(name)
    }

    /// `<name>/device` links to the PCI function behind a vfio character
    /// device.
    pub fn vfio_dev(&self, name: &str) -> PathBuf {
        self.class("vfio-dev").join(name)
    }

    pub fn infiniband(&self) -> PathBuf {
        self.class("infiniband")
    }

    /// Only the devices userspace can open verbs on, which is not every
    /// InfiniBand device: a separate tree, not a subset of one.
    pub fn infiniband_verbs(&self) -> PathBuf {
        self.class("infiniband_verbs")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;

    #[test]
    fn defaults_to_the_running_kernel() {
        assert_eq!(
            Sysfs::default().devices(),
            Path::new("/sys/bus/pci/devices")
        );
    }

    #[test]
    fn derives_every_path_from_the_root_it_was_given() {
        let sysfs = Sysfs::new(Path::new("/tmp/fake"));

        assert_eq!(sysfs.devices(), Path::new("/tmp/fake/bus/pci/devices"));
    }

    #[rstest]
    #[case::canonical("0000:65:00.0")]
    #[case::no_domain("65:00.0")]
    #[case::upper_case("0000:6F:00.0")]
    #[case::unpadded("0:6:0.0")]
    fn spells_an_address_the_way_sysfs_does(#[case] address: &str) {
        let path = Sysfs::default().device(address).expect("a PCI address");

        assert_eq!(path.parent(), Some(Path::new("/sys/bus/pci/devices")));
        assert!(
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().len() == "0000:00:00.0".len()),
            "{path:?}"
        );
    }

    #[rstest]
    #[case::traversal("../../../etc")]
    #[case::absolute("/etc/shadow")]
    #[case::separator("0000:65:00.0/..")]
    #[case::not_hex("zzzz:65:00.0")]
    #[case::empty("")]
    fn refuses_an_address_that_is_not_one(#[case] address: &str) {
        assert!(Sysfs::default().device(address).is_none());
    }

    /// Hyphens kept, unlike a module: this is the name the kernel registered.
    #[rstest]
    #[case::vfio_pci("vfio-pci", true)]
    #[case::traversal("../../../etc", false)]
    #[case::absolute("/etc", false)]
    #[case::separator("vfio-pci/..", false)]
    #[case::empty("", false)]
    fn takes_a_driver_name_and_not_a_path(#[case] name: &str, #[case] accepted: bool) {
        let path = Sysfs::new(Path::new("/tmp/fake")).driver(name);

        assert_eq!(path.is_some(), accepted, "{name:?}");
        if let Some(path) = path {
            assert_eq!(path, Path::new("/tmp/fake/bus/pci/drivers").join(name));
        }
    }

    #[rstest]
    #[case::underscored("nvgrace_gpu_vfio_pci", true)]
    #[case::hyphenated("nvgrace-gpu-vfio-pci", true)]
    #[case::traversal("../../target", false)]
    #[case::absolute("/etc", false)]
    #[case::separator("vfio_pci/../..", false)]
    #[case::parent("..", false)]
    #[case::current(".", false)]
    #[case::empty("", false)]
    fn takes_a_module_name_and_not_a_path(#[case] name: &str, #[case] accepted: bool) {
        let path = Sysfs::new(Path::new("/tmp/fake")).module(name);

        assert_eq!(path.is_some(), accepted, "{name:?}");
        if let Some(path) = path {
            assert_eq!(path.parent(), Some(Path::new("/tmp/fake/module")));
        }
    }
}
