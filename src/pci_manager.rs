// Copyright (c) Ant Group
//
// SPDX-License-Identifier: Apache-2.0
//
#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;

use crate::pci_ids::{Class, Device, FromId};

use crate::{normalize_bdf, PciDev, Sysfs};

const PCI_CONFIG_SPACE_SZ: u64 = 256;

const UNKNOWN_DEVICE: &str = "UNKNOWN_DEVICE";
const UNKNOWN_CLASS: &str = "UNKNOWN_CLASS";

fn address_to_id(address: &str) -> u64 {
    let cleaned_address = address.replace(":", "").replace(".", "");
    u64::from_str_radix(&cleaned_address, 16).unwrap_or(0)
}

#[derive(Clone, Debug, Default)]
pub struct PCIDevice {
    pub device_path: PathBuf,
    pub address: String,
    pub vendor: u16,
    pub class: u32,
    pub class_name: String,
    pub device: u16,
    pub device_name: String,
    pub driver: String,
    pub iommu_group: i64,
    pub numa_node: i64,
    /// Kept so a device opens under the tree it was found in.
    sysfs: Sysfs,
}

impl PCIDevice {
    /// Map this device's BAR0 for register access. Enumeration reads
    /// attributes and needs no privilege; this needs root and wakes the
    /// device if it is suspended.
    pub fn open(&self) -> io::Result<PciDev> {
        PciDev::open_in(&self.sysfs, &self.address)
    }
}

#[derive(Clone, Debug, Default)]
pub struct PCIDeviceManager {
    sysfs: Sysfs,
}

impl PCIDeviceManager {
    pub fn new(sysfs: Sysfs) -> Self {
        PCIDeviceManager { sysfs }
    }

    pub fn get_all_devices(&self, vendor: Option<u16>) -> io::Result<Vec<PCIDevice>> {
        let mut pci_devices = Vec::new();
        let device_dirs = fs::read_dir(self.sysfs.devices())?;

        let mut cache: HashMap<String, PCIDevice> = HashMap::new();

        for entry in device_dirs {
            let device_dir = entry?;
            let device_address = device_dir.file_name().to_string_lossy().to_string();
            if let Ok(Some(dev)) =
                self.get_device_by_pci_bus_id(&device_address, vendor, &mut cache)
            {
                pci_devices.push(dev);
            }
        }

        pci_devices.sort_by_key(|dev| address_to_id(&dev.address));

        Ok(pci_devices)
    }

    pub fn get_device_by_pci_bus_id(
        &self,
        address: &str,
        vendor: Option<u16>,
        cache: &mut HashMap<String, PCIDevice>,
    ) -> io::Result<Option<PCIDevice>> {
        let address = normalize_bdf(address).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{address:?} is not a PCI address"),
            )
        })?;

        if let Some(device) = cache.get(&address) {
            return Ok(Some(device.clone()));
        }

        let device_path = self.sysfs.devices().join(&address);

        // read vendor ID
        let vendor_str = fs::read_to_string(device_path.join("vendor"))?;
        let vendor_id = u16::from_str_radix(vendor_str.trim().trim_start_matches("0x"), 16)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if let Some(vend_id) = vendor {
            if vendor_id != vend_id {
                return Ok(None);
            }
        }

        let class_str = fs::read_to_string(device_path.join("class"))?;
        let class_id = u32::from_str_radix(class_str.trim().trim_start_matches("0x"), 16).unwrap();

        let device_str = fs::read_to_string(device_path.join("device"))?;
        let device_id =
            u16::from_str_radix(device_str.trim().trim_start_matches("0x"), 16).unwrap();

        let driver = match fs::read_link(device_path.join("driver")) {
            Ok(path) => path.file_name().unwrap().to_string_lossy().to_string(),
            Err(_) => String::new(),
        };

        let iommu_group = match fs::read_link(device_path.join("iommu_group")) {
            Ok(path) => path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
                .parse::<i64>()
                .unwrap_or(-1),
            Err(_) => -1,
        };

        let numa_node = fs::read_to_string(device_path.join("numa_node"))
            .map(|numa| numa.trim().parse::<i64>().unwrap_or(-1))
            .unwrap_or(-1);

        let device_name = Device::from_vid_pid(vendor_id, device_id)
            .map_or(UNKNOWN_DEVICE, |device| device.name())
            .to_owned();

        // sysfs prints the whole 24-bit class code; the database is keyed on
        // the base class in its top byte.
        let class_name = Class::from_id((class_id >> 16) as u8)
            .map_or(UNKNOWN_CLASS, |class| class.name())
            .to_owned();

        let pci_device = PCIDevice {
            device_path,
            address: address.clone(),
            vendor: vendor_id,
            class: class_id,
            device: device_id,
            driver,
            iommu_group,
            numa_node,
            device_name,
            class_name,
            sysfs: self.sysfs.clone(),
        };

        cache.insert(address, pci_device.clone());

        Ok(Some(pci_device))
    }
}

/// A PCIe function's config space is larger than a conventional PCI one.
pub fn is_pcie_device(bdf: &str, sysfs: &Sysfs) -> bool {
    let Some(device) = sysfs.device(bdf) else {
        return false;
    };

    match fs::metadata(device.join("config")) {
        Ok(metadata) => metadata.len() > PCI_CONFIG_SPACE_SZ,
        // Error reading the file, assume it's not a PCIe device
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs::{self, Fake};

    use rstest::{fixture, rstest};

    /// An e1000 as a QEMU guest sees one: the database knows both its ids
    /// and its class.
    const E1000: (u16, u16, u32) = (0x8086, 0x100e, 0x020000);

    #[fixture]
    fn fake() -> Fake {
        testfs::fake()
    }

    fn manager(fake: &Fake) -> PCIDeviceManager {
        PCIDeviceManager::new(fake.sysfs.clone())
    }

    #[rstest]
    fn enumeration_reads_a_device_and_names_it(fake: Fake) {
        let (vendor, device, class) = E1000;
        fake.add_pci_device("0000:ff:1f.0", vendor, device, class, Some("igb"));

        let devices = manager(&fake).get_all_devices(None).unwrap();

        assert_eq!(devices.len(), 1);
        let found = &devices[0];
        assert_eq!(found.address, "0000:ff:1f.0");
        assert_eq!(found.device_path, fake.device("0000:ff:1f.0"));
        assert_eq!(found.vendor, vendor);
        assert_eq!(found.device, device);
        assert_eq!(found.class, class);
        assert_eq!(found.device_name, "82540EM Gigabit Ethernet Controller");
        assert_eq!(found.class_name, "Network controller");
        assert_eq!(found.driver, "igb");
        assert_eq!(found.numa_node, 0);
    }

    /// A device the database has not heard of is still a device the caller
    /// asked about, so it comes back unknown rather than dropped.
    #[rstest]
    fn enumeration_reports_a_device_the_database_does_not_know(fake: Fake) {
        fake.add_pci_device("0000:65:00.0", 0xeeee, 0xeeee, 0x200000, None);

        let devices = manager(&fake).get_all_devices(None).unwrap();

        assert_eq!(devices[0].device_name, UNKNOWN_DEVICE);
        assert_eq!(devices[0].class_name, UNKNOWN_CLASS);
        assert_eq!(devices[0].driver, "");
    }

    /// The base class, not the whole 24-bit code: comparing the two is what
    /// used to leave every device UNKNOWN_CLASS.
    #[rstest]
    fn a_class_name_comes_from_the_base_class(fake: Fake) {
        fake.add_pci_device("0000:03:00.0", 0x10de, 0x2330, 0x030200, None);

        let devices = manager(&fake).get_all_devices(None).unwrap();

        assert_eq!(devices[0].class_name, "Display controller");
        assert_eq!(devices[0].device_name, "GH100 [H100 SXM5 80GB]");
    }

    /// Bus order, not the order the directory happened to list them in.
    #[rstest]
    fn enumeration_sorts_by_address(fake: Fake) {
        let (vendor, device, class) = E1000;
        for address in [
            "0000:ff:1f.0",
            "0001:00:00.0",
            "0000:03:00.1",
            "0000:03:00.0",
        ] {
            fake.add_pci_device(address, vendor, device, class, None);
        }

        let devices = manager(&fake).get_all_devices(None).unwrap();

        assert_eq!(
            devices
                .iter()
                .map(|d| d.address.as_str())
                .collect::<Vec<_>>(),
            [
                "0000:03:00.0",
                "0000:03:00.1",
                "0000:ff:1f.0",
                "0001:00:00.0"
            ]
        );
    }

    #[rstest]
    fn a_vendor_filter_keeps_only_that_vendor(fake: Fake) {
        let (vendor, device, class) = E1000;
        fake.add_pci_device("0000:03:00.0", 0x10de, 0x2330, 0x030200, None);
        fake.add_pci_device("0000:65:00.0", vendor, device, class, None);

        let devices = manager(&fake).get_all_devices(Some(0x10de)).unwrap();

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].address, "0000:03:00.0");
    }

    #[rstest]
    fn a_device_reports_the_iommu_group_it_is_in(fake: Fake) {
        fake.add_pci_device("0000:03:00.0", 0x10de, 0x2330, 0x030200, Some("vfio-pci"));
        fake.set_iommu_group("0000:03:00.0", 17);

        let devices = manager(&fake).get_all_devices(None).unwrap();

        assert_eq!(devices[0].iommu_group, 17);
    }

    /// The IOMMU being off is not an error: no group is what tells a caller
    /// the device cannot be passed through.
    #[rstest]
    fn a_device_outside_an_iommu_group_reports_none(fake: Fake) {
        let (vendor, device, class) = E1000;
        fake.add_pci_device("0000:03:00.0", vendor, device, class, None);

        let devices = manager(&fake).get_all_devices(None).unwrap();

        assert_eq!(devices[0].iommu_group, -1);
    }

    /// Removing the tree is what proves the second read never reached
    /// sysfs; the two spellings prove one device is one entry.
    #[rstest]
    fn a_second_lookup_of_a_device_comes_from_the_cache(fake: Fake) {
        fake.add_pci_device("0000:03:00.0", 0x10de, 0x2330, 0x030200, None);
        let manager = manager(&fake);
        let mut cache = HashMap::new();

        let first = manager
            .get_device_by_pci_bus_id("03:00.0", None, &mut cache)
            .unwrap()
            .expect("the device should be found");
        fs::remove_dir_all(fake.device("0000:03:00.0")).unwrap();
        let cached = manager
            .get_device_by_pci_bus_id("0000:03:00.0", None, &mut cache)
            .unwrap()
            .expect("the device should still be found");

        assert_eq!(cache.len(), 1);
        assert_eq!(cached.address, first.address);
        assert_eq!(cached.device_name, first.device_name);
    }

    /// Enumeration and direct lookup have to agree on one spelling, or the
    /// cache keys and the addresses handed back drift apart.
    #[rstest]
    fn a_lookup_canonicalises_the_address_it_reports(fake: Fake) {
        let (vendor, device, class) = E1000;
        fake.add_pci_device("0000:ff:1f.0", vendor, device, class, None);

        let found = manager(&fake)
            .get_device_by_pci_bus_id("FF:1F.0", None, &mut HashMap::new())
            .unwrap()
            .expect("the device should be found");

        assert_eq!(found.address, "0000:ff:1f.0");
    }

    /// The handle has to be opened under the root the device was enumerated
    /// from, or a caller pointed at a test tree would silently read /sys.
    #[rstest]
    fn open_looks_under_the_root_the_device_came_from(fake: Fake) {
        let (vendor, device, class) = E1000;
        fake.add_pci_device("0000:03:00.0", vendor, device, class, None);
        let manager = manager(&fake);
        let device = &manager.get_all_devices(None).unwrap()[0];

        // No resource0 in the fake tree, so this cannot succeed — but the
        // error must name the fake root, not /sys/bus/pci/devices.
        let err = device.open().unwrap_err().to_string();

        assert!(err.contains(&*fake.root().to_string_lossy()), "{err}");
    }

    /// A lookup joins its argument onto the sysfs root, so anything that is
    /// not a PCI address has to be refused rather than followed.
    #[rstest]
    #[case::traversal("../../../etc/shadow")]
    #[case::separator("0000:ff:1f.0/../../..")]
    #[case::absolute("/etc/shadow")]
    #[case::nonsense("nonsense")]
    fn a_lookup_refuses_an_address_that_is_not_one(fake: Fake, #[case] address: &str) {
        let err = manager(&fake)
            .get_device_by_pci_bus_id(address, None, &mut HashMap::new())
            .unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    /// A vendor id that will not parse means the tree is not sysfs, which
    /// beats handing back a device with a made-up vendor.
    #[rstest]
    fn a_lookup_refuses_a_vendor_id_it_cannot_parse(fake: Fake) {
        fake.add_device("0000:03:00.0", None);
        fs::write(fake.device("0000:03:00.0").join("vendor"), "nonsense\n").unwrap();

        let err = manager(&fake)
            .get_device_by_pci_bus_id("0000:03:00.0", None, &mut HashMap::new())
            .unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[rstest]
    fn a_lookup_of_a_device_that_is_not_there_fails(fake: Fake) {
        let err = manager(&fake)
            .get_device_by_pci_bus_id("0000:03:00.0", None, &mut HashMap::new())
            .unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    /// Exactly 256 bytes is a conventional PCI function; the boundary is
    /// the whole test.
    #[rstest]
    #[case::pcie(PCI_CONFIG_SPACE_SZ + 1, true)]
    #[case::conventional_pci(PCI_CONFIG_SPACE_SZ, false)]
    fn a_pcie_function_has_an_extended_config_space(
        fake: Fake,
        #[case] size: u64,
        #[case] expected: bool,
    ) {
        fake.add_device("0000:ff:00.0", None);
        fs::write(
            fake.device("0000:ff:00.0").join("config"),
            vec![0; size as usize],
        )
        .unwrap();

        assert_eq!(is_pcie_device("ff:00.0", &fake.sysfs), expected);
    }

    #[rstest]
    #[case::no_config_space("0000:ff:00.0")]
    #[case::no_such_device("0000:03:00.0")]
    #[case::not_an_address("../../../etc")]
    fn without_a_config_space_a_device_is_not_pcie(fake: Fake, #[case] address: &str) {
        fake.add_device("0000:ff:00.0", None);

        assert!(!is_pcie_device(address, &fake.sysfs));
    }
}
