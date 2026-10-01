// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

use super::{Catalog, PciIdentity, Properties};
use crate::{attr_hex, context, failed, normalize_bdf, Sysfs};
use std::{fs, io};

/// A PCI function and optional catalog metadata. Unknown vendors remain visible.
#[derive(Debug, PartialEq, Eq)]
pub struct Device<'a> {
    pub bdf: String,
    pub identity: PciIdentity,
    pub class: u32,
    pub properties: Option<Properties<'a>>,
}

/// Enumerate PCI functions for every vendor, without drivers, VPD or BAR access.
/// Results are sorted by BDF; unreadable identities are errors.
pub fn discover<'a>(sysfs: &Sysfs, catalog: Catalog<'a>) -> io::Result<Vec<Device<'a>>> {
    let root = sysfs.devices();
    let mut devices = Vec::new();
    for entry in fs::read_dir(&root).map_err(|e| context(e, root.display()))? {
        let entry = entry?;
        let Some(bdf) = entry.file_name().to_str().and_then(normalize_bdf) else {
            continue;
        };
        let path = root.join(&bdf);
        let read_id = |name| {
            u16::try_from(attr_hex(&path, name)?).map_err(|_| {
                failed(
                    io::ErrorKind::InvalidData,
                    format!("{}: {name} exceeds 16 bits", path.display()),
                )
            })
        };
        let identity = PciIdentity::new(
            read_id("vendor")?,
            read_id("device")?,
            read_id("subsystem_vendor")?,
            read_id("subsystem_device")?,
        );
        devices.push(Device {
            bdf,
            identity,
            class: attr_hex(&path, "class")?,
            properties: catalog.lookup(identity),
        });
    }
    devices.sort_by(|a, b| a.bdf.cmp(&b.bdf));
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        catalog::{CatalogFile, DeviceKind},
        testfs,
    };

    fn device(fake: &testfs::Fake, bdf: &str, vendor: u16, id: u16, class: u32) {
        fake.add_pci_device(bdf, vendor, id, class, None);
        fs::write(fake.device(bdf).join("subsystem_vendor"), "0x1234").unwrap();
        fs::write(fake.device(bdf).join("subsystem_device"), "0x0001").unwrap();
    }

    #[test]
    fn all_vendors_and_unknown_devices_remain_visible() {
        let fake = testfs::fake();
        device(&fake, "0000:03:00.0", 0x8086, 0xffff, 0x020000);
        device(&fake, "0000:01:00.0", 0x15b3, 0x1021, 0x020700);
        device(&fake, "0000:02:00.0", 0x15b3, 0xd2f4, 0x060000);
        fs::create_dir(fake.sysfs.devices().join("not-a-device")).unwrap();
        let devices = discover(&fake.sysfs, Catalog::builtin()).unwrap();
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0].bdf, "0000:01:00.0");
        assert_eq!(devices[0].properties.unwrap().kind, DeviceKind::Nic);
        assert_eq!(devices[1].properties.unwrap().profile, "Quantum-3");
        assert!(devices[2].properties.is_none());
        assert_eq!(devices[2].identity.vendor, 0x8086);
    }

    #[test]
    fn external_vendor_records_reach_discovery() {
        let fake = testfs::fake();
        device(&fake, "0000:01:00.0", 0x1234, 0x5678, 0x020000);
        fs::write(
            fake.root().join("devices.catalog"),
            "1234 5678 1234 0001 nic FutureNIC pcie\n8086 ffff * * other FutureDevice unknown",
        )
        .unwrap();
        let loaded = CatalogFile::read(fake.root()).unwrap();
        let catalog = loaded.catalog();
        assert_eq!(
            discover(&fake.sysfs, catalog).unwrap()[0]
                .properties
                .unwrap()
                .profile,
            "FutureNIC"
        );
        fs::write(
            fake.device("0000:01:00.0").join("subsystem_vendor"),
            "0x5678",
        )
        .unwrap();
        assert!(discover(&fake.sysfs, catalog).unwrap()[0]
            .properties
            .is_none());
    }

    #[test]
    fn unreadable_or_out_of_range_identity_is_an_error() {
        let fake = testfs::fake();
        device(&fake, "0000:01:00.0", 0x15b3, 0x1021, 0x020700);
        fs::write(fake.device("0000:01:00.0").join("vendor"), "0x10000").unwrap();
        assert_eq!(
            discover(&fake.sysfs, Catalog::builtin())
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        fs::remove_file(fake.device("0000:01:00.0").join("vendor")).unwrap();
        assert!(discover(&fake.sysfs, Catalog::builtin()).is_err());
    }
}
