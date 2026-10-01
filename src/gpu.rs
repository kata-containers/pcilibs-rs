// SPDX-FileCopyrightText: Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

//! NVIDIA GPU profiles usable before firmware or driver access.
//! General PCI identities and descriptive metadata live in [`crate::catalog`].

// Keep the existing path usable while the database lives at crate::catalog.
pub use crate::catalog;
pub use crate::catalog::Attachment;

mod chips;
pub(crate) use chips::chip_for_range;
pub use chips::{Chip, Family, CHIPS};

/// Find a supported NVIDIA chip by its range or an unambiguous bundled mapping.
pub fn chip_for(devid: u16) -> Option<&'static Chip> {
    chip_for_range(devid).or_else(|| catalog_chip(catalog::Catalog::builtin(), devid))
}

fn catalog_chip(catalog: catalog::Catalog<'_>, device: u16) -> Option<&'static Chip> {
    let mut found: Option<&Chip> = None;
    for entry in catalog
        .entries()
        .filter(|e| e.vendor == 0x10de && e.device == device)
    {
        let chip = properties(entry.vendor, device, entry.properties)?.chip;
        if found.is_some_and(|previous| previous.name != chip.name) {
            return None;
        }
        found = Some(chip);
    }
    found
}

/// Conservative NVIDIA compatibility API for callers without subsystem identity.
/// Use a catalog lookup when the exact attachment is needed.
pub fn is_c2c(devid: u16) -> bool {
    catalog::Catalog::builtin().may_be_coherent(0x10de, devid)
}

/// NVIDIA device IDs alone alias PCIe, SXM and coherent variants on some chips.
/// NVIDIA gpu-admin-tools, gpu/devid_properties.py.
/// Device-only callers can use the conservative `is_c2c` compatibility API.
/// This compatibility API assumes NVIDIA subsystem vendor; use a full catalog
/// identity for boards with another subsystem vendor.
pub fn attachment(device: u16, subsystem_device: u16) -> Attachment {
    catalog::Catalog::builtin()
        .lookup(catalog::PciIdentity::new(
            0x10de,
            device,
            0x10de,
            subsystem_device,
        ))
        .map_or(Attachment::Unknown, |properties| properties.attachment)
}

/// NVIDIA register support is separate from descriptive PCI catalog metadata.
#[derive(Clone, Copy)]
pub struct Properties {
    pub chip: &'static Chip,
    pub attachment: Attachment,
}

impl Properties {
    pub fn in_band_cc_supported(self) -> bool {
        match self.attachment {
            Attachment::Pcie | Attachment::Sxm => true,
            Attachment::Coherent => self.chip.c2c_cc_supported,
            Attachment::Unknown => false,
        }
    }
}

/// Resolve only profiles supported by the NVIDIA register implementation.
pub fn properties(vendor: u16, device: u16, record: catalog::Properties<'_>) -> Option<Properties> {
    if vendor != 0x10de
        || record.kind != catalog::DeviceKind::Gpu
        || record.attachment == Attachment::Unknown
    {
        return None;
    }
    let chip = CHIPS.iter().find(|chip| chip.name == record.profile)?;
    if chip_for_range(device).is_some_and(|range| range.name != chip.name) {
        return None;
    }
    Some(Properties {
        chip,
        attachment: record.attachment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_only_lookup_requires_an_unambiguous_nvidia_profile() {
        for (text, expected) in [
            ("15b3 ffff * * nic GH100 pcie", None),
            (
                "10de ffff * 0001 gpu GH100 sxm\n10de ffff * 0002 gpu GH100 pcie",
                Some("GH100"),
            ),
            (
                "10de ffff * 0001 gpu GH100 sxm\n10de ffff * 0002 gpu GB100 sxm",
                None,
            ),
            (
                "10de ffff * 0001 gpu GH100 sxm\n10de ffff * 0002 gpu FutureGPU sxm",
                None,
            ),
        ] {
            let catalog = catalog::Catalog::parse(text).unwrap();
            assert_eq!(catalog_chip(catalog, 0xffff).map(|c| c.name), expected);
        }
    }

    #[test]
    fn catalog_metadata_cannot_grant_nvidia_register_support() {
        use crate::catalog::{DeviceKind, Properties as Record};
        let record = Record {
            kind: DeviceKind::Gpu,
            profile: "GH100",
            attachment: Attachment::Pcie,
        };
        assert!(properties(0x15b3, 0x2330, record).is_none());
        assert!(properties(0x10de, 0x2902, record).is_none());
        assert!(properties(
            0x10de,
            0xffff,
            Record {
                kind: DeviceKind::Nic,
                ..record
            }
        )
        .is_none());
        assert!(properties(
            0x10de,
            0xffff,
            Record {
                profile: "FutureGPU",
                ..record
            }
        )
        .is_none());
        assert!(properties(
            0x10de,
            0xffff,
            Record {
                attachment: Attachment::Unknown,
                ..record
            }
        )
        .is_none());
        assert!(properties(0x10de, 0xffff, record)
            .unwrap()
            .in_band_cc_supported());
        assert!(!properties(
            0x10de,
            0xffff,
            Record {
                attachment: Attachment::Coherent,
                ..record
            }
        )
        .unwrap()
        .in_band_cc_supported());
        assert!(properties(
            0x10de,
            0xffff,
            Record {
                profile: "GR100",
                attachment: Attachment::Coherent,
                ..record
            }
        )
        .unwrap()
        .in_band_cc_supported());
        assert!(!Properties {
            chip: &CHIPS[0],
            attachment: Attachment::Unknown
        }
        .in_band_cc_supported());
    }

    #[test]
    fn c2c_blocks_enable_only() {
        assert!(is_c2c(0x2342)); // GH200
        assert_eq!(chip_for(0x2342).unwrap().name, "GH100");
    }

    #[test]
    fn subsystem_identity_separates_aliasing_parts() {
        assert_eq!(attachment(0x29bc, 0x1985), Attachment::Sxm);
        assert_eq!(attachment(0x29bc, 0x2045), Attachment::Coherent);
        assert_eq!(attachment(0x29bc, 0x1997), Attachment::Pcie);
        assert_eq!(attachment(0x29bc, 0xffff), Attachment::Unknown);
    }

    #[test]
    fn pci_ranges_and_coherent_rubin_work_without_cc() {
        for chip in CHIPS {
            assert_eq!(chip_for(chip.devid.0).unwrap().name, chip.name);
            assert_eq!(chip_for(chip.devid.1).unwrap().name, chip.name);
        }
        assert!(chip_for(0xffff).is_none());
        for device in [0x3041, 0x307e, 0x30ff] {
            assert!(is_c2c(device));
            assert!(chip_for(device).unwrap().c2c_cc_supported);
        }
        assert!(!is_c2c(0x2330));
    }
}
