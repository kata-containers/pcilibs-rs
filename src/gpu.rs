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

//! PCI identity must be usable before firmware or driver access.

pub mod catalog;

mod chips;
pub(crate) use chips::chip_for_range;
pub use chips::{Attachment, Chip, Family, CHIPS};

/// Find a supported chip by its compiled range or a bundled catalog mapping.
pub fn chip_for(devid: u16) -> Option<&'static Chip> {
    chip_for_range(devid).or_else(|| catalog::Catalog::builtin().lookup_chip(devid))
}

/// Conservative compatibility API for callers without subsystem identity.
/// Use a catalog lookup when the exact attachment is needed.
pub fn is_c2c(devid: u16) -> bool {
    catalog::Catalog::builtin().may_be_coherent(devid)
}

/// Device IDs alone alias PCIe, SXM and coherent variants on some chips.
/// NVIDIA gpu-admin-tools, gpu/devid_properties.py.
/// Device-only callers can use the conservative `is_c2c` compatibility API.
pub fn attachment(device: u16, subsystem_device: u16) -> Attachment {
    catalog::Catalog::builtin()
        .lookup(device, subsystem_device)
        .map_or(Attachment::Unknown, |properties| properties.attachment)
}

#[cfg(test)]
mod tests {
    use super::*;

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
