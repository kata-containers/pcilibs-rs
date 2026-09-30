// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Hopper,
    Blackwell,
    Rubin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attachment {
    Pcie,
    Sxm,
    Coherent,
    Unknown,
}

/// One CC-capable GPU generation: a PCI device-id range and the two
/// per-generation register facts and CC capabilities.
/// Supporting a new chip is one row.
pub struct Chip {
    pub family: Family,
    pub name: &'static str,
    /// Inclusive PCI device-id range.
    pub devid: (u16, u16),
    /// Hopper uses the EMEM RPC channel, extra PRC knobs and a different
    /// CC-state register; Blackwell and Rubin use MNOC and have a boot
    /// BAR0 firewall.
    pub hopper: bool,
    /// Whether in-band CC enablement is supported on coherent variants.
    /// gpu-admin-tools v2026.09.29 `nvidia_gpu_tools.py::Gpu.__init__`
    /// restricts C2C enablement on Hopper and Blackwell only.
    pub c2c_cc_supported: bool,
    /// NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE: reads 0xff once the FSP
    /// has finished booting the GPU.
    pub boot_complete: u32,
}

/// Device-id ranges from gpu-admin-tools (`gpu/devid_chips.py`).
#[rustfmt::skip]
pub const CHIPS: &[Chip] = &[
    Chip { family: Family::Hopper, name: "GH100", devid: (0x22f0, 0x237f), hopper: true, c2c_cc_supported: false, boot_complete: 0x200bc },
    Chip { family: Family::Blackwell, name: "GB100", devid: (0x2900, 0x297f), hopper: false, c2c_cc_supported: false, boot_complete: 0x200bc },
    Chip { family: Family::Blackwell, name: "GB102", devid: (0x2980, 0x29ff), hopper: false, c2c_cc_supported: false, boot_complete: 0x200bc },
    Chip { family: Family::Blackwell, name: "GB110", devid: (0x3180, 0x31ff), hopper: false, c2c_cc_supported: false, boot_complete: 0x200bc },
    Chip { family: Family::Blackwell, name: "GB112", devid: (0x3200, 0x327f), hopper: false, c2c_cc_supported: false, boot_complete: 0x200bc },
    Chip { family: Family::Blackwell, name: "GB202", devid: (0x2b80, 0x2bff), hopper: false, c2c_cc_supported: false, boot_complete: 0xad00bc },
    Chip { family: Family::Blackwell, name: "GB203", devid: (0x2c00, 0x2c7f), hopper: false, c2c_cc_supported: false, boot_complete: 0xad00bc },
    Chip { family: Family::Blackwell, name: "GB205", devid: (0x2f00, 0x2f7f), hopper: false, c2c_cc_supported: false, boot_complete: 0xad00bc },
    Chip { family: Family::Blackwell, name: "GB206", devid: (0x2d00, 0x2d7f), hopper: false, c2c_cc_supported: false, boot_complete: 0xad00bc },
    Chip { family: Family::Blackwell, name: "GB207", devid: (0x2d80, 0x2dff), hopper: false, c2c_cc_supported: false, boot_complete: 0xad00bc },
    // gpu-admin-tools v2026.09.29 (44f261a7): gpu/devid_chips.py;
    // gpu/regs/gr100/therm.py and gr102/therm.py import the GB202 boot register.
    Chip { family: Family::Rubin, name: "GR100", devid: (0x3000, 0x307f), hopper: false, c2c_cc_supported: true, boot_complete: 0xad00bc },
    Chip { family: Family::Rubin, name: "GR102", devid: (0x3080, 0x30ff), hopper: false, c2c_cc_supported: true, boot_complete: 0xad00bc },
];

/// Conservative compatibility API for callers without subsystem identity.
/// Use a catalog lookup when the exact attachment is needed.
pub fn is_c2c(devid: u16) -> bool {
    catalog::Catalog::builtin().may_be_coherent(devid)
}

pub fn chip_for(devid: u16) -> Option<&'static Chip> {
    CHIPS
        .iter()
        .find(|c| (c.devid.0..=c.devid.1).contains(&devid))
}

/// Device IDs alone alias PCIe, SXM and coherent variants on some chips.
/// NVIDIA gpu-admin-tools v2026.09.29, gpu/devid_properties.py (44f261a7).
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
