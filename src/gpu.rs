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

/// Keep the existing conservative CC/VFIO check independent of exact attachment.
/// Device-only matching over-matches some 0x29bc/0x31c2 variants; narrowing that
/// policy belongs in a separate change. Coherent Rubin still allows in-band CC.
const C2C_DEVIDS: &[u16] = &[
    0x2342, 0x2343, 0x2345, 0x2348, // GH200
    0x2941, 0x297e, 0x29bc, // GB200
    0x31c2, // GB300
    0x3041, 0x307e, 0x30ff, // Rubin C2C variants
];

/// Also the set needing a vfio driver that can map coherent memory. Wider than
/// that driver's own table: a part can be coherently attached before any
/// released kernel claims it.
pub fn is_c2c(devid: u16) -> bool {
    C2C_DEVIDS.contains(&devid)
}

pub fn chip_for(devid: u16) -> Option<&'static Chip> {
    CHIPS
        .iter()
        .find(|c| (c.devid.0..=c.devid.1).contains(&devid))
}

/// Device IDs alone alias PCIe, SXM and coherent variants on some chips.
/// NVIDIA gpu-admin-tools v2026.09.29, gpu/devid_properties.py (44f261a7).
/// This precise classification does not narrow the conservative VFIO `is_c2c` set.
pub fn attachment(device: u16, subsystem_device: u16) -> Attachment {
    ATTACHMENTS
        .iter()
        .find(|(id, _)| *id == (device, subsystem_device))
        .map_or(Attachment::Unknown, |(_, attachment)| *attachment)
}

const ATTACHMENTS: &[((u16, u16), Attachment)] = &[
    ((0x2321, 0x1839), Attachment::Pcie),
    ((0x2322, 0x17a4), Attachment::Pcie),
    ((0x2324, 0x17a6), Attachment::Sxm),
    ((0x2324, 0x17a8), Attachment::Sxm),
    ((0x2328, 0x1905), Attachment::Sxm),
    ((0x2328, 0x1906), Attachment::Sxm),
    ((0x2329, 0x198b), Attachment::Sxm),
    ((0x2329, 0x198c), Attachment::Sxm),
    ((0x232c, 0x2063), Attachment::Sxm),
    ((0x232c, 0x2064), Attachment::Sxm),
    ((0x2330, 0x16c0), Attachment::Sxm),
    ((0x2330, 0x16c1), Attachment::Sxm),
    ((0x2330, 0x2044), Attachment::Sxm),
    ((0x2330, 0x20c1), Attachment::Sxm),
    ((0x2331, 0x1626), Attachment::Pcie),
    ((0x2335, 0x18be), Attachment::Sxm),
    ((0x2335, 0x18bf), Attachment::Sxm),
    ((0x2336, 0x16c2), Attachment::Sxm),
    ((0x2336, 0x16c7), Attachment::Sxm),
    ((0x2337, 0x16e5), Attachment::Sxm),
    ((0x2337, 0x16ef), Attachment::Sxm),
    ((0x2338, 0x16f6), Attachment::Sxm),
    ((0x2338, 0x16f7), Attachment::Sxm),
    ((0x2339, 0x17d9), Attachment::Sxm),
    ((0x2339, 0x17fc), Attachment::Sxm),
    ((0x233a, 0x183a), Attachment::Pcie),
    ((0x233b, 0x1996), Attachment::Pcie),
    ((0x233d, 0x1626), Attachment::Pcie),
    ((0x2342, 0x16eb), Attachment::Coherent),
    ((0x2342, 0x16ec), Attachment::Coherent),
    ((0x2342, 0x16ed), Attachment::Coherent),
    ((0x2342, 0x1805), Attachment::Coherent),
    ((0x2342, 0x1809), Attachment::Coherent),
    ((0x2342, 0x1935), Attachment::Coherent),
    ((0x2342, 0x1937), Attachment::Coherent),
    ((0x2343, 0x16ec), Attachment::Coherent),
    ((0x2345, 0x16ed), Attachment::Coherent),
    ((0x2348, 0x18d2), Attachment::Coherent),
    ((0x2901, 0x1999), Attachment::Sxm),
    ((0x2901, 0x199b), Attachment::Sxm),
    ((0x2901, 0x199d), Attachment::Sxm),
    ((0x2901, 0x20da), Attachment::Sxm),
    ((0x2920, 0x197f), Attachment::Sxm),
    ((0x2920, 0x20de), Attachment::Sxm),
    ((0x2924, 0x18b6), Attachment::Pcie),
    ((0x2924, 0x20d4), Attachment::Pcie),
    ((0x2925, 0x18b7), Attachment::Pcie),
    ((0x293d, 0x18b6), Attachment::Pcie),
    ((0x293d, 0x197f), Attachment::Sxm),
    ((0x293d, 0x1999), Attachment::Sxm),
    ((0x2941, 0x0000), Attachment::Coherent),
    ((0x2941, 0x2045), Attachment::Coherent),
    ((0x2941, 0x2046), Attachment::Coherent),
    ((0x2941, 0x20ca), Attachment::Coherent),
    ((0x297e, 0x2046), Attachment::Coherent),
    ((0x29bc, 0x1985), Attachment::Sxm),
    ((0x29bc, 0x1997), Attachment::Pcie),
    ((0x29bc, 0x1998), Attachment::Pcie),
    ((0x29bc, 0x2045), Attachment::Coherent),
    ((0x29f1, 0x20dc), Attachment::Sxm),
    ((0x3002, 0x2277), Attachment::Sxm),
    ((0x3041, 0x221a), Attachment::Coherent),
    ((0x307e, 0x221a), Attachment::Coherent),
    ((0x30ff, 0x221b), Attachment::Coherent),
    ((0x30ff, 0x221c), Attachment::Coherent),
    ((0x3182, 0x20e5), Attachment::Sxm),
    ((0x3182, 0x20e6), Attachment::Sxm),
    ((0x3182, 0x220c), Attachment::Sxm),
    ((0x3183, 0x22f2), Attachment::Sxm),
    ((0x3184, 0x22f3), Attachment::Sxm),
    ((0x31a1, 0x2274), Attachment::Coherent),
    ((0x31c2, 0x20e5), Attachment::Sxm),
    ((0x31c2, 0x20e6), Attachment::Sxm),
    ((0x31c2, 0x21f1), Attachment::Coherent),
    ((0x31c3, 0x23ab), Attachment::Coherent),
    ((0x31fe, 0x20e5), Attachment::Sxm),
    ((0x3224, 0x215f), Attachment::Sxm),
    ((0x323e, 0x215f), Attachment::Sxm),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c2c_blocks_enable_only() {
        assert!(C2C_DEVIDS.contains(&0x2342)); // GH200
        assert_eq!(chip_for(0x2342).unwrap().name, "GH100");
    }

    #[test]
    fn every_c2c_id_is_a_known_chip() {
        for devid in C2C_DEVIDS {
            assert!(chip_for(*devid).is_some(), "{devid:#06x} has no chip row");
        }
    }

    #[test]
    fn subsystem_identity_separates_aliasing_parts() {
        assert_eq!(attachment(0x29bc, 0x1985), Attachment::Sxm);
        assert_eq!(attachment(0x29bc, 0x2045), Attachment::Coherent);
        assert_eq!(attachment(0x29bc, 0x1997), Attachment::Pcie);
        assert_eq!(attachment(0x29bc, 0xffff), Attachment::Unknown);
        for ((device, ssid), kind) in ATTACHMENTS {
            assert_eq!(attachment(*device, *ssid), *kind);
            assert!(chip_for(*device).is_some());
        }
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
