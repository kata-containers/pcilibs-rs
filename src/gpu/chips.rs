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
    /// gpu-admin-tools `nvidia_gpu_tools.py::Gpu.__init__`
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
    // gpu-admin-tools: gpu/devid_chips.py;
    // gpu/regs/gr100/therm.py and gr102/therm.py import the GB202 boot register.
    Chip { family: Family::Rubin, name: "GR100", devid: (0x3000, 0x307f), hopper: false, c2c_cc_supported: true, boot_complete: 0xad00bc },
    Chip { family: Family::Rubin, name: "GR102", devid: (0x3080, 0x30ff), hopper: false, c2c_cc_supported: true, boot_complete: 0xad00bc },
];

pub fn chip_for_range(devid: u16) -> Option<&'static Chip> {
    CHIPS
        .iter()
        .find(|c| (c.devid.0..=c.devid.1).contains(&devid))
}
