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

//! In-band NVIDIA confidential computing control: per-GPU CC, and
//! Protected PCIe across a whole HGX baseboard.
//!
//! Rust port of the CC subset of NVIDIA's gpu-admin-tools: query the
//! current mode, set a new one through FSP PRC knobs, and reset the
//! device so it takes effect.  Talks to the hardware only through
//! sysfs (identity, reset) and a BAR0 mapping (registers) — no driver
//! involved.  The device must be idle: not held by nvidia/vfio while in use.
//!
//! ```no_run
//! use pcilibs_rs::cc::{CcMode, Gpu};
//!
//! fn provision(bdf: &str) -> anyhow::Result<()> {
//!     let gpu = Gpu::open(bdf)?;
//!     if gpu.query_cc_mode()? != CcMode::On {
//!         gpu.set_cc_mode(CcMode::On)?;
//!         gpu.reset()?; // the new mode takes effect on reset
//!     }
//!     Ok(())
//! }
//! ```
//!
//! [`discover`] lists CC-capable GPUs without opening them (no root, no
//! device wake-up); [`Gpu::open`] maps BAR0 and needs root.
//!
//! The register access itself is not NVIDIA-specific and lives in the crate
//! root as [`PciDev`].

pub mod fsp;

use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};

use crate::{attr_hex, normalize_bdf, poll, PciDev, Sysfs};
use fsp::FspRpc;

/// The three CC modes (gpu-admin-tools `--set-cc-mode` values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CcMode {
    /// Confidential computing disabled.
    Off,
    /// Full confidential computing.
    On,
    /// CC with the profiling/debugging interfaces left open.
    DevTools,
}

impl std::str::FromStr for CcMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            "devtools" => Ok(Self::DevTools),
            _ => bail!("invalid CC mode {s:?} (expected off, on or devtools)"),
        }
    }
}

impl std::fmt::Display for CcMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::On => "on",
            Self::DevTools => "devtools",
        })
    }
}

/// Protected PCIe: one mode for a whole HGX baseboard, mutually exclusive
/// with per-GPU CC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpcieMode {
    Off,
    On,
}

impl std::str::FromStr for PpcieMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            _ => bail!("invalid PPCIE mode {s:?} (expected off or on)"),
        }
    }
}

impl std::fmt::Display for PpcieMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::On => "on",
        })
    }
}

/// Ordered knob writes for `mode` (gpu-admin-tools `set_ppcie_mode`).
///
/// Knob 34 is cleared unconditionally: upstream gates it on
/// `is_nvswitch() or is_hopper`, and only those two reach here.
/// Switches have no BAR0 decoupler, so theirs is only written back to 0.
fn ppcie_knob_plan(mode: PpcieMode, bar0_decoupler: bool) -> Vec<(u32, u16)> {
    let mut plan = Vec::new();
    if mode == PpcieMode::On {
        for knob in [
            fsp::KNOB_2,
            fsp::KNOB_4,
            fsp::KNOB_CCD,
            fsp::KNOB_CCM,
            fsp::KNOB_34,
        ] {
            plan.push((knob, 0));
        }
    }
    let decoupler = if mode == PpcieMode::On && bar0_decoupler {
        2
    } else {
        0
    };
    plan.push((fsp::KNOB_BAR0_DECOUPLER, decoupler));
    plan.push((fsp::KNOB_PPCIE, u16::from(mode == PpcieMode::On)));
    plan
}

/// One CC-capable GPU generation: a PCI device-id range and the two
/// per-generation register facts.  Supporting a new chip is one row.
pub struct Chip {
    pub name: &'static str,
    /// Inclusive PCI device-id range.
    pub devid: (u16, u16),
    /// Hopper uses the EMEM RPC channel, extra PRC knobs and a different
    /// CC-state register; Blackwell uses MNOC and has a boot BAR0 firewall.
    pub hopper: bool,
    /// NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE: reads 0xff once the FSP
    /// has finished booting the GPU.
    pub boot_complete: u32,
}

/// Device-id ranges from gpu-admin-tools (`gpu/devid_chips.py`).
#[rustfmt::skip]
pub const CHIPS: &[Chip] = &[
    Chip { name: "GH100", devid: (0x22f0, 0x237f), hopper: true, boot_complete: 0x200bc },
    Chip { name: "GB100", devid: (0x2900, 0x297f), hopper: false, boot_complete: 0x200bc },
    Chip { name: "GB102", devid: (0x2980, 0x29ff), hopper: false, boot_complete: 0x200bc },
    Chip { name: "GB110", devid: (0x3180, 0x31ff), hopper: false, boot_complete: 0x200bc },
    Chip { name: "GB112", devid: (0x3200, 0x327f), hopper: false, boot_complete: 0x200bc },
    Chip { name: "GB202", devid: (0x2b80, 0x2bff), hopper: false, boot_complete: 0xad00bc },
    Chip { name: "GB203", devid: (0x2c00, 0x2c7f), hopper: false, boot_complete: 0xad00bc },
    Chip { name: "GB205", devid: (0x2f00, 0x2f7f), hopper: false, boot_complete: 0xad00bc },
    Chip { name: "GB206", devid: (0x2d00, 0x2d7f), hopper: false, boot_complete: 0xad00bc },
    Chip { name: "GB207", devid: (0x2d80, 0x2dff), hopper: false, boot_complete: 0xad00bc },
];

/// CC is owned by system firmware here, so raising it in band is refused.
///
/// From gpu-admin-tools' `has_c2c`, which keys on (device, subsystem device)
/// pairs; only the device half is kept, no subsystem id being read anywhere in
/// this crate. That over-matches `0x29bc` and `0x31c2`, which have
/// non-coherent variants — a needless refusal, chosen over mistaking a
/// coherent GPU for an ordinary one.
const C2C_DEVIDS: &[u16] = &[
    0x2342, 0x2343, 0x2345, 0x2348, // GH200
    0x2941, 0x297e, 0x29bc, // GB200
    0x31c2, // GB300
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

const NV_PMC_BOOT_0: u32 = 0x0;
/// CC state lives in secure scratch, bits 1:0: 0 off, 1 on, 3 devtools.
const CC_STATE_HOPPER: u32 = 0x1182cc;
const CC_STATE_BLACKWELL: u32 = 0x590;
/// PPCIE state is bit 5 of that same Hopper scratch register.
const PPCIE_STATE_HOPPER: u32 = 0x1182cc;
const PPCIE_STATE_HOPPER_BIT: u32 = 0x20;
const PPCIE_STATE_SWITCH: u32 = 0x28c50;
const PPCIE_STATE_SWITCH_BIT: u32 = 0x1;
const BOOT_COMPLETE_OK: u32 = 0xff;

/// PCI addresses of all CC-capable NVIDIA GPUs on the node, sorted.
/// Reads only sysfs identity attributes: no root, no BAR0 mapping, and
/// no wake-up of runtime-suspended devices.
pub fn discover(sysfs: &Sysfs) -> Result<Vec<String>> {
    let mut bdfs = Vec::new();
    for entry in std::fs::read_dir(sysfs.devices()).context("read sysfs PCI tree")? {
        let entry = entry?;
        let Some(bdf) = entry.file_name().to_str().and_then(normalize_bdf) else {
            continue;
        };
        let dir = sysfs.devices().join(&bdf);
        let vendor = attr_hex(&dir, "vendor").unwrap_or(0);
        let device = attr_hex(&dir, "device").unwrap_or(0);
        if vendor == 0x10de && chip_for(device as u16).is_some() {
            bdfs.push(bdf);
        }
    }
    bdfs.sort();
    Ok(bdfs)
}

/// One CC-capable NVIDIA GPU.
pub struct Gpu {
    pci: PciDev,
    pub chip: &'static Chip,
    pub c2c: bool,
}

impl Gpu {
    /// Open a GPU by PCI address (`0000:65:00.0`; domain optional).
    pub fn open(bdf: &str) -> Result<Self> {
        Self::open_in(&Sysfs::default(), bdf)
    }

    pub fn open_in(sysfs: &Sysfs, bdf: &str) -> Result<Self> {
        let pci = PciDev::open_in(sysfs, bdf)?;
        ensure!(
            pci.vendor == 0x10de,
            "{}: vendor {:#06x} is not NVIDIA",
            pci.bdf,
            pci.vendor
        );
        let chip = chip_for(pci.device).with_context(|| {
            format!(
                "{}: device {:#06x} is not a CC-capable GPU (Hopper or Blackwell)",
                pci.bdf, pci.device
            )
        })?;
        let c2c = C2C_DEVIDS.contains(&pci.device);
        let gpu = Self { pci, chip, c2c };

        gpu.wait_for_bar0()?;
        let boot0 = gpu.pci.read32(NV_PMC_BOOT_0);
        ensure!(boot0 != 0xffff_ffff, "{}: BAR0 not accessible", gpu.bdf());
        ensure!(
            boot0 != 0xbadf_0200 && boot0 != 0xbad0_0200,
            "{}: GPU is in a security-fault state (BOOT_0 = {boot0:#010x})",
            gpu.bdf()
        );
        Ok(gpu)
    }

    /// [`discover`] and open every CC-capable GPU.  Fails on the first GPU
    /// that cannot be opened; open individually for per-GPU error handling.
    pub fn enumerate(sysfs: &Sysfs) -> Result<Vec<Gpu>> {
        discover(sysfs)?
            .iter()
            .map(|bdf| Gpu::open_in(sysfs, bdf))
            .collect()
    }

    pub fn bdf(&self) -> &str {
        &self.pci.bdf
    }

    pub fn devid(&self) -> u16 {
        self.pci.device
    }

    /// Blackwell keeps a BAR0 firewall up during boot; every register
    /// reads all-ones until the FSP lowers it.
    fn wait_for_bar0(&self) -> Result<()> {
        if self.chip.hopper {
            return Ok(());
        }
        poll("BAR0 firewall", Duration::from_secs(15), || {
            self.pci.read32(NV_PMC_BOOT_0) != 0xffff_ffff
        })?;

        Ok(())
    }

    /// Wait until the FSP reports boot complete (scratch reads 0xff).
    pub fn wait_for_boot(&self) -> Result<()> {
        self.wait_for_bar0()?;
        poll("GPU boot complete", Duration::from_secs(10), || {
            self.pci.read32(self.chip.boot_complete) == BOOT_COMPLETE_OK
        })?;

        Ok(())
    }

    /// The mode the GPU is currently running with.
    pub fn query_cc_mode(&self) -> Result<CcMode> {
        self.wait_for_boot()?;
        let reg = if self.chip.hopper {
            CC_STATE_HOPPER
        } else {
            CC_STATE_BLACKWELL
        };
        match self.pci.read32(reg) & 0x3 {
            0x0 => Ok(CcMode::Off),
            0x1 => Ok(CcMode::On),
            0x3 => Ok(CcMode::DevTools),
            _ => bail!(
                "{}: invalid CC state (devtools without CC); fix by setting a CC mode",
                self.bdf()
            ),
        }
    }

    /// Persist a new CC mode in the FSP.  It takes effect on the next GPU
    /// reset — call [`Gpu::reset`] afterwards.
    pub fn set_cc_mode(&self, mode: CcMode) -> Result<()> {
        if self.c2c && mode != CcMode::Off {
            bail!(
                "{}: enabling CC in-band is not supported on C2C (Grace) systems",
                self.bdf()
            );
        }
        // FSP RPC may come up before the boot-complete scratch; if it never
        // does, the RPC polls below fail with a clear error.
        let _ = self.wait_for_boot();

        let rpc = if self.chip.hopper {
            FspRpc::emem(&self.pci)?
        } else {
            FspRpc::mnoc(&self.pci)
        };

        let (ccm, ccd, bar0_decoupler) = match mode {
            CcMode::On => (1, 0, 2),
            CcMode::DevTools => (1, 1, 0),
            CcMode::Off => (0, 0, 0),
        };

        if self.chip.hopper {
            if ccm == 1 {
                // Knobs that conflict with CC are cleared first.
                for knob in [fsp::KNOB_2, fsp::KNOB_4, fsp::KNOB_34] {
                    rpc.knob_check_and_write(knob, 0)?;
                }
                match rpc.knob_read(fsp::KNOB_PPCIE) {
                    Ok(0) => {}
                    Ok(_) => rpc.knob_write(fsp::KNOB_PPCIE, 0)?,
                    // Older firmware without the PPCIE knob.
                    Err(err) if fsp::is_invalid_knob(&err) => {}
                    Err(err) => return Err(err),
                }
            }
            rpc.knob_check_and_write(fsp::KNOB_BAR0_DECOUPLER, bar0_decoupler)?;
        }

        // CCM goes on first and off last so a CCD-only state (invalid)
        // never exists.
        if ccm == 1 {
            rpc.knob_check_and_write(fsp::KNOB_CCM, ccm)?;
            rpc.knob_check_and_write(fsp::KNOB_CCD, ccd)?;
        } else {
            rpc.knob_check_and_write(fsp::KNOB_CCD, ccd)?;
            rpc.knob_check_and_write(fsp::KNOB_CCM, ccm)?;
        }
        Ok(())
    }

    /// Hopper only, not Hopper-plus: Blackwell encrypts NVLink and reports
    /// no PPCIE support (gpu-admin-tools `is_ppcie_query_supported`).
    pub fn supports_ppcie(&self) -> bool {
        self.chip.hopper
    }

    pub fn query_ppcie_mode(&self) -> Result<PpcieMode> {
        ensure!(
            self.supports_ppcie(),
            "{}: {} does not support PPCIE (Hopper only)",
            self.bdf(),
            self.chip.name
        );
        self.wait_for_boot()?;
        Ok(ppcie_state(
            self.pci.read32(PPCIE_STATE_HOPPER),
            PPCIE_STATE_HOPPER_BIT,
        ))
    }

    /// Takes effect on reset, and only once every GPU and switch on the
    /// baseboard carries the same mode.
    pub fn set_ppcie_mode(&self, mode: PpcieMode) -> Result<()> {
        ensure!(
            self.supports_ppcie(),
            "{}: {} does not support PPCIE (Hopper only)",
            self.bdf(),
            self.chip.name
        );
        let _ = self.wait_for_boot();
        let rpc = FspRpc::emem(&self.pci)?;
        apply_ppcie_plan(&rpc, mode, true).with_context(|| format!("{}: set PPCIE", self.bdf()))
    }

    /// Function-level reset, then wait for the GPU to boot back up.
    /// This is what makes a previously set CC mode active.
    pub fn reset(&self) -> Result<()> {
        self.pci.sysfs_reset()?;
        self.wait_for_boot()
    }
}

fn ppcie_state(reg: u32, bit: u32) -> PpcieMode {
    if reg & bit == bit {
        PpcieMode::On
    } else {
        PpcieMode::Off
    }
}

/// One NVSwitch generation, keyed by `NV_PMC_BOOT_0` rather than PCI
/// device id — switches have no distinguishing id (gpu-admin-tools
/// `NVSWITCH_MAP`).
pub struct Switch {
    pub name: &'static str,
    pub boot0: u32,
    /// Reads 0xff once the FSP has finished booting the switch.
    pub boot_complete: u32,
}

/// LimeRock (gen2, boot0 0x6000a1) is absent: no FSP, so no PRC knobs.
pub const SWITCHES: &[Switch] = &[Switch {
    name: "NVSwitch_gen3",
    boot0: 0x7000a1,
    boot_complete: 0x660bc,
}];

pub fn switch_for(boot0: u32) -> Option<&'static Switch> {
    SWITCHES.iter().find(|s| s.boot0 == boot0)
}

const PCI_CLASS_BRIDGE_OTHER: u32 = 0x0680;

/// PCI addresses of all NVIDIA NVSwitches on the node, sorted.  Generation
/// is only knowable from BAR0, so [`NvSwitch::open`] does that check.
pub fn discover_switches(sysfs: &Sysfs) -> Result<Vec<String>> {
    let mut bdfs = Vec::new();
    for entry in std::fs::read_dir(sysfs.devices()).context("read sysfs PCI tree")? {
        let entry = entry?;
        let Some(bdf) = entry.file_name().to_str().and_then(normalize_bdf) else {
            continue;
        };
        let dir = sysfs.devices().join(&bdf);
        let vendor = attr_hex(&dir, "vendor").unwrap_or(0);
        let class = attr_hex(&dir, "class").unwrap_or(0);
        if vendor == 0x10de && class >> 8 == PCI_CLASS_BRIDGE_OTHER {
            bdfs.push(bdf);
        }
    }
    bdfs.sort();
    Ok(bdfs)
}

/// One NVSwitch.  PPCIE is the only mode it carries; there is no
/// per-switch CC.
pub struct NvSwitch {
    pci: PciDev,
    pub switch: &'static Switch,
}

impl NvSwitch {
    pub fn open(bdf: &str) -> Result<Self> {
        Self::open_in(&Sysfs::default(), bdf)
    }

    pub fn open_in(sysfs: &Sysfs, bdf: &str) -> Result<Self> {
        let pci = PciDev::open_in(sysfs, bdf)?;
        ensure!(
            pci.vendor == 0x10de,
            "{}: vendor {:#06x} is not NVIDIA",
            pci.bdf,
            pci.vendor
        );
        let boot0 = pci.read32(NV_PMC_BOOT_0);
        ensure!(boot0 != 0xffff_ffff, "{}: BAR0 not accessible", pci.bdf);
        let switch = switch_for(boot0).with_context(|| {
            format!(
                "{}: NV_PMC_BOOT_0 {boot0:#010x} is not a PPCIE-capable NVSwitch",
                pci.bdf
            )
        })?;
        Ok(Self { pci, switch })
    }

    pub fn enumerate(sysfs: &Sysfs) -> Result<Vec<NvSwitch>> {
        discover_switches(sysfs)?
            .iter()
            .map(|bdf| NvSwitch::open_in(sysfs, bdf))
            .collect()
    }

    pub fn bdf(&self) -> &str {
        &self.pci.bdf
    }

    pub fn wait_for_boot(&self) -> Result<()> {
        poll("NVSwitch boot complete", Duration::from_secs(10), || {
            self.pci.read32(self.switch.boot_complete) == BOOT_COMPLETE_OK
        })?;

        Ok(())
    }

    pub fn query_ppcie_mode(&self) -> Result<PpcieMode> {
        self.wait_for_boot()?;
        Ok(ppcie_state(
            self.pci.read32(PPCIE_STATE_SWITCH),
            PPCIE_STATE_SWITCH_BIT,
        ))
    }

    /// EMEM, like Hopper: only Blackwell GPUs moved to MNOC, and
    /// gpu-admin-tools gives GPU and switch the same `FspFalcon`.
    pub fn set_ppcie_mode(&self, mode: PpcieMode) -> Result<()> {
        let _ = self.wait_for_boot();
        let rpc = FspRpc::emem(&self.pci)?;
        apply_ppcie_plan(&rpc, mode, false).with_context(|| format!("{}: set PPCIE", self.bdf()))
    }

    pub fn reset(&self) -> Result<()> {
        self.pci.sysfs_reset()?;
        self.wait_for_boot()
    }
}

fn apply_ppcie_plan(rpc: &FspRpc, mode: PpcieMode, bar0_decoupler: bool) -> Result<()> {
    // Probe first so firmware too old for PPCIE is reported as that, and
    // not as a failure of whichever knob the plan writes first.
    if let Err(err) = rpc.knob_read(fsp::KNOB_PPCIE) {
        if fsp::is_invalid_knob(&err) {
            bail!("firmware does not support PPCIE; a firmware update is required");
        }
        return Err(err);
    }
    for (knob, value) in ppcie_knob_plan(mode, bar0_decoupler) {
        rpc.knob_check_and_write(knob, value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::fsp::fake::{Fault, Fsp};
    use super::*;
    use crate::testfs::{self, Fake};
    use rstest::{fixture, rstest};
    use std::sync::Arc;

    const BDF: &str = "0000:65:00.0";
    /// H100 SXM5, and the GH200 that is the same chip coherently attached.
    const GH100: u16 = 0x2330;
    const GH200: u16 = 0x2342;
    const GB100: u16 = 0x2901;
    /// A100: enumerable, but no CC.
    const AMPERE: u16 = 0x20b0;
    /// An NVSwitch, which is known by its class and not this id.
    const NVSWITCH: u16 = 0x22a3;
    const NVSWITCH_CLASS: u32 = 0x068000;
    const NVSWITCH_GEN3: u32 = 0x7000a1;
    const GPU_CLASS: u32 = 0x030200;

    /// The registers here sit megabytes apart, so a BAR0 file is sparse.
    const BAR0_LEN: u64 = 16 << 20;

    #[fixture]
    fn fake() -> Fake {
        testfs::fake()
    }

    /// The whole chain: the message wanted is often a cause rather than the
    /// error at the top.
    fn why<T>(result: Result<T>) -> String {
        format!("{:#}", result.err().expect("an error"))
    }

    /// A booted GPU whose registers a model answers.  Opening one only reads
    /// BAR0, which a file can do; talking to its FSP needs a reply.
    fn modelled_gpu(devid: u16) -> (Fake, Arc<Fsp>, Gpu) {
        let fake = testfs::fake();
        fake.add_device(BDF, None);
        std::fs::write(fake.device(BDF).join("reset"), "").unwrap();

        let chip = chip_for(devid).expect("a CC-capable device id");
        let firmware = if chip.hopper {
            Fsp::emem()
        } else {
            Fsp::mnoc()
        };
        firmware.set_register(chip.boot_complete, BOOT_COMPLETE_OK);
        let gpu = Gpu {
            pci: PciDev::modelled(
                fake.device(BDF),
                BDF,
                0x10de,
                devid,
                Box::new(firmware.clone()),
            ),
            chip,
            c2c: is_c2c(devid),
        };

        (fake, firmware, gpu)
    }

    fn modelled_switch() -> (Fake, Arc<Fsp>, NvSwitch) {
        let fake = testfs::fake();
        fake.add_device(BDF, None);
        std::fs::write(fake.device(BDF).join("reset"), "").unwrap();

        let switch = switch_for(NVSWITCH_GEN3).expect("a PPCIE-capable switch");
        let firmware = Fsp::emem();
        firmware.set_register(switch.boot_complete, BOOT_COMPLETE_OK);
        let nvswitch = NvSwitch {
            pci: PciDev::modelled(
                fake.device(BDF),
                BDF,
                0x10de,
                NVSWITCH,
                Box::new(firmware.clone()),
            ),
            switch,
        };

        (fake, firmware, nvswitch)
    }

    #[test]
    fn chip_lookup() {
        assert_eq!(chip_for(0x2330).unwrap().name, "GH100"); // H100 SXM
        assert!(chip_for(0x2330).unwrap().hopper);
        assert_eq!(chip_for(0x2901).unwrap().name, "GB100");
        assert!(!chip_for(0x2901).unwrap().hopper);
        assert_eq!(chip_for(0x2b85).unwrap().name, "GB202");
        assert_eq!(chip_for(0x2b85).unwrap().boot_complete, 0xad00bc);
        assert!(chip_for(0x20b0).is_none()); // A100: no CC
    }

    #[test]
    fn c2c_blocks_enable_only() {
        assert!(C2C_DEVIDS.contains(&0x2342)); // GH200
        assert_eq!(chip_for(0x2342).unwrap().name, "GH100");
    }

    /// NVIDIA also marks 0x3041, 0x307e and 0x30ff coherent; they are absent
    /// because `CHIPS` has no row for them, and this is what keeps that honest.
    #[test]
    fn every_c2c_id_is_a_known_chip() {
        for devid in C2C_DEVIDS {
            assert!(chip_for(*devid).is_some(), "{devid:#06x} has no chip row");
        }
    }

    #[test]
    fn cc_mode_roundtrip() {
        for mode in [CcMode::Off, CcMode::On, CcMode::DevTools] {
            assert_eq!(mode.to_string().parse::<CcMode>().unwrap(), mode);
        }
        assert!("auto".parse::<CcMode>().is_err());
    }

    #[rstest]
    #[case::off(PpcieMode::Off)]
    #[case::on(PpcieMode::On)]
    fn ppcie_mode_roundtrip(#[case] mode: PpcieMode) {
        assert_eq!(mode.to_string().parse::<PpcieMode>().unwrap(), mode);
    }

    #[rstest]
    #[case::devtools("devtools")]
    #[case::empty("")]
    fn ppcie_has_no_devtools_variant(#[case] input: &str) {
        assert!(input.parse::<PpcieMode>().is_err());
    }

    #[rstest]
    #[case::gpu(true)]
    #[case::nvswitch(false)]
    fn enabling_ppcie_turns_cc_off_first(#[case] bar0_decoupler: bool) {
        let plan = ppcie_knob_plan(PpcieMode::On, bar0_decoupler);
        let position = |knob| {
            plan.iter()
                .position(|(k, _)| *k == knob)
                .unwrap_or_else(|| panic!("{knob:#x} is not in the plan: {plan:x?}"))
        };
        let ccm = position(fsp::KNOB_CCM);

        assert!(
            ccm < position(fsp::KNOB_PPCIE),
            "CC must be cleared before PPCIE is set: {plan:x?}"
        );
        assert_eq!(plan[ccm].1, 0);
        assert_eq!(plan.iter().find(|(k, _)| *k == fsp::KNOB_CCD).unwrap().1, 0);
    }

    #[rstest]
    fn the_ppcie_knob_is_written_last() {
        for mode in [PpcieMode::Off, PpcieMode::On] {
            let plan = ppcie_knob_plan(mode, true);
            assert_eq!(plan.last().unwrap().0, fsp::KNOB_PPCIE);
            assert_eq!(plan.last().unwrap().1, u16::from(mode == PpcieMode::On));
        }
    }

    #[rstest]
    #[case::gpu_on(PpcieMode::On, true, 2)]
    #[case::gpu_off(PpcieMode::Off, true, 0)]
    #[case::switch_on(PpcieMode::On, false, 0)]
    #[case::switch_off(PpcieMode::Off, false, 0)]
    fn bar0_decoupler_is_a_gpu_only_filter(
        #[case] mode: PpcieMode,
        #[case] bar0_decoupler: bool,
        #[case] expected: u16,
    ) {
        let plan = ppcie_knob_plan(mode, bar0_decoupler);
        let value = plan
            .iter()
            .find(|(k, _)| *k == fsp::KNOB_BAR0_DECOUPLER)
            .unwrap()
            .1;
        assert_eq!(value, expected);
    }

    #[rstest]
    fn disabling_ppcie_leaves_the_cc_knobs_alone() {
        let plan = ppcie_knob_plan(PpcieMode::Off, true);
        assert_eq!(
            plan,
            vec![(fsp::KNOB_BAR0_DECOUPLER, 0), (fsp::KNOB_PPCIE, 0)]
        );
    }

    #[rstest]
    #[case::hopper_off(0x0000_0000, PPCIE_STATE_HOPPER_BIT, PpcieMode::Off)]
    #[case::hopper_on(0x0000_0020, PPCIE_STATE_HOPPER_BIT, PpcieMode::On)]
    // CC and PPCIE share this register; its low bits are not PPCIE.
    #[case::hopper_cc_on(0x0000_0001, PPCIE_STATE_HOPPER_BIT, PpcieMode::Off)]
    #[case::switch_off(0x0000_0000, PPCIE_STATE_SWITCH_BIT, PpcieMode::Off)]
    #[case::switch_on(0x0000_0001, PPCIE_STATE_SWITCH_BIT, PpcieMode::On)]
    fn ppcie_state_decoding(#[case] reg: u32, #[case] bit: u32, #[case] expected: PpcieMode) {
        assert_eq!(ppcie_state(reg, bit), expected);
    }

    #[rstest]
    fn switch_lookup_is_laguna_plus_only() {
        assert_eq!(switch_for(0x7000a1).unwrap().name, "NVSwitch_gen3");
        assert_eq!(switch_for(0x7000a1).unwrap().boot_complete, 0x660bc);
        assert!(switch_for(0x6000a1).is_none(), "LimeRock has no FSP");
    }

    #[rstest]
    #[case::h100(0x2330, true)]
    #[case::gb100(0x2901, false)]
    #[case::gb202(0x2b85, false)]
    fn ppcie_is_hopper_only(#[case] devid: u16, #[case] expected: bool) {
        assert_eq!(chip_for(devid).unwrap().hopper, expected);
    }

    #[rstest]
    #[case::gh200(GH200, true)]
    #[case::h100_sxm(GH100, false)]
    fn c2c_is_the_coherently_attached_parts(#[case] devid: u16, #[case] expected: bool) {
        assert_eq!(is_c2c(devid), expected);
    }

    #[rstest]
    fn discovery_lists_cc_capable_gpus_in_address_order(fake: Fake) {
        fake.add_pci_device("0000:65:00.0", 0x10de, GH100, GPU_CLASS, None);
        fake.add_pci_device("0000:03:00.0", 0x10de, GB100, GPU_CLASS, None);
        fake.add_pci_device("0000:04:00.0", 0x8086, 0x100e, 0x020000, None);
        fake.add_pci_device("0000:05:00.0", 0x10de, AMPERE, GPU_CLASS, None);
        fake.add_device("0000:06:00.0", None); // no ids to read
        std::fs::create_dir(fake.sysfs.devices().join("not-an-address")).unwrap();

        assert_eq!(
            discover(&fake.sysfs).unwrap(),
            ["0000:03:00.0", "0000:65:00.0"]
        );
    }

    /// Class rather than device id: switches have no distinguishing one.
    #[rstest]
    fn switch_discovery_lists_nvidia_bridges_in_address_order(fake: Fake) {
        fake.add_pci_device("0000:0a:00.0", 0x10de, NVSWITCH, NVSWITCH_CLASS, None);
        fake.add_pci_device("0000:09:00.0", 0x10de, NVSWITCH, NVSWITCH_CLASS, None);
        fake.add_pci_device("0000:0b:00.0", 0x1af4, 0x1000, NVSWITCH_CLASS, None);
        fake.add_pci_device("0000:0c:00.0", 0x10de, GH100, GPU_CLASS, None);
        std::fs::create_dir(fake.sysfs.devices().join("not-an-address")).unwrap();

        assert_eq!(
            discover_switches(&fake.sysfs).unwrap(),
            ["0000:09:00.0", "0000:0a:00.0"]
        );
    }

    /// A caller told there are no GPUs would configure nothing and call it
    /// done.
    #[rstest]
    fn discovery_without_a_pci_tree_fails(fake: Fake) {
        std::fs::remove_dir(fake.sysfs.devices()).unwrap();

        assert!(discover(&fake.sysfs).is_err());
        assert!(discover_switches(&fake.sysfs).is_err());
    }

    #[rstest]
    fn opening_a_gpu_names_its_chip(fake: Fake) {
        fake.add_mappable_device(BDF, 0x10de, GH100, GPU_CLASS, BAR0_LEN);

        let gpu = Gpu::open_in(&fake.sysfs, BDF).unwrap();

        assert_eq!(gpu.chip.name, "GH100");
        assert_eq!((gpu.bdf(), gpu.devid()), (BDF, GH100));
        assert!(!gpu.c2c);
    }

    #[rstest]
    fn opening_a_blackwell_gpu_waits_for_the_bar0_firewall(fake: Fake) {
        fake.add_mappable_device(BDF, 0x10de, GB100, GPU_CLASS, BAR0_LEN);

        assert_eq!(Gpu::open_in(&fake.sysfs, BDF).unwrap().chip.name, "GB100");
    }

    #[rstest]
    fn enumeration_opens_every_gpu_it_discovers(fake: Fake) {
        fake.add_mappable_device(BDF, 0x10de, GH100, GPU_CLASS, BAR0_LEN);

        let gpus = Gpu::enumerate(&fake.sysfs).unwrap();

        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].bdf(), BDF);
    }

    #[rstest]
    #[case::another_vendor(0x8086, GH100, "is not NVIDIA")]
    #[case::no_cc(0x10de, AMPERE, "is not a CC-capable GPU")]
    fn opening_refuses_what_is_not_a_cc_capable_gpu(
        fake: Fake,
        #[case] vendor: u16,
        #[case] devid: u16,
        #[case] expected: &str,
    ) {
        fake.add_mappable_device(BDF, vendor, devid, GPU_CLASS, BAR0_LEN);

        let err = why(Gpu::open_in(&fake.sysfs, BDF));

        assert!(err.contains(expected), "{err}");
    }

    /// A caller handed one of these would read it back as a register value.
    #[rstest]
    #[case::bar0_inaccessible(0xffff_ffff, "BAR0 not accessible")]
    #[case::security_fault(0xbadf_0200, "security-fault state")]
    #[case::security_fault_early(0xbad0_0200, "security-fault state")]
    fn opening_refuses_a_gpu_that_will_not_answer(
        fake: Fake,
        #[case] boot0: u32,
        #[case] expected: &str,
    ) {
        fake.add_mappable_device(BDF, 0x10de, GH100, GPU_CLASS, BAR0_LEN);
        fake.set_register(BDF, NV_PMC_BOOT_0 as u64, boot0);

        let err = why(Gpu::open_in(&fake.sysfs, BDF));

        assert!(err.contains(expected), "{err}");
    }

    /// The two entry points that read the running kernel, so all a test can
    /// hold them to is refusing an address before they look.
    #[rstest]
    fn opening_by_address_alone_reads_the_running_kernel() {
        assert!(Gpu::open("nonsense").is_err());
        assert!(NvSwitch::open("nonsense").is_err());
    }

    #[rstest]
    #[case::hopper_off(GH100, CC_STATE_HOPPER, 0x0, CcMode::Off)]
    #[case::hopper_on(GH100, CC_STATE_HOPPER, 0x1, CcMode::On)]
    #[case::hopper_devtools(GH100, CC_STATE_HOPPER, 0x3, CcMode::DevTools)]
    #[case::blackwell_off(GB100, CC_STATE_BLACKWELL, 0x0, CcMode::Off)]
    #[case::blackwell_on(GB100, CC_STATE_BLACKWELL, 0x1, CcMode::On)]
    fn a_gpu_reports_the_cc_mode_it_is_running(
        #[case] devid: u16,
        #[case] register: u32,
        #[case] state: u32,
        #[case] expected: CcMode,
    ) {
        let (_fake, firmware, gpu) = modelled_gpu(devid);
        firmware.set_register(register, state);

        assert_eq!(gpu.query_cc_mode().unwrap(), expected);
    }

    /// Devtools without CC is not a mode, and there is no reading of the
    /// register that makes it one.
    #[rstest]
    fn a_gpu_in_a_cc_state_that_is_not_one_is_reported() {
        let (_fake, firmware, gpu) = modelled_gpu(GH100);
        firmware.set_register(CC_STATE_HOPPER, 0x2);

        let err = why(gpu.query_cc_mode());

        assert!(err.contains("invalid CC state"), "{err}");
    }

    #[rstest]
    #[case::hopper_on(GH100, CcMode::On, &[(fsp::KNOB_CCM, 1), (fsp::KNOB_CCD, 0), (fsp::KNOB_BAR0_DECOUPLER, 2)])]
    #[case::hopper_devtools(GH100, CcMode::DevTools, &[(fsp::KNOB_CCM, 1), (fsp::KNOB_CCD, 1), (fsp::KNOB_BAR0_DECOUPLER, 0)])]
    #[case::hopper_off(GH100, CcMode::Off, &[(fsp::KNOB_CCM, 0), (fsp::KNOB_CCD, 0), (fsp::KNOB_BAR0_DECOUPLER, 0)])]
    #[case::blackwell_on(GB100, CcMode::On, &[(fsp::KNOB_CCM, 1), (fsp::KNOB_CCD, 0)])]
    fn setting_a_cc_mode_writes_the_knobs_it_means(
        #[case] devid: u16,
        #[case] mode: CcMode,
        #[case] expected: &[(u32, u16)],
    ) {
        let (_fake, firmware, gpu) = modelled_gpu(devid);

        gpu.set_cc_mode(mode).unwrap();

        for (knob, value) in expected {
            assert_eq!(firmware.knob(*knob), Some(*value), "knob {knob:#x}");
        }
    }

    /// Both PPCIE and the three Hopper knobs that predate CC are mutually
    /// exclusive with it in the firmware.
    #[rstest]
    fn enabling_cc_on_hopper_clears_what_conflicts_with_it() {
        let (_fake, firmware, gpu) = modelled_gpu(GH100);
        for knob in [fsp::KNOB_2, fsp::KNOB_4, fsp::KNOB_34, fsp::KNOB_PPCIE] {
            firmware.set_knob(knob, 1);
        }

        gpu.set_cc_mode(CcMode::On).unwrap();

        for knob in [fsp::KNOB_2, fsp::KNOB_4, fsp::KNOB_34, fsp::KNOB_PPCIE] {
            assert_eq!(firmware.knob(knob), Some(0), "knob {knob:#x}");
        }
    }

    /// Firmware without a PPCIE knob has no PPCIE to clear.
    #[rstest]
    fn enabling_cc_on_firmware_without_the_ppcie_knob_goes_through() {
        let (_fake, firmware, gpu) = modelled_gpu(GH100);
        firmware.forget_knob(fsp::KNOB_PPCIE);

        gpu.set_cc_mode(CcMode::On).unwrap();

        assert_eq!(firmware.knob(fsp::KNOB_CCM), Some(1));
    }

    /// Any other failure reading it is one, though: CC would go on with
    /// PPCIE left in an unknown state.
    #[rstest]
    fn enabling_cc_stops_if_the_ppcie_knob_will_not_read() {
        let (_fake, firmware, gpu) = modelled_gpu(GH100);
        firmware.fail_knob(fsp::KNOB_PPCIE, Fault::Completion(0x5));

        let err = why(gpu.set_cc_mode(CcMode::On));

        assert!(err.contains("completion code 0x5"), "{err}");
        assert_eq!(firmware.knob(fsp::KNOB_CCM), Some(0));
    }

    /// The mode belongs to system firmware on a coherently attached part,
    /// so raising it in band is refused before anything is written.
    #[rstest]
    fn enabling_cc_on_a_c2c_gpu_is_refused() {
        let (_fake, firmware, gpu) = modelled_gpu(GH200);
        assert!(gpu.c2c);

        let err = why(gpu.set_cc_mode(CcMode::On));

        assert!(err.contains("not supported on C2C"), "{err}");
        assert_eq!(firmware.writes(), 0);
    }

    /// Turning it off there is not raising it, so that still goes through.
    #[rstest]
    fn disabling_cc_on_a_c2c_gpu_goes_through() {
        let (_fake, firmware, gpu) = modelled_gpu(GH200);
        firmware.set_knob(fsp::KNOB_CCM, 1);

        gpu.set_cc_mode(CcMode::Off).unwrap();

        assert_eq!(firmware.knob(fsp::KNOB_CCM), Some(0));
    }

    #[rstest]
    #[case::off(0x0000_0000, PpcieMode::Off)]
    #[case::on(0x0000_0020, PpcieMode::On)]
    fn a_hopper_gpu_reports_the_ppcie_mode_it_is_running(
        #[case] state: u32,
        #[case] expected: PpcieMode,
    ) {
        let (_fake, firmware, gpu) = modelled_gpu(GH100);
        firmware.set_register(PPCIE_STATE_HOPPER, state);

        assert!(gpu.supports_ppcie());
        assert_eq!(gpu.query_ppcie_mode().unwrap(), expected);
    }

    #[rstest]
    fn a_blackwell_gpu_has_no_ppcie_mode_either_way() {
        let (_fake, _firmware, gpu) = modelled_gpu(GB100);
        assert!(!gpu.supports_ppcie());

        for err in [
            why(gpu.query_ppcie_mode()),
            why(gpu.set_ppcie_mode(PpcieMode::On)),
        ] {
            assert!(err.contains("does not support PPCIE"), "{err}");
        }
    }

    #[rstest]
    fn setting_ppcie_on_a_gpu_clears_its_cc_knobs_first() {
        let (_fake, firmware, gpu) = modelled_gpu(GH100);
        firmware.set_knob(fsp::KNOB_CCM, 1);

        gpu.set_ppcie_mode(PpcieMode::On).unwrap();

        assert_eq!(firmware.knob(fsp::KNOB_CCM), Some(0));
        assert_eq!(firmware.knob(fsp::KNOB_PPCIE), Some(1));
        assert_eq!(firmware.knob(fsp::KNOB_BAR0_DECOUPLER), Some(2));
    }

    #[rstest]
    fn setting_ppcie_on_firmware_without_the_knob_says_so() {
        let (_fake, firmware, gpu) = modelled_gpu(GH100);
        firmware.forget_knob(fsp::KNOB_PPCIE);

        let err = why(gpu.set_ppcie_mode(PpcieMode::On));

        assert!(err.contains("firmware does not support PPCIE"), "{err}");
        assert_eq!(firmware.writes(), 0);
    }

    #[rstest]
    fn setting_ppcie_reports_a_failure_that_is_not_a_missing_knob() {
        let (_fake, firmware, gpu) = modelled_gpu(GH100);
        firmware.fail(Fault::Completion(0x5));

        let err = why(gpu.set_ppcie_mode(PpcieMode::On));

        assert!(err.contains("set PPCIE"), "{err}");
        assert!(err.contains("completion code 0x5"), "{err}");
    }

    #[rstest]
    fn resetting_a_gpu_goes_through_sysfs_and_waits_for_boot() {
        let (fake, _firmware, gpu) = modelled_gpu(GH100);

        gpu.reset().unwrap();

        assert_eq!(
            std::fs::read_to_string(fake.device(BDF).join("reset")).unwrap(),
            "1"
        );
    }

    #[rstest]
    fn opening_a_switch_names_its_generation(fake: Fake) {
        fake.add_mappable_device(BDF, 0x10de, NVSWITCH, NVSWITCH_CLASS, BAR0_LEN);
        fake.set_register(BDF, NV_PMC_BOOT_0 as u64, NVSWITCH_GEN3);

        let switch = NvSwitch::open_in(&fake.sysfs, BDF).unwrap();

        assert_eq!((switch.switch.name, switch.bdf()), ("NVSwitch_gen3", BDF));
    }

    #[rstest]
    #[case::another_vendor(0x1af4, NVSWITCH_GEN3, "is not NVIDIA")]
    #[case::bar0_inaccessible(0x10de, 0xffff_ffff, "BAR0 not accessible")]
    #[case::limerock(0x10de, 0x6000a1, "not a PPCIE-capable NVSwitch")]
    fn opening_refuses_what_is_not_a_ppcie_capable_switch(
        fake: Fake,
        #[case] vendor: u16,
        #[case] boot0: u32,
        #[case] expected: &str,
    ) {
        fake.add_mappable_device(BDF, vendor, NVSWITCH, NVSWITCH_CLASS, BAR0_LEN);
        fake.set_register(BDF, NV_PMC_BOOT_0 as u64, boot0);

        let err = why(NvSwitch::open_in(&fake.sysfs, BDF));

        assert!(err.contains(expected), "{err}");
    }

    #[rstest]
    fn switch_enumeration_opens_every_switch_it_discovers(fake: Fake) {
        fake.add_mappable_device(BDF, 0x10de, NVSWITCH, NVSWITCH_CLASS, BAR0_LEN);
        fake.set_register(BDF, NV_PMC_BOOT_0 as u64, NVSWITCH_GEN3);

        let switches = NvSwitch::enumerate(&fake.sysfs).unwrap();

        assert_eq!(switches.len(), 1);
        assert_eq!(switches[0].bdf(), BDF);
    }

    #[rstest]
    #[case::off(0x0000_0000, PpcieMode::Off)]
    #[case::on(0x0000_0001, PpcieMode::On)]
    fn a_switch_reports_the_ppcie_mode_it_is_running(
        #[case] state: u32,
        #[case] expected: PpcieMode,
    ) {
        let (_fake, firmware, switch) = modelled_switch();
        firmware.set_register(PPCIE_STATE_SWITCH, state);

        assert_eq!(switch.query_ppcie_mode().unwrap(), expected);
    }

    #[rstest]
    fn setting_ppcie_on_a_switch_leaves_the_decoupler_off() {
        let (_fake, firmware, switch) = modelled_switch();

        switch.set_ppcie_mode(PpcieMode::On).unwrap();

        assert_eq!(firmware.knob(fsp::KNOB_PPCIE), Some(1));
        assert_eq!(firmware.knob(fsp::KNOB_BAR0_DECOUPLER), Some(0));
    }

    #[rstest]
    fn resetting_a_switch_goes_through_sysfs_and_waits_for_boot() {
        let (fake, _firmware, switch) = modelled_switch();

        switch.reset().unwrap();

        assert_eq!(
            std::fs::read_to_string(fake.device(BDF).join("reset")).unwrap(),
            "1"
        );
    }
}
