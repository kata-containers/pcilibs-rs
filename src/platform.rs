// Copyright (c) 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA accelerator platform classification, independent of the system OEM.
//! HGX variants describe hardware profiles shared by HGX-based OEM/DGX systems;
//! PCI evidence cannot establish an exact chassis model or NVL72 rack membership.
//! The classifier uses neither an allocator nor firmware/driver access.

use crate::gpu::{self, Attachment, Family};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Unknown,
    HgxHx00,
    HgxBx00,
    HgxRx00,
    Coherent(Family),
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FabricInterface {
    None,
    DirectNvSwitch,
    ConnectX,
    Mixed,
}

impl FabricInterface {
    pub const fn from_presence(direct_switches: bool, management_pfs: bool) -> Self {
        match (direct_switches, management_pfs) {
            (false, false) => Self::None,
            (true, false) => Self::DirectNvSwitch,
            (false, true) => Self::ConnectX,
            (true, true) => Self::Mixed,
        }
    }
}

/// Only NVIDIA VGA/3D PCI functions belong in the classifier input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuIdentity {
    pub device: u16,
    pub subsystem_device: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Platform {
    pub kind: Kind,
    pub fabric: FabricInterface,
    pub gpu_count: usize,
}

/// A missing/unknown GPU identity cannot prove Bx00 versus Rx00 in a ServiceVM.
/// Partial assignments need no fixed GPU or management-PF count.
pub fn classify(gpus: impl IntoIterator<Item = GpuIdentity>, fabric: FabricInterface) -> Platform {
    let mut first = None;
    let mut unknown = false;
    let mut mixed = false;
    let mut gpu_count = 0;
    for gpu in gpus {
        gpu_count += 1;
        let Some(chip) = gpu::chip_for(gpu.device) else {
            unknown = true;
            continue;
        };
        let attachment = gpu::attachment(gpu.device, gpu.subsystem_device);
        if attachment == Attachment::Unknown {
            unknown = true;
            continue;
        }
        let identity = (chip.family, attachment);
        match first {
            None => first = Some(identity),
            Some(previous) => mixed |= previous != identity,
        }
    }
    let kind = if mixed || fabric == FabricInterface::Mixed {
        Kind::Mixed
    } else if unknown {
        Kind::Unknown
    } else {
        match (first, fabric) {
            (Some((family, Attachment::Coherent)), _) => Kind::Coherent(family),
            (Some((Family::Hopper, Attachment::Sxm)), FabricInterface::DirectNvSwitch) => {
                Kind::HgxHx00
            }
            (Some((Family::Blackwell, Attachment::Sxm)), FabricInterface::ConnectX) => {
                Kind::HgxBx00
            }
            (Some((Family::Rubin, Attachment::Sxm)), FabricInterface::ConnectX) => Kind::HgxRx00,
            _ => Kind::Unknown,
        }
    };
    Platform {
        kind,
        fabric,
        gpu_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(device: u16, subsystem_device: u16) -> GpuIdentity {
        GpuIdentity {
            device,
            subsystem_device,
        }
    }

    #[test]
    fn hgx_profiles_require_matching_gpu_attachment_and_interface() {
        for (identity, fabric, expected) in [
            (
                gpu(0x2330, 0x16c0),
                FabricInterface::DirectNvSwitch,
                Kind::HgxHx00,
            ),
            (
                gpu(0x2335, 0x18be),
                FabricInterface::DirectNvSwitch,
                Kind::HgxHx00,
            ),
            (
                gpu(0x2901, 0x1999),
                FabricInterface::ConnectX,
                Kind::HgxBx00,
            ),
            (
                gpu(0x3182, 0x20e5),
                FabricInterface::ConnectX,
                Kind::HgxBx00,
            ),
            (
                gpu(0x3002, 0x2277),
                FabricInterface::ConnectX,
                Kind::HgxRx00,
            ),
        ] {
            for count in [1, 4, 8] {
                assert_eq!(
                    classify(core::iter::repeat_n(identity, count), fabric),
                    Platform {
                        kind: expected,
                        fabric,
                        gpu_count: count
                    }
                );
            }
        }
    }

    #[test]
    fn coherent_attachment_preserves_the_fabric_interface() {
        for (identity, family) in [
            (gpu(0x2342, 0x16eb), Family::Hopper),
            (gpu(0x2941, 0x2046), Family::Blackwell),
            (gpu(0x3041, 0x221a), Family::Rubin),
            (gpu(0x307e, 0x221a), Family::Rubin),
            (gpu(0x30ff, 0x221b), Family::Rubin),
        ] {
            for fabric in [FabricInterface::None, FabricInterface::ConnectX] {
                let platform = classify([identity], fabric);
                assert_eq!(platform.kind, Kind::Coherent(family));
                assert_eq!(platform.fabric, fabric);
            }
        }
    }

    #[test]
    fn unknown_and_pcie_parts_do_not_prove_an_hgx_platform() {
        for identities in [
            [gpu(0xffff, 0xffff)],
            [gpu(0x2901, 0xffff)],
            [gpu(0x2331, 0x1626)],
        ] {
            assert_eq!(
                classify(identities, FabricInterface::DirectNvSwitch).kind,
                Kind::Unknown
            );
        }
        assert_eq!(
            classify([gpu(0x2901, 0x1999)], FabricInterface::None).kind,
            Kind::Unknown
        );
        assert_eq!(
            classify([gpu(0x2330, 0x16c0)], FabricInterface::ConnectX).kind,
            Kind::Unknown
        );
        for identities in [
            [gpu(0xffff, 0), gpu(0x2901, 0x1999)],
            [gpu(0x2901, 0x1999), gpu(0xffff, 0)],
        ] {
            assert_eq!(
                classify(identities, FabricInterface::ConnectX).kind,
                Kind::Unknown
            );
        }
    }

    #[test]
    fn management_only_assignment_keeps_the_family_unknown() {
        for fabric in [
            FabricInterface::None,
            FabricInterface::DirectNvSwitch,
            FabricInterface::ConnectX,
        ] {
            assert_eq!(
                classify([], fabric),
                Platform {
                    kind: Kind::Unknown,
                    fabric,
                    gpu_count: 0
                }
            );
        }
    }

    #[test]
    fn mixed_families_attachments_and_interfaces_are_explicit() {
        assert_eq!(
            classify(
                [gpu(0x2330, 0x16c0), gpu(0x2901, 0x1999)],
                FabricInterface::ConnectX
            )
            .kind,
            Kind::Mixed
        );
        assert_eq!(
            classify(
                [gpu(0x29bc, 0x1985), gpu(0x29bc, 0x2045)],
                FabricInterface::ConnectX
            )
            .kind,
            Kind::Mixed
        );
        assert_eq!(classify([], FabricInterface::Mixed).kind, Kind::Mixed);
    }

    #[test]
    fn interface_depends_on_presence() {
        assert_eq!(
            FabricInterface::from_presence(false, false),
            FabricInterface::None
        );
        assert_eq!(
            FabricInterface::from_presence(true, false),
            FabricInterface::DirectNvSwitch
        );
        assert_eq!(
            FabricInterface::from_presence(false, true),
            FabricInterface::ConnectX
        );
        assert_eq!(
            FabricInterface::from_presence(true, true),
            FabricInterface::Mixed
        );
    }
}
