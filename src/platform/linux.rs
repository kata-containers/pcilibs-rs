// Copyright (c) 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

use crate::nvlink::ManagementFunction;
use crate::{attr_hex, context, failed, normalize_bdf, nvlink, Sysfs};
use std::{fs, io};

const NVIDIA: u32 = 0x10de;
const MELLANOX: u32 = 0x15b3;

/// Lists are sorted by canonical BDF so boot-time selection is deterministic.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Topology {
    /// Include pre-CC GPUs so init policy does not depend on CC support.
    pub gpus: Vec<String>,
    /// Bridge class identifies candidates; BAR0 must confirm their generation.
    pub switches: Vec<String>,
    /// Include sibling PFs because full-capability PFs may lack the VPD marker.
    pub management_functions: Vec<ManagementFunction>,
}

impl Topology {
    pub fn fabric_interface(&self) -> crate::platform::FabricInterface {
        crate::platform::FabricInterface::from_presence(
            !self.switches.is_empty(),
            !self.management_functions.is_empty(),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DetectedPlatform {
    pub topology: Topology,
    pub platform: crate::platform::Platform,
}

/// Keep family evidence beside the management interface when GPUs are absent.
pub fn discover(sysfs: &Sysfs) -> io::Result<DetectedPlatform> {
    discover_with_catalog(sysfs, crate::gpu::catalog::Catalog::builtin())
}

pub fn discover_with_catalog(
    sysfs: &Sysfs,
    catalog: crate::gpu::catalog::Catalog<'_>,
) -> io::Result<DetectedPlatform> {
    let topology = discover_topology(sysfs)?;
    let gpus = topology
        .gpus
        .iter()
        .map(|bdf| {
            let path = sysfs.devices().join(bdf);
            let read_id = |name| {
                u16::try_from(attr_hex(&path, name)?).map_err(|_| {
                    failed(
                        io::ErrorKind::InvalidData,
                        format!("{}: {name} exceeds 16 bits", path.display()),
                    )
                })
            };
            Ok(crate::platform::GpuIdentity {
                device: read_id("device")?,
                subsystem_device: read_id("subsystem_device")?,
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let platform =
        crate::platform::classify_with_catalog(gpus, topology.fabric_interface(), catalog);
    Ok(DetectedPlatform { topology, platform })
}

pub(crate) struct PciFunction {
    pub bdf: String,
    pub vendor: u32,
    pub class: u32,
}

pub(crate) fn pci_functions(sysfs: &Sysfs) -> io::Result<Vec<PciFunction>> {
    let root = sysfs.devices();
    let entries = fs::read_dir(&root).map_err(|e| context(e, root.display()))?;
    let mut functions = Vec::new();
    for entry in entries {
        let entry = entry?;
        let Some(bdf) = entry.file_name().to_str().and_then(normalize_bdf) else {
            continue;
        };
        let path = root.join(&bdf);
        let vendor = attr_hex(&path, "vendor")?;
        if matches!(vendor, NVIDIA | MELLANOX) {
            functions.push(PciFunction {
                bdf,
                vendor,
                class: attr_hex(&path, "class")?,
            });
        }
    }
    functions.sort_by(|a, b| a.bdf.cmp(&b.bdf));
    Ok(functions)
}

fn is_gpu(function: &PciFunction) -> bool {
    function.vendor == NVIDIA && matches!(function.class >> 8, 0x0300 | 0x0302)
}

/// Init needs GPU BDFs before loading drivers, without BAR or VPD access.
/// Identity errors propagate so a failed scan cannot select CPU-only mode.
pub fn discover_gpus(sysfs: &Sysfs) -> io::Result<Vec<String>> {
    Ok(pci_functions(sysfs)?
        .into_iter()
        .filter(is_gpu)
        .map(|device| device.bdf)
        .collect())
}

/// PCI inventory exists independently of NVLink switches or RDMA drivers.
/// VPD may require root; errors propagate to avoid selecting the wrong startup mode.
pub fn discover_topology(sysfs: &Sysfs) -> io::Result<Topology> {
    let functions = pci_functions(sysfs)?;
    let gpus = functions
        .iter()
        .filter(|function| is_gpu(function))
        .map(|function| function.bdf.clone())
        .collect();
    let (switches, management_functions) = nvlink::discover_fabric(sysfs, &functions)?;
    Ok(Topology {
        gpus,
        switches,
        management_functions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs::{self, Fake};

    fn mlx(fake: &Fake, bdf: &str, marker: bool) {
        fake.add_pci_device(bdf, 0x15b3, 0x1021, 0x020700, None);
        if marker {
            fs::write(
                fake.device(bdf).join("vpd"),
                b"\x90\x0e\x00VA\x0bSMDL=SW_MNG\x78",
            )
            .unwrap();
        }
    }

    #[test]
    fn platform_preserves_the_service_vm_interface_without_gpus() {
        let fake = testfs::fake();
        mlx(&fake, "0000:05:00.0", true);
        let detected = discover(&fake.sysfs).unwrap();
        assert_eq!(detected.platform.kind, crate::platform::Kind::Unknown);
        assert_eq!(
            detected.platform.fabric,
            crate::platform::FabricInterface::ConnectX
        );
        assert_eq!(detected.platform.gpu_count, 0);
        assert_eq!(detected.topology.management_functions.len(), 1);
    }

    #[test]
    fn subsystem_identity_refines_the_managed_fabric_family() {
        let fake = testfs::fake();
        mlx(&fake, "0000:05:00.0", true);
        let bdf = "0000:40:00.0";
        fake.add_pci_device(bdf, 0x10de, 0x3002, 0x030200, None);
        fs::write(fake.device(bdf).join("subsystem_device"), "0x2277").unwrap();
        assert_eq!(
            discover(&fake.sysfs).unwrap().platform.kind,
            crate::platform::Kind::HgxRx00
        );
        fs::write(fake.device(bdf).join("device"), "0x2941").unwrap();
        fs::write(fake.device(bdf).join("subsystem_device"), "0x2046").unwrap();
        assert_eq!(
            discover(&fake.sysfs).unwrap().platform.kind,
            crate::platform::Kind::Coherent(crate::gpu::Family::Blackwell)
        );
    }

    #[test]
    fn missing_or_invalid_gpu_subsystem_identity_is_an_error() {
        let fake = testfs::fake();
        let bdf = "0000:40:00.0";
        fake.add_pci_device(bdf, 0x10de, 0x3002, 0x030200, None);
        assert!(discover(&fake.sysfs).is_err());
        for value in ["not hex", "0x10000"] {
            fs::write(fake.device(bdf).join("subsystem_device"), value).unwrap();
            assert!(discover(&fake.sysfs).is_err());
        }
        fs::write(fake.device(bdf).join("subsystem_device"), "0x2277").unwrap();
        fs::write(fake.device(bdf).join("device"), "0x10000").unwrap();
        assert!(discover(&fake.sysfs).is_err());
    }

    #[test]
    fn pcie_only_gpu_needs_neither_switches_nor_an_rdma_tree() {
        let fake = testfs::fake();
        fake.add_pci_device("0000:01:00.0", 0x10de, 0x2331, 0x030200, None);
        fs::write(
            fake.device("0000:01:00.0").join("subsystem_device"),
            "0x1626",
        )
        .unwrap();
        let detected = discover(&fake.sysfs).unwrap();
        assert_eq!(
            detected.platform.kind,
            crate::platform::Kind::Pcie(crate::gpu::Family::Hopper)
        );
        assert_eq!(
            detected.platform.fabric,
            crate::platform::FabricInterface::None
        );
        assert_eq!(detected.topology.gpus, ["0000:01:00.0"]);
        assert!(detected.topology.switches.is_empty());
        assert!(detected.topology.management_functions.is_empty());
        assert!(!fake.sysfs.infiniband().exists());
    }

    #[test]
    fn non_nvidia_pcie_system_has_no_nvidia_fabric() {
        let fake = testfs::fake();
        fake.add_pci_device("0000:01:00.0", 0x8086, 0x1234, 0x020000, None);
        let detected = discover(&fake.sysfs).unwrap();
        assert_eq!(detected.platform.gpu_count, 0);
        assert_eq!(
            detected.platform.fabric,
            crate::platform::FabricInterface::None
        );
        assert_eq!(detected.platform.kind, crate::platform::Kind::Unknown);
    }
    #[test]
    fn loaded_extensions_reach_sysfs_discovery_without_changing_defaults() {
        let fake = testfs::fake();
        mlx(&fake, "0000:05:00.0", true);
        let bdf = "0000:40:00.0";
        fake.add_pci_device(bdf, 0x10de, 0xffff, 0x030200, None);
        fs::write(fake.device(bdf).join("subsystem_device"), "0x1234").unwrap();
        let path = fake.root().join("gpus.catalog");
        fs::write(&path, "ffff 1234 GR100 sxm\n").unwrap();
        let file = crate::gpu::catalog::CatalogFile::read(&path).unwrap();
        assert_eq!(
            discover_with_catalog(&fake.sysfs, file.catalog())
                .unwrap()
                .platform
                .kind,
            crate::platform::Kind::HgxRx00
        );
        assert_eq!(
            discover(&fake.sysfs).unwrap().platform.kind,
            crate::platform::Kind::Unknown
        );
        fs::write(fake.device(bdf).join("subsystem_device"), "0x1235").unwrap();
        assert_eq!(
            discover_with_catalog(&fake.sysfs, file.catalog())
                .unwrap()
                .platform
                .kind,
            crate::platform::Kind::Unknown
        );
    }
}
