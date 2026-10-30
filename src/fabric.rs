// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! The NVLink fabric partition view: what Fabric Manager in a Service VM
//! publishes, and how a host maps it onto the GPUs it can pass through.
//!
//! In shared NVSwitch mode (HGX guide 19.0, §1.2 and §6.2) Fabric Manager
//! knows a GPU only by its physical ID, the module slot on the baseboard: it
//! has no PCI view of the host and, holding no GPUs, no UUIDs either.  The
//! host knows a GPU only by PCI address.  The bridge is the firmware's slot
//! designation, which the kernel exposes as the function's `label`
//! attribute ("GPU0, SMX7" on DGX systems): its slot number is the physical
//! ID.  Verified on a DGX NVL8 against Fabric Manager's own in-band probe
//! records, module by module.  OEM firmware may label differently, and a GPU
//! without a parsable label resolves to no partition, so an unknown board
//! degrades to nothing advertised, never to a wrong partition.
//!
//! The wire types in [`v1`] are generated from
//! `proto/kata/fabric/v1/fabric.proto` with serde derives, so the JSON a
//! client prints and the message it streams are one schema.  Field names
//! follow NVIDIA's `fmpm` sample client (`partitionId`, `isActive`,
//! `physicalId`), so the guide's shell snippets keep working.
//!
//! ```no_run
//! use pcilibs_rs::{fabric, platform, Sysfs};
//!
//! # fn main() -> std::io::Result<()> {
//! let sysfs = Sysfs::default();
//! let topology = platform::discover_topology(&sysfs)?;
//! let gpus = fabric::gpu_modules(&sysfs, &topology.gpus)?;
//! # let view = fabric::v1::FabricView::default();
//! if let Some(resolved) = fabric::resolve(&view, &gpus) {
//!     for partition in resolved.partitions.iter().filter(|p| p.advertisable()) {
//!         println!("{}: {:?}", partition.id, partition.bdfs().collect::<Vec<_>>());
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;

use crate::platform::GpuFunction;
use crate::{context, Sysfs};

pub mod v1 {
    //! Generated from `proto/kata/fabric/v1/fabric.proto`.
    include!(concat!(env!("OUT_DIR"), "/kata.fabric.v1.rs"));
}

/// Environment variable naming the endpoint to publish to.  The device
/// plugin injects it, with [`TOKEN_ENV`], into the one container it gives
/// the fabric to (device plugin `Allocate`, `envs`), so only the fabric
/// holder learns where to publish and with which token.
pub const ENDPOINT_ENV: &str = "KATA_FABRIC_VIEW_ENDPOINT";
/// Environment variable carrying the token; see [`ENDPOINT_ENV`].
pub const TOKEN_ENV: &str = "KATA_FABRIC_VIEW_TOKEN";
/// gRPC metadata key the token travels under on every stream.
pub const TOKEN_METADATA_KEY: &str = "kata-fabric-view-token";
/// Where the plugin listens for a publisher on the host itself; filesystem
/// permissions are the credential there.
pub const HOST_SOCKET: &str = "/var/run/kata-device-plugin/fabric-view.sock";

/// How the fabric owner wants the GPUs of active partitions consumed; the
/// `unit` word of a [`v1::FabricView`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Unit {
    /// One schedulable unit per active partition, taken whole.  The default,
    /// because tenants that share an active partition can reach each other
    /// over NVLink.
    #[default]
    Partition,
    /// One unit per GPU, any count: bare-metal semantics, for an owner who
    /// holds the whole node.
    Gpu,
}

impl Unit {
    /// The word as it travels in the view.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Partition => "partition",
            Self::Gpu => "gpu",
        }
    }

    /// An unset word is the default; a word this crate does not know is
    /// `None`, so a view from a newer publisher is refused whole rather
    /// than interpreted partially.
    pub fn from_view(word: &str) -> Option<Self> {
        match word {
            "" | "partition" => Some(Self::Partition),
            "gpu" => Some(Self::Gpu),
            _ => None,
        }
    }
}

/// The module slot in a firmware slot label: 7 for "GPU0, SMX7".  Accepts
/// `SXM` and the `SMX` spelling DGX firmware uses, upper or lower case,
/// with the number attached or as the next token.  Slots are 1-based and
/// Fabric Manager addresses at most 16 GPUs; anything else is no slot.
pub fn parse_module_label(label: &str) -> Option<u8> {
    let mut tokens = label
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty());
    while let Some(token) = tokens.next() {
        let upper = token.to_ascii_uppercase();
        let Some(rest) = upper
            .strip_prefix("SXM")
            .or_else(|| upper.strip_prefix("SMX"))
        else {
            continue;
        };
        let digits = if rest.is_empty() {
            tokens.next()?
        } else {
            rest
        };
        return digits.parse().ok().filter(|slot| (1..=16).contains(slot));
    }
    None
}

/// The module slot of the GPU at `bdf`: `None` when the firmware labels the
/// function with no slot, or not at all.  Other read errors propagate, so an
/// unreadable sysfs never looks like an unlabeled board.
pub fn module_slot(sysfs: &Sysfs, bdf: &str) -> io::Result<Option<u8>> {
    let path = sysfs.devices().join(bdf).join("label");
    match fs::read_to_string(&path) {
        Ok(label) => Ok(parse_module_label(&label)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(context(e, path.display())),
    }
}

/// A GPU function and the module slot its firmware label names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuModule {
    pub bdf: String,
    pub module: Option<u8>,
}

/// The module slot of every GPU in `gpus`, in their order.
pub fn gpu_modules(sysfs: &Sysfs, gpus: &[GpuFunction]) -> io::Result<Vec<GpuModule>> {
    gpus.iter()
        .map(|gpu| {
            Ok(GpuModule {
                bdf: gpu.bdf.clone(),
                module: module_slot(sysfs, &gpu.bdf)?,
            })
        })
        .collect()
}

/// One GPU of a partition, as seen from this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub physical_id: u32,
    /// The function in that module slot, `None` when this host has no
    /// labeled GPU there.
    pub bdf: Option<String>,
}

/// A partition of the view, resolved against this host's GPUs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPartition {
    pub id: u32,
    pub active: bool,
    /// Absent from Fabric Manager's unsupported list, where partitions land
    /// when their NVLinks failed.
    pub supported: bool,
    /// In the view's order, which is Fabric Manager's.
    pub members: Vec<Member>,
}

impl ResolvedPartition {
    /// Every member GPU is on this host.  A partition without members is
    /// never complete: there is nothing to hand out.
    pub fn complete(&self) -> bool {
        !self.members.is_empty() && self.members.iter().all(|member| member.bdf.is_some())
    }

    /// A unit a scheduler may hand out: active, supported, and all here.
    pub fn advertisable(&self) -> bool {
        self.active && self.supported && self.complete()
    }

    /// The PCI addresses of the members this host has, in member order.
    pub fn bdfs(&self) -> impl Iterator<Item = &str> {
        self.members
            .iter()
            .filter_map(|member| member.bdf.as_deref())
    }
}

/// A view mapped onto this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub unit: Unit,
    /// By partition id.
    pub partitions: Vec<ResolvedPartition>,
}

/// Map a published view onto this host's GPUs.  A module slot two GPUs
/// claim identifies neither, so both resolve to nothing; a slot no GPU
/// claims leaves its partitions incomplete.  `None` when the view's unit
/// word is not one this crate knows.
pub fn resolve(view: &v1::FabricView, gpus: &[GpuModule]) -> Option<Resolved> {
    let unit = Unit::from_view(&view.unit)?;
    let mut by_module: BTreeMap<u8, Option<&str>> = BTreeMap::new();
    for gpu in gpus {
        if let Some(module) = gpu.module {
            by_module
                .entry(module)
                .and_modify(|bdf| *bdf = None)
                .or_insert(Some(&gpu.bdf));
        }
    }
    let unsupported: BTreeSet<u32> = view
        .unsupported
        .iter()
        .flat_map(|list| list.partition_info.iter())
        .map(|partition| partition.partition_id)
        .collect();
    let partitions: BTreeMap<u32, ResolvedPartition> = view
        .partitions
        .iter()
        .flat_map(|list| list.partition_info.iter())
        .map(|partition| {
            let members = partition
                .gpu_info
                .iter()
                .map(|gpu| Member {
                    physical_id: gpu.physical_id,
                    bdf: u8::try_from(gpu.physical_id)
                        .ok()
                        .and_then(|module| by_module.get(&module).copied().flatten())
                        .map(str::to_owned),
                })
                .collect();
            (
                partition.partition_id,
                ResolvedPartition {
                    id: partition.partition_id,
                    active: partition.is_active,
                    supported: !unsupported.contains(&partition.partition_id),
                    members,
                },
            )
        })
        .collect();
    Some(Resolved {
        unit,
        partitions: partitions.into_values().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs;

    /// This DGX NVL8's table: 1 is the node, 2 and 3 the halves, 4 to 7 the
    /// pairs (1,2) (3,4) (5,6) (7,8), 8 to 15 the singles.  Partition 1 is
    /// active.
    const RUBIN: &str = include_str!("../testdata/fabric/dgx-rubin-nvl8-partitions.json");

    /// The same box's GPUs with their firmware labels; SMX1 is absent, its
    /// PCIe branch having failed.
    fn rubin_gpus() -> Vec<GpuModule> {
        [
            ("0000:1a:00.0", 7),
            ("0000:3c:00.0", 6),
            ("0000:62:00.0", 8),
            ("0000:73:00.0", 5),
            ("0000:9a:00.0", 3),
            ("0000:bc:00.0", 2),
            ("0000:df:00.0", 4),
        ]
        .into_iter()
        .map(|(bdf, module)| GpuModule {
            bdf: bdf.to_owned(),
            module: Some(module),
        })
        .collect()
    }

    fn rubin_view(unit: &str) -> v1::FabricView {
        v1::FabricView {
            partitions: Some(serde_json::from_str(RUBIN).unwrap()),
            unsupported: None,
            unit: unit.to_owned(),
        }
    }

    fn partition(id: u32, active: bool, gpus: &[u32]) -> v1::Partition {
        v1::Partition {
            partition_id: id,
            is_active: active,
            gpu_info: gpus
                .iter()
                .map(|&physical_id| v1::PartitionGpu {
                    physical_id,
                    ..Default::default()
                })
                .collect(),
        }
    }

    /// HGX H100 as the guide prints it (19.0, §6.2 and §6.3): 0-based ids,
    /// pairs are PCIe siblings, and nvidia-smi's module ids by bus.
    fn h100_view() -> v1::FabricView {
        let mut partition_info = vec![
            partition(0, false, &[8, 7, 6, 5, 4, 3, 2, 1]),
            partition(1, true, &[4, 3, 2, 1]),
            partition(2, false, &[8, 7, 6, 5]),
            partition(3, false, &[3, 1]),
            partition(4, false, &[4, 2]),
            partition(5, true, &[7, 5]),
            partition(6, false, &[8, 6]),
        ];
        for gpu in 1..=8 {
            partition_info.push(partition(6 + gpu, matches!(gpu, 6 | 8), &[gpu]));
        }
        v1::FabricView {
            partitions: Some(v1::PartitionList {
                version: 1,
                max_num_partitions: 15,
                partition_info,
            }),
            unsupported: None,
            unit: String::new(),
        }
    }

    fn h100_gpus() -> Vec<GpuModule> {
        [
            ("0f", 5),
            ("10", 7),
            ("41", 6),
            ("44", 8),
            ("86", 1),
            ("87", 3),
            ("b8", 2),
            ("bb", 4),
        ]
        .into_iter()
        .map(|(bus, module)| GpuModule {
            bdf: format!("0000:{bus}:00.0"),
            module: Some(module),
        })
        .collect()
    }

    fn advertisable(resolved: &Resolved) -> Vec<(u32, Vec<&str>)> {
        resolved
            .partitions
            .iter()
            .filter(|p| p.advertisable())
            .map(|p| (p.id, p.bdfs().collect()))
            .collect()
    }

    #[test]
    fn module_label_spellings() {
        for (label, slot) in [
            ("GPU0, SMX7", Some(7)),
            ("GPU 7, SMX1\n", Some(1)),
            ("GPU_SXM_3", Some(3)),
            ("sxm8", Some(8)),
            ("SXM 16", Some(16)),
            ("GPU0", None),
            ("", None),
            ("SMX0", None),
            ("SMX17", None),
            ("SMX", None),
            ("SMXx", None),
            ("SMX 1x", None),
        ] {
            assert_eq!(parse_module_label(label), slot, "{label:?}");
        }
    }

    #[test]
    fn module_slot_reads_the_firmware_label() {
        let fake = testfs::fake();
        let labeled = "0000:1a:00.0";
        let unlabeled = "0000:3c:00.0";
        let unreadable = "0000:62:00.0";
        for bdf in [labeled, unlabeled, unreadable] {
            fake.add_pci_device(bdf, 0x10de, 0x303f, 0x030200, None);
        }
        fs::write(fake.device(labeled).join("label"), "GPU0, SMX7\n").unwrap();
        // A directory where the attribute should be fails the read the way
        // an unreadable attribute does.
        fs::create_dir(fake.device(unreadable).join("label")).unwrap();

        assert_eq!(module_slot(&fake.sysfs, labeled).unwrap(), Some(7));
        assert_eq!(module_slot(&fake.sysfs, unlabeled).unwrap(), None);
        assert!(module_slot(&fake.sysfs, unreadable).is_err());

        let gpus: Vec<GpuFunction> = [labeled, unlabeled]
            .into_iter()
            .map(|bdf| GpuFunction {
                bdf: bdf.to_owned(),
                vendor: 0x10de,
            })
            .collect();
        assert_eq!(
            gpu_modules(&fake.sysfs, &gpus).unwrap(),
            [
                GpuModule {
                    bdf: labeled.to_owned(),
                    module: Some(7)
                },
                GpuModule {
                    bdf: unlabeled.to_owned(),
                    module: None
                },
            ]
        );
        assert!(gpu_modules(
            &fake.sysfs,
            &[GpuFunction {
                bdf: unreadable.to_owned(),
                vendor: 0x10de
            }]
        )
        .is_err());
    }

    #[test]
    fn unit_words() {
        assert_eq!(Unit::from_view(""), Some(Unit::Partition));
        assert_eq!(Unit::from_view("partition"), Some(Unit::Partition));
        assert_eq!(Unit::from_view("gpu"), Some(Unit::Gpu));
        assert_eq!(Unit::from_view("GPU"), None);
        assert_eq!(Unit::default(), Unit::Partition);
        for unit in [Unit::Partition, Unit::Gpu] {
            assert_eq!(Unit::from_view(unit.as_str()), Some(unit));
        }
    }

    #[test]
    fn json_keeps_fmpm_names() {
        let list: v1::PartitionList = serde_json::from_str(RUBIN).unwrap();
        assert_eq!(list.partition_info.len(), 15);
        assert_eq!(list.max_num_partitions, 15);
        let node = &list.partition_info[0];
        assert!(node.is_active);
        assert_eq!(
            node.gpu_info
                .iter()
                .map(|g| g.physical_id)
                .collect::<Vec<_>>(),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(node.gpu_info[0].num_nvlinks_available, 36);
        assert_eq!(node.gpu_info[0].nvlink_line_rate_mbps, 41000);

        let json = serde_json::to_value(rubin_view("gpu")).unwrap();
        let gpu = &json["partitions"]["partitionInfo"][0]["gpuInfo"][0];
        for key in [
            "physicalId",
            "uuid",
            "pciBusId",
            "numNvLinksAvailable",
            "maxNumNvLinks",
            "nvlinkLineRateMBps",
        ] {
            assert!(gpu.get(key).is_some(), "{key} missing: {gpu}");
        }
        assert_eq!(json["partitions"]["partitionInfo"][0]["isActive"], true);
        assert_eq!(json["unit"], "gpu");
        assert!(json["unsupported"].is_null());
    }

    #[test]
    fn rubin_node_partition_is_incomplete_without_smx1() {
        let resolved = resolve(&rubin_view(""), &rubin_gpus()).unwrap();
        assert_eq!(resolved.unit, Unit::Partition);
        assert_eq!(resolved.partitions.len(), 15);
        let node = &resolved.partitions[0];
        assert!(node.active && node.supported && !node.complete());
        assert_eq!(node.members[0].bdf, None);
        assert_eq!(node.bdfs().count(), 7);
        // Nothing active is complete, nothing complete is active.
        assert!(advertisable(&resolved).is_empty());
        assert!(resolved.partitions[2].complete());
        assert!(!resolved.partitions[2].active);
    }

    #[test]
    fn rubin_half_and_pair_resolve_in_fabric_manager_order() {
        let mut view = rubin_view("partition");
        let list = view.partitions.as_mut().unwrap();
        for p in &mut list.partition_info {
            p.is_active = matches!(p.partition_id, 3 | 5);
        }
        let resolved = resolve(&view, &rubin_gpus()).unwrap();
        assert_eq!(
            advertisable(&resolved),
            [
                (
                    3,
                    vec![
                        "0000:73:00.0",
                        "0000:3c:00.0",
                        "0000:1a:00.0",
                        "0000:62:00.0"
                    ]
                ),
                (5, vec!["0000:9a:00.0", "0000:df:00.0"]),
            ]
        );
    }

    #[test]
    fn h100_guide_table_resolves_by_module_id() {
        let resolved = resolve(&h100_view(), &h100_gpus()).unwrap();
        assert_eq!(
            advertisable(&resolved),
            [
                (
                    1,
                    vec![
                        "0000:bb:00.0",
                        "0000:87:00.0",
                        "0000:b8:00.0",
                        "0000:86:00.0"
                    ]
                ),
                (5, vec!["0000:10:00.0", "0000:0f:00.0"]),
                (12, vec!["0000:41:00.0"]),
                (14, vec!["0000:44:00.0"]),
            ]
        );
    }

    #[test]
    fn unsupported_partitions_are_withheld() {
        let mut view = h100_view();
        view.unsupported = Some(v1::UnsupportedPartitionList {
            version: 1,
            partition_info: vec![v1::UnsupportedPartition {
                partition_id: 5,
                gpu_physical_ids: vec![7, 5],
            }],
        });
        let resolved = resolve(&view, &h100_gpus()).unwrap();
        let five = resolved.partitions.iter().find(|p| p.id == 5).unwrap();
        assert!(five.active && five.complete() && !five.supported);
        assert_eq!(
            advertisable(&resolved)
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>(),
            [1, 12, 14]
        );
    }

    #[test]
    fn a_slot_two_gpus_claim_identifies_neither() {
        let mut gpus = h100_gpus();
        gpus.push(GpuModule {
            bdf: "0000:c0:00.0".to_owned(),
            module: Some(1),
        });
        gpus.push(GpuModule {
            bdf: "0000:c1:00.0".to_owned(),
            module: None,
        });
        let resolved = resolve(&h100_view(), &gpus).unwrap();
        let one = resolved.partitions.iter().find(|p| p.id == 1).unwrap();
        assert!(!one.complete());
        assert_eq!(one.members[3].physical_id, 1);
        assert_eq!(one.members[3].bdf, None);
        assert_eq!(
            advertisable(&resolved)
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>(),
            [5, 12, 14]
        );
    }

    #[test]
    fn unknown_unit_refuses_the_view() {
        assert!(resolve(&rubin_view("tray"), &rubin_gpus()).is_none());
        assert_eq!(
            resolve(&rubin_view("gpu"), &rubin_gpus()).unwrap().unit,
            Unit::Gpu
        );
    }

    #[test]
    fn empty_and_degenerate_views() {
        let empty = resolve(&v1::FabricView::default(), &h100_gpus()).unwrap();
        assert_eq!(empty.unit, Unit::Partition);
        assert!(empty.partitions.is_empty());

        // No members: nothing to hand out, so never complete.  A physical id
        // beyond any slot resolves to no function.  A repeated id keeps the
        // last definition, as a map would.
        let view = v1::FabricView {
            partitions: Some(v1::PartitionList {
                version: 1,
                max_num_partitions: 3,
                partition_info: vec![
                    partition(7, true, &[]),
                    partition(8, true, &[300]),
                    partition(9, false, &[1]),
                    partition(9, true, &[2]),
                ],
            }),
            unsupported: None,
            unit: String::new(),
        };
        let resolved = resolve(&view, &h100_gpus()).unwrap();
        assert_eq!(
            resolved.partitions.iter().map(|p| p.id).collect::<Vec<_>>(),
            [7, 8, 9]
        );
        assert!(!resolved.partitions[0].complete());
        assert_eq!(resolved.partitions[1].members[0].bdf, None);
        assert_eq!(
            resolved.partitions[2].bdfs().collect::<Vec<_>>(),
            ["0000:b8:00.0"]
        );
        assert_eq!(advertisable(&resolved), [(9, vec!["0000:b8:00.0"])]);
    }
}
