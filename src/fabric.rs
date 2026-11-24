// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! Fabric Manager's GPU partitions mapped onto a host's PCI functions.
//!
//! In shared NVSwitch mode (NVIDIA HGX integration guide, release 19.0,
//! §6.2) Fabric Manager names a GPU by physical ID, its module slot on the
//! baseboard, and the host names it by PCI address.  The firmware's slot
//! label on the function (`label` attribute, "GPU0, SMX7" on DGX systems)
//! bridges the two.  A GPU without a parsable label resolves to no
//! partition, so an unknown board yields nothing rather than a wrong
//! partition.
//!
//! The input is Fabric Manager's partition lists reduced to ids and
//! physical IDs, however a caller obtained them.
//!
//! ```no_run
//! use pcilibs_rs::{fabric, platform, Sysfs};
//!
//! # fn main() -> std::io::Result<()> {
//! let sysfs = Sysfs::default();
//! let topology = platform::discover_topology(&sysfs)?;
//! let gpus = fabric::gpu_modules(&sysfs, &topology.gpus)?;
//! # let partitions: Vec<fabric::Partition> = Vec::new();
//! # let unsupported: Vec<u32> = Vec::new();
//! for partition in fabric::resolve(&partitions, &unsupported, &gpus) {
//!     if partition.advertisable() {
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

/// The module slot of the GPU at `bdf`, `None` when the firmware labels
/// the function with no slot or not at all.  Other read errors propagate,
/// so an unreadable sysfs never looks like an unlabeled board; an address
/// that is not one is an error.
pub fn module_slot(sysfs: &Sysfs, bdf: &str) -> io::Result<Option<u8>> {
    let Some(device) = sysfs.device(bdf) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{bdf:?} is not a PCI address"),
        ));
    };
    let path = device.join("label");
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

/// A partition as Fabric Manager lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    pub id: u32,
    pub active: bool,
    /// Its GPUs by physical ID, in Fabric Manager's order.
    pub physical_ids: Vec<u32>,
}

/// One GPU of a partition, as seen from this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub physical_id: u32,
    /// The function in that module slot, `None` when this host has no
    /// labeled GPU there.
    pub bdf: Option<String>,
}

/// A partition resolved against this host's GPUs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPartition {
    pub id: u32,
    pub active: bool,
    /// Absent from Fabric Manager's unsupported list, where partitions land
    /// when their NVLinks failed.
    pub supported: bool,
    /// No other active partition names one of its GPUs; Fabric Manager
    /// never activates overlapping partitions, so an overlap is a wrong list.
    pub exclusive: bool,
    /// In Fabric Manager's order.
    pub members: Vec<Member>,
}

impl ResolvedPartition {
    /// Every member GPU is on this host.  A partition without members is
    /// never complete: there is nothing to hand out.
    pub fn complete(&self) -> bool {
        !self.members.is_empty() && self.members.iter().all(|member| member.bdf.is_some())
    }

    /// A unit a scheduler may hand out: active, supported, exclusive, and
    /// all here.
    pub fn advertisable(&self) -> bool {
        self.active && self.supported && self.exclusive && self.complete()
    }

    /// The PCI addresses of the members this host has, in member order.
    pub fn bdfs(&self) -> impl Iterator<Item = &str> {
        self.members
            .iter()
            .filter_map(|member| member.bdf.as_deref())
    }
}

/// Map partitions onto this host's GPUs; `unsupported` is the ids of Fabric
/// Manager's unsupported list.  A slot two GPUs claim identifies neither, a
/// slot no GPU claims leaves its partitions incomplete, a GPU two active
/// partitions claim makes neither exclusive.  Ordered by partition id, the
/// last definition of a repeated id standing.
pub fn resolve(
    partitions: &[Partition],
    unsupported: &[u32],
    gpus: &[GpuModule],
) -> Vec<ResolvedPartition> {
    let mut by_module: BTreeMap<u8, Option<&str>> = BTreeMap::new();
    for gpu in gpus {
        if let Some(module) = gpu.module {
            by_module
                .entry(module)
                .and_modify(|bdf| *bdf = None)
                .or_insert(Some(&gpu.bdf));
        }
    }
    let unsupported: BTreeSet<u32> = unsupported.iter().copied().collect();
    let mut resolved: BTreeMap<u32, ResolvedPartition> = partitions
        .iter()
        .map(|partition| {
            let members = partition
                .physical_ids
                .iter()
                .map(|&physical_id| Member {
                    physical_id,
                    bdf: u8::try_from(physical_id)
                        .ok()
                        .and_then(|module| by_module.get(&module).copied().flatten())
                        .map(str::to_owned),
                })
                .collect();
            (
                partition.id,
                ResolvedPartition {
                    id: partition.id,
                    active: partition.active,
                    supported: !unsupported.contains(&partition.id),
                    exclusive: true,
                    members,
                },
            )
        })
        .collect();
    let mut claims: BTreeMap<u32, usize> = BTreeMap::new();
    for member in resolved
        .values()
        .filter(|partition| partition.active)
        .flat_map(|partition| partition.members.iter())
    {
        *claims.entry(member.physical_id).or_default() += 1;
    }
    for partition in resolved.values_mut().filter(|partition| partition.active) {
        partition.exclusive = partition
            .members
            .iter()
            .all(|member| claims.get(&member.physical_id) == Some(&1));
    }
    resolved.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs;

    fn partition(id: u32, active: bool, physical_ids: &[u32]) -> Partition {
        Partition {
            id,
            active,
            physical_ids: physical_ids.to_vec(),
        }
    }

    /// This DGX NVL8's table as Fabric Manager lists it: 1 is the node, 2
    /// and 3 the halves, 4 to 7 the pairs (1,2) (3,4) (5,6) (7,8), 8 to 15
    /// the singles.  `active` names the active ones.
    fn rubin(active: &[u32]) -> Vec<Partition> {
        let mut table = vec![
            partition(1, false, &[1, 2, 3, 4, 5, 6, 7, 8]),
            partition(2, false, &[1, 2, 3, 4]),
            partition(3, false, &[5, 6, 7, 8]),
        ];
        for (id, pair) in (4..).zip([[1, 2], [3, 4], [5, 6], [7, 8]]) {
            table.push(partition(id, false, &pair));
        }
        for gpu in 1..=8 {
            table.push(partition(7 + gpu, false, &[gpu]));
        }
        for partition in &mut table {
            partition.active = active.contains(&partition.id);
        }
        table
    }

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

    /// The same box with SMX1 back at its slot.
    fn rubin_gpus_complete() -> Vec<GpuModule> {
        let mut gpus = rubin_gpus();
        gpus.push(GpuModule {
            bdf: "0000:f0:00.0".to_owned(),
            module: Some(1),
        });
        gpus
    }

    /// HGX H100 as the guide prints it (19.0, §6.2 and §6.3): 0-based ids,
    /// pairs are PCIe siblings, and nvidia-smi's module ids by bus.
    fn h100() -> Vec<Partition> {
        let mut table = vec![
            partition(0, false, &[8, 7, 6, 5, 4, 3, 2, 1]),
            partition(1, true, &[4, 3, 2, 1]),
            partition(2, false, &[8, 7, 6, 5]),
            partition(3, false, &[3, 1]),
            partition(4, false, &[4, 2]),
            partition(5, true, &[7, 5]),
            partition(6, false, &[8, 6]),
        ];
        for gpu in 1..=8 {
            table.push(partition(6 + gpu, matches!(gpu, 6 | 8), &[gpu]));
        }
        table
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

    fn advertisable(resolved: &[ResolvedPartition]) -> Vec<(u32, Vec<&str>)> {
        resolved
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
        // Every spelling of the address names the same function; a string
        // that is no address is an error, not an unlabeled GPU.
        assert_eq!(module_slot(&fake.sysfs, "1a:00.0").unwrap(), Some(7));
        assert_eq!(module_slot(&fake.sysfs, "0000:1A:00.0").unwrap(), Some(7));
        assert_eq!(
            module_slot(&fake.sysfs, "not-a-bdf").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );

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
    fn rubin_node_partition_is_incomplete_without_smx1() {
        let resolved = resolve(&rubin(&[1]), &[], &rubin_gpus());
        assert_eq!(resolved.len(), 15);
        let node = &resolved[0];
        assert!(node.active && node.supported && !node.complete());
        assert_eq!(node.members[0].bdf, None);
        assert_eq!(node.bdfs().count(), 7);
        // Nothing active is complete, nothing complete is active.
        assert!(advertisable(&resolved).is_empty());
        assert!(resolved[2].complete());
        assert!(!resolved[2].active);
    }

    #[test]
    fn eight_active_singles_are_eight_units() {
        // The node carved all the way down: every GPU its own unit, 8 to 15
        // in Fabric Manager's numbering.  With SMX1 gone from PCIe its
        // single is the one that drops out.
        let singles: Vec<u32> = (8..=15).collect();
        let table = rubin(&singles);
        let resolved = resolve(&table, &[], &rubin_gpus_complete());
        let units = advertisable(&resolved);
        assert_eq!(units.len(), 8);
        assert_eq!(units.iter().map(|(id, _)| *id).collect::<Vec<_>>(), singles);
        assert!(units.iter().all(|(_, bdfs)| bdfs.len() == 1));
        assert_eq!(units[0].1, ["0000:f0:00.0"]);
        assert_eq!(units[7].1, ["0000:62:00.0"]);

        assert_eq!(
            advertisable(&resolve(&table, &[], &rubin_gpus()))
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>(),
            (9..=15).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_mixed_carve_resolves_every_active_partition() {
        // One half and two pairs: three units of two sizes, all at once.
        let resolved = resolve(&rubin(&[2, 6, 7]), &[], &rubin_gpus_complete());
        assert_eq!(
            advertisable(&resolved),
            [
                (
                    2,
                    vec![
                        "0000:f0:00.0",
                        "0000:bc:00.0",
                        "0000:9a:00.0",
                        "0000:df:00.0"
                    ]
                ),
                (6, vec!["0000:73:00.0", "0000:3c:00.0"]),
                (7, vec!["0000:1a:00.0", "0000:62:00.0"]),
            ]
        );
    }

    #[test]
    fn overlapping_active_partitions_hand_out_no_gpu_twice() {
        // Fabric Manager refuses to activate a half and a pair inside it
        // together; a list that claims both is wrong about GPUs 1 and 2, so
        // neither unit exists.  A single in the other half is untouched.
        let resolved = resolve(&rubin(&[2, 4, 12]), &[], &rubin_gpus_complete());
        let by_id = |id: u32| resolved.iter().find(|p| p.id == id).unwrap();
        assert!(by_id(2).active && by_id(2).complete() && !by_id(2).exclusive);
        assert!(by_id(4).active && by_id(4).complete() && !by_id(4).exclusive);
        assert!(by_id(12).exclusive);
        // Inactive partitions claim no GPU.
        assert!(by_id(5).exclusive && !by_id(5).active);
        assert_eq!(advertisable(&resolved), [(12, vec!["0000:73:00.0"])]);
    }

    #[test]
    fn rubin_half_and_pair_resolve_in_fabric_manager_order() {
        let resolved = resolve(&rubin(&[3, 5]), &[], &rubin_gpus());
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
        let resolved = resolve(&h100(), &[], &h100_gpus());
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
        let resolved = resolve(&h100(), &[5], &h100_gpus());
        let five = resolved.iter().find(|p| p.id == 5).unwrap();
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
        let resolved = resolve(&h100(), &[], &gpus);
        let one = resolved.iter().find(|p| p.id == 1).unwrap();
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
    fn empty_and_degenerate_lists() {
        assert!(resolve(&[], &[], &h100_gpus()).is_empty());

        // No members: nothing to hand out, so never complete.  A physical id
        // beyond any slot resolves to no function.  A repeated id keeps the
        // last definition, as a map would.
        let table = [
            partition(7, true, &[]),
            partition(8, true, &[300]),
            partition(9, false, &[1]),
            partition(9, true, &[2]),
        ];
        let resolved = resolve(&table, &[], &h100_gpus());
        assert_eq!(resolved.iter().map(|p| p.id).collect::<Vec<_>>(), [7, 8, 9]);
        assert!(!resolved[0].complete());
        assert_eq!(resolved[1].members[0].bdf, None);
        assert_eq!(resolved[2].bdfs().collect::<Vec<_>>(), ["0000:b8:00.0"]);
        assert_eq!(advertisable(&resolved), [(9, vec!["0000:b8:00.0"])]);
    }
}
