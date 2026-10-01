// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! Vendor-neutral PCI identity catalogs, independent of the library release.
//!
//! Records are `VENDOR DEVICE SUBSYSTEM_VENDOR SUBSYSTEM_DEVICE KIND PROFILE ATTACHMENT`.
//! IDs are hexadecimal. Only subsystem fields permit `*`. Records must be sorted
//! numerically by the four identity columns, with wildcards before exact values.
//! Blank lines and whole-line `#` comments are allowed; there is no header.
//!
//! Every `data/*.catalog` file is validated and bundled at build time. With `std`,
//! `CatalogFile::read` also accepts a directory. Extensions cannot contradict
//! bundled identities. Identical records survive updates to the built-in data.
//! Profile names describe hardware; vendor-specific code decides whether it can
//! operate that hardware. Parsing and lookup require neither std nor allocation.

use records::{lines, parse_entry};
mod records;
pub use records::{
    Attachment, DeviceKind, Entry, Error, PciIdentity, Properties, MAX_BYTES, MAX_ENTRIES,
};

#[cfg(feature = "std")]
mod file;
#[cfg(feature = "std")]
mod input;
#[cfg(feature = "std")]
pub use file::CatalogFile;
#[cfg(feature = "std")]
mod linux;
#[cfg(feature = "std")]
pub use linux::{discover, Device};

const BUILTIN: &str = include_str!(concat!(env!("OUT_DIR"), "/pci-devices.catalog"));

#[derive(Clone, Copy, Debug)]
pub struct Catalog<'a> {
    text: &'a str,
    extension: bool,
}

impl<'a> Catalog<'a> {
    pub const fn builtin() -> Catalog<'static> {
        Catalog {
            text: BUILTIN,
            extension: false,
        }
    }

    /// Validate a complete snapshot before any of its records can be used.
    pub fn parse(text: &'a str) -> Result<Self, Error> {
        records::validate(text)?;
        for (line, record) in lines(text) {
            let entry = parse_entry(record).expect("validated catalog record");
            for builtin in Catalog::builtin().entries() {
                if entry.overlaps(builtin) && entry.properties != builtin.properties {
                    return Err(Error {
                        line,
                        message: "entry conflicts with a built-in identity",
                    });
                }
            }
        }
        Ok(Self {
            text,
            extension: true,
        })
    }

    /// Iterate the records supplied to this snapshot. An extension's iterator
    /// does not repeat the built-in records that lookup falls back to.
    pub fn entries(self) -> impl Iterator<Item = Entry<'a>> {
        lines(self.text).map(|(_, record)| parse_entry(record).expect("validated catalog record"))
    }

    /// Conservative device-only query; subsystem variants may differ.
    pub fn may_be_coherent(self, vendor: u16, device: u16) -> bool {
        self.entries().any(|entry| {
            entry.vendor == vendor
                && entry.device == device
                && entry.properties.attachment == Attachment::Coherent
        }) || (self.extension && Catalog::builtin().may_be_coherent(vendor, device))
    }

    /// Vendor and both subsystem IDs are part of PCI identity.
    pub fn lookup(self, identity: PciIdentity) -> Option<Properties<'a>> {
        for entry in self.entries() {
            if (entry.vendor, entry.device) > (identity.vendor, identity.device) {
                break;
            }
            if entry.matches(identity) {
                return Some(entry.properties);
            }
        }
        if self.extension {
            Catalog::builtin().lookup(identity)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn id(vendor: u16, device: u16, subvendor: u16, subsystem: u16) -> PciIdentity {
        PciIdentity::new(vendor, device, subvendor, subsystem)
    }

    #[test]
    fn bundled_data_covers_gpus_nics_and_switches() {
        let catalog = Catalog::parse(BUILTIN).unwrap();
        assert_eq!(
            catalog
                .lookup(id(0x10de, 0x2321, 0x10de, 0x1839))
                .unwrap()
                .attachment,
            Attachment::Pcie
        );
        for (subsystem, attachment) in [
            (0x1985, Attachment::Sxm),
            (0x2045, Attachment::Coherent),
            (0x1997, Attachment::Pcie),
        ] {
            assert_eq!(
                catalog
                    .lookup(id(0x10de, 0x29bc, 0x10de, subsystem))
                    .unwrap()
                    .attachment,
                attachment
            );
        }
        assert!(catalog.lookup(id(0x10de, 0x29bc, 0x10de, 0xffff)).is_none());
        assert!(catalog.lookup(id(0x8086, 0x29bc, 0x10de, 0x2045)).is_none());
        assert_eq!(
            catalog.lookup(id(0x15b3, 0x1021, 0, 0)).unwrap().kind,
            DeviceKind::Nic
        );
        assert_eq!(
            catalog.lookup(id(0x15b3, 0xd2f4, 0, 0)).unwrap().profile,
            "Quantum-3"
        );
        assert_eq!(
            catalog.lookup(id(0x15b3, 0xd2f4, 0, 0)).unwrap().kind,
            DeviceKind::Switch
        );
        assert!(catalog.may_be_coherent(0x10de, 0x29bc));
        assert!(!catalog.may_be_coherent(0x15b3, 0x29bc));
        assert!(!catalog.may_be_coherent(0xffff, 0xffff));
    }

    #[test]
    fn vendors_and_subsystem_vendors_do_not_alias() {
        let catalog = Catalog::parse("1002 ffff 1002 0001 gpu FutureGPU pcie\n1002 ffff 1abc 0001 gpu OEMGPU pcie\n15b3 ffff * * nic FutureNIC pcie\n8086 ffff * * other FutureDevice unknown\n").unwrap();
        assert_eq!(
            catalog
                .lookup(id(0x1002, 0xffff, 0x1002, 1))
                .unwrap()
                .profile,
            "FutureGPU"
        );
        assert_eq!(
            catalog
                .lookup(id(0x1002, 0xffff, 0x1abc, 1))
                .unwrap()
                .profile,
            "OEMGPU"
        );
        assert!(catalog.lookup(id(0x1002, 0xffff, 0xffff, 1)).is_none());
        assert_eq!(
            catalog.lookup(id(0x15b3, 0xffff, 0, 0)).unwrap().kind,
            DeviceKind::Nic
        );
        assert_eq!(
            catalog.lookup(id(0x8086, 0xffff, 0, 0)).unwrap().attachment,
            Attachment::Unknown
        );
        assert!(catalog.lookup(id(0x10de, 0xffff, 0, 0)).is_none());
    }

    #[test]
    fn future_profiles_need_no_compiled_vendor_table() {
        let catalog = Catalog::parse("1234 abcd * * bridge FutureBridge pcie").unwrap();
        let properties = catalog.lookup(id(0x1234, 0xabcd, 0x1111, 0x2222)).unwrap();
        assert_eq!(properties.kind, DeviceKind::Bridge);
        assert_eq!(properties.profile, "FutureBridge");
        assert!(catalog.lookup(id(0x10de, 0x3041, 0x10de, 0x221a)).is_some());
    }

    #[test]
    fn identical_builtin_records_survive_updates() {
        let catalog = Catalog::parse("10de 3041 * 221a gpu GR100 coherent").unwrap();
        assert_eq!(
            catalog
                .lookup(id(0x10de, 0x3041, 0x10de, 0x221a))
                .unwrap()
                .profile,
            "GR100"
        );
        let wider = Catalog::parse("10de 3041 * * gpu GR100 coherent").unwrap();
        assert!(wider.lookup(id(0x10de, 0x3041, 1, 2)).is_some());
        assert!(Catalog::parse("10de 3041 * 221a gpu GR100 pcie").is_err());
        assert!(Catalog::parse("15b3 1021 * * switch ConnectX-7 pcie").is_err());
    }

    #[rstest]
    #[case::empty("")]
    #[case::comments("# no extensions\n\n")]
    fn empty_extensions_keep_builtins(#[case] text: &str) {
        let catalog = Catalog::parse(text).unwrap();
        assert!(catalog.lookup(id(0x10de, 0x3041, 0x10de, 0x221a)).is_some());
        assert!(catalog.may_be_coherent(0x10de, 0x3041));
        assert!(catalog.lookup(id(0xffff, 0xffff, 0, 0)).is_none());
    }

    #[test]
    fn whitespace_comments_and_hex_case_work() {
        let catalog =
            Catalog::parse("\r\n # comment\r\n\n 0xABCD\t00FF\t*\t0xABCD nic FutureNIC pcie\r\n")
                .unwrap();
        assert!(catalog.lookup(id(0xabcd, 0xff, 0, 0xabcd)).is_some());
    }

    #[rstest]
    #[case::vendor("* 0001 * * gpu GH100 pcie", 1)]
    #[case::device("10de zzzz * * gpu GH100 pcie", 1)]
    #[case::subvendor("10de 0001 xxxx * gpu GH100 pcie", 1)]
    #[case::subdevice("10de 0001 * xxxx gpu GH100 pcie", 1)]
    #[case::short("10de 001 * * gpu GH100 pcie", 1)]
    #[case::wide("10de 10000 * * gpu GH100 pcie", 1)]
    #[case::missing("10de", 1)]
    #[case::kind("10de 0001 * * invalid GH100 pcie", 1)]
    #[case::missing_profile("10de 0001 * * gpu", 1)]
    #[case::profile("10de 0001 * * gpu bad! pcie", 1)]
    #[case::missing_attachment("10de 0001 * * gpu GH100", 1)]
    #[case::attachment("10de 0001 * * gpu GH100 invalid", 1)]
    #[case::extra("10de 0001 * * gpu GH100 pcie extra", 1)]
    #[case::duplicate("10de 0001 * * gpu GH100 pcie\n10de 0001 * * gpu GH100 pcie", 2)]
    #[case::unsorted("15b3 0001 * * nic NIC pcie\n10de 0001 * * gpu GH100 pcie", 2)]
    #[case::overlap("10de 0001 * * gpu GH100 pcie\n10de 0001 10de 0001 gpu GH100 pcie", 2)]
    #[case::nonadjacent_overlap("10de 0001 * 0001 gpu GH100 pcie\n10de 0001 * 0002 gpu GH100 pcie\n10de 0001 10de 0001 gpu GH100 pcie", 3)]
    #[case::comments("# comment\n\ninvalid", 3)]
    fn rejects_invalid_records(#[case] text: &str, #[case] line: usize) {
        let error = Catalog::parse(text).unwrap_err();
        assert_eq!(error.line, line);
        assert!(!error.message.is_empty());
    }
}
