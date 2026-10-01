// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA GPU identity extensions, independent of the library release.
//!
//! Each record is `DEVICE SUBSYSTEM_DEVICE CHIP ATTACHMENT`, sorted by numeric
//! device/subsystem ID. IDs are four hexadecimal digits (optional `0x`);
//! subsystem `*` explicitly covers every variant. Attachments are `pcie`, `sxm`,
//! or `coherent`. Chip names must name an existing [`super::CHIPS`] profile. Blank lines and whole-line
//! `#` comments are allowed. Wildcards cannot overlap exact records.
//!
//! External records extend the bundled mappings. Conflicting built-in facts are
//! rejected; identical records remain valid after a library update incorporates
//! them. Identities absent from both catalogs stay unknown.
//! Every `data/*.catalog` file is validated and bundled at build time.
//! With `std`, `CatalogFile::read` also accepts a directory of `.catalog` files.
//! Empty or comment-only extensions add no mappings.
//! Parsing and lookup borrow the input and require neither `std` nor allocation.

use super::{Attachment, Chip};
use records::{lines, parse_entry};

mod records;
pub use records::{Error, Properties, MAX_BYTES, MAX_ENTRIES};

#[cfg(feature = "std")]
mod input;

#[cfg(feature = "std")]
mod file;
#[cfg(feature = "std")]
pub use file::CatalogFile;

const BUILTIN: &str = include_str!(concat!(env!("OUT_DIR"), "/nvidia-gpus.catalog"));

impl Properties {
    /// Coherent attachment and firmware CC enablement are separate capabilities.
    pub fn in_band_cc_supported(self) -> bool {
        match self.attachment {
            Attachment::Pcie | Attachment::Sxm => true,
            Attachment::Coherent => self.chip.c2c_cc_supported,
            Attachment::Unknown => false,
        }
    }
}

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

    /// Validate the entire snapshot before any of its records can be used.
    pub fn parse(text: &'a str) -> Result<Self, Error> {
        let fail = |line, message| Error { line, message };
        records::validate(text)?;
        for (line, record) in lines(text) {
            let entry = parse_entry(record).expect("validated catalog record");
            for (_, record) in lines(BUILTIN) {
                let builtin = parse_entry(record).expect("bundled catalog record");
                if entry.device == builtin.device
                    && (entry.properties.chip.name != builtin.properties.chip.name
                        || ((entry.subsystem.is_none()
                            || builtin.subsystem.is_none()
                            || entry.subsystem == builtin.subsystem)
                            && entry.properties.attachment != builtin.properties.attachment))
                {
                    return Err(fail(line, "entry conflicts with a built-in identity"));
                }
            }
        }
        Ok(Self {
            text,
            extension: true,
        })
    }

    pub(super) fn lookup_chip(self, device: u16) -> Option<&'static Chip> {
        lines(self.text).find_map(|(_, record)| {
            let entry = parse_entry(record).expect("validated catalog record");
            (entry.device == device).then_some(entry.properties.chip)
        })
    }

    /// Device-only callers cannot distinguish coherent and non-coherent variants.
    pub fn may_be_coherent(self, device: u16) -> bool {
        lines(self.text).any(|(_, record)| {
            let entry = parse_entry(record).expect("validated catalog record");
            entry.device == device && entry.properties.attachment == Attachment::Coherent
        }) || (self.extension && Catalog::builtin().may_be_coherent(device))
    }

    /// NVIDIA vendor and GPU class must be established by the caller.
    pub fn lookup(self, device: u16, subsystem_device: u16) -> Option<Properties> {
        for (_, record) in lines(self.text) {
            let entry = parse_entry(record).expect("validated catalog record");
            if entry.device > device {
                break;
            }
            if entry.device == device && entry.subsystem.is_none_or(|id| id == subsystem_device) {
                return Some(entry.properties);
            }
        }
        if self.extension {
            Catalog::builtin().lookup(device, subsystem_device)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::CHIPS;
    use rstest::rstest;

    #[test]
    fn bundled_snapshot_is_valid_and_preserves_aliased_variants() {
        let catalog = Catalog::parse(BUILTIN).unwrap();
        assert_eq!(
            Catalog::builtin()
                .lookup(0x2321, 0x1839)
                .unwrap()
                .attachment,
            Attachment::Pcie
        );
        assert!(Catalog::parse("2321 1839 GH100 coherent").is_err());
        assert_eq!(
            catalog.lookup(0x29bc, 0x1985).unwrap().attachment,
            Attachment::Sxm
        );
        assert_eq!(
            catalog.lookup(0x29bc, 0x2045).unwrap().attachment,
            Attachment::Coherent
        );
        assert!(catalog.may_be_coherent(0x29bc));
        assert!(!catalog.may_be_coherent(0xffff));
        assert!(catalog.may_be_coherent(0x31a1));
        assert!(catalog.may_be_coherent(0x31c3));
        assert!(!Properties {
            chip: &CHIPS[0],
            attachment: Attachment::Unknown,
        }
        .in_band_cc_supported());
        assert!(catalog.lookup(0x29bc, 0xffff).is_none());
    }

    #[test]
    fn new_ids_extend_builtin_profiles_without_compiled_ranges() {
        let catalog = Catalog::parse(
            "# synthetic identities for this test\n0x0001 * GH100 coherent\nffff 1234 GR100 coherent\n",
        ).unwrap();
        assert_eq!(catalog.lookup(1, 0).unwrap().chip.name, "GH100");
        assert_eq!(catalog.lookup_chip(1).unwrap().name, "GH100");
        assert!(catalog.may_be_coherent(1));
        assert!(!catalog.lookup(1, 0).unwrap().in_band_cc_supported());
        assert!(catalog
            .lookup(0xffff, 0x1234)
            .unwrap()
            .in_band_cc_supported());
        assert!(catalog.lookup(0xffff, 0x1235).is_none());
        assert_eq!(
            catalog.lookup(0x3041, 0x221a).unwrap().attachment,
            Attachment::Coherent
        );
        assert!(catalog.may_be_coherent(0x3041));
        assert!(catalog.lookup(0, 0).is_none());
    }

    #[rstest]
    #[case::hopper_coherent("0001 * GH100 coherent", false)]
    #[case::blackwell_coherent("0001 * GB100 coherent", false)]
    #[case::rubin_coherent("0001 * GR100 coherent", true)]
    #[case::blackwell_sxm("0001 * GB100 sxm", true)]
    #[case::hopper_pcie("0001 * GH100 pcie", true)]
    fn attachment_does_not_imply_cc_capability(#[case] text: &str, #[case] supported: bool) {
        assert_eq!(
            Catalog::parse(text)
                .unwrap()
                .lookup(1, 0)
                .unwrap()
                .in_band_cc_supported(),
            supported
        );
    }

    #[rstest]
    #[case::builtin_attachment_conflict("3041 221a GR100 pcie", 1)]
    #[case::builtin_wildcard_conflict("29bc * GB102 coherent", 1)]
    #[case::builtin_chip_conflict("3043 * GH100 coherent", 1)]
    #[case::missing_identity("0001", 1)]
    #[case::wide_id("10000 * GR100 sxm", 1)]
    #[case::bad_hex("zzzz * GR100 sxm", 1)]
    #[case::short_hex("001 * GR100 sxm", 1)]
    #[case::bad_subsystem("0001 xxxx GR100 sxm", 1)]
    #[case::missing_chip("0001 *", 1)]
    #[case::unknown_chip("0001 * FUTURE sxm", 1)]
    #[case::bad_attachment("0001 * GR100 unknown", 1)]
    #[case::missing_attachment("0001 * GR100", 1)]
    #[case::extra_field("0001 * GR100 sxm extra", 1)]
    #[case::duplicate("0001 * GR100 sxm\n0001 * GR100 sxm", 2)]
    #[case::duplicate_conflict("0001 0001 GR100 sxm\n0001 0001 GR100 coherent", 2)]
    #[case::unsorted("0002 * GR100 sxm\n0001 * GR100 sxm", 2)]
    #[case::wildcard_overlap("0001 * GR100 sxm\n0001 0001 GR100 coherent", 2)]
    #[case::conflicting_chip("0001 0001 GR100 sxm\n0001 0002 GH100 sxm", 2)]
    #[case::comments_before_error("# comment\n\ninvalid", 3)]
    fn rejects_invalid_snapshots(#[case] text: &str, #[case] line: usize) {
        let error = Catalog::parse(text).unwrap_err();
        assert_eq!(error.line, line);
        assert!(!error.message.is_empty());
    }

    #[rstest]
    #[case::empty("")]
    #[case::comments("# no extensions\n\n")]
    fn empty_extensions_preserve_builtin_mappings(#[case] text: &str) {
        let catalog = Catalog::parse(text).unwrap();
        assert_eq!(
            catalog.lookup(0x2321, 0x1839).unwrap().attachment,
            Attachment::Pcie
        );
        assert!(catalog.may_be_coherent(0x3041));
        assert!(catalog.lookup(0xffff, 0).is_none());
    }

    #[test]
    fn whitespace_comments_and_hex_case_are_unambiguous() {
        let catalog = Catalog::parse("\r\n # comment\r\n\n 00FF\t0xABCD GR100 sxm\r\n").unwrap();
        assert_eq!(
            catalog.lookup(0xff, 0xabcd).unwrap().attachment,
            Attachment::Sxm
        );
    }
    #[test]
    fn extensions_remain_valid_when_the_library_learns_the_same_identity() {
        let catalog = Catalog::parse("3041 221a GR100 coherent\n").unwrap();
        assert_eq!(
            catalog.lookup(0x3041, 0x221a).unwrap().attachment,
            Attachment::Coherent
        );
        assert_eq!(
            catalog.lookup(0x29bc, 0x1985).unwrap().attachment,
            Attachment::Sxm
        );
        let empty = Catalog::parse("# no extensions\n").unwrap();
        assert_eq!(
            empty.lookup(0x3041, 0x221a).unwrap().attachment,
            Attachment::Coherent
        );
        let wildcard = Catalog::parse("3041 * GR100 coherent\n").unwrap();
        assert_eq!(
            wildcard.lookup(0x3041, 0xffff).unwrap().attachment,
            Attachment::Coherent
        );
    }
}
