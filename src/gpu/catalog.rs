// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA GPU identity extensions, independent of the library release.
//!
//! Each record is `DEVICE SUBSYSTEM_DEVICE CHIP ATTACHMENT`, sorted by numeric
//! device/subsystem ID. IDs are four hexadecimal digits (optional `0x`);
//! subsystem `*` explicitly covers every variant. Attachments are `pcie`, `sxm`, or `coherent`. Chip names
//! must name an existing [`super::CHIPS`] profile. Blank lines and whole-line
//! `#` comments are allowed. Wildcards cannot overlap exact records.
//!
//! External records extend the bundled mappings. Conflicting built-in facts are
//! rejected; identical records remain valid after a library update incorporates
//! them. Identities absent from both catalogs stay unknown.
//! Empty or comment-only extensions add no mappings.
//! Parsing and lookup borrow the input and require neither `std` nor allocation.

use core::fmt;

use super::{Attachment, Chip, CHIPS};

#[cfg(feature = "std")]
mod file;
#[cfg(feature = "std")]
pub use file::CatalogFile;

pub const MAX_BYTES: usize = 64 * 1024;
pub const MAX_ENTRIES: usize = 1024;
const BUILTIN: &str = include_str!("../../data/nvidia-gpus.catalog");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Error {
    /// One-based line number; zero denotes a whole-file error.
    pub line: usize,
    pub message: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU catalog line {}: {}", self.line, self.message)
    }
}

impl core::error::Error for Error {}

#[derive(Clone, Copy)]
pub struct Properties {
    pub chip: &'static Chip,
    pub attachment: Attachment,
}

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
        if text.len() > MAX_BYTES {
            return Err(fail(0, "catalog exceeds byte limit"));
        }
        let mut previous: Option<Entry> = None;
        let mut count = 0;
        for (line, record) in lines(text) {
            count += 1;
            if count > MAX_ENTRIES {
                return Err(fail(line, "catalog exceeds entry limit"));
            }
            let entry = parse_entry(record).map_err(|message| fail(line, message))?;
            if let Some(prev) = previous {
                if (entry.device, entry.subsystem) <= (prev.device, prev.subsystem) {
                    return Err(fail(line, "duplicate or unsorted identity"));
                }
                if entry.device == prev.device {
                    if prev.subsystem.is_none() || entry.subsystem.is_none() {
                        return Err(fail(line, "wildcard overlaps a subsystem identity"));
                    }
                    if entry.properties.chip.name != prev.properties.chip.name {
                        return Err(fail(line, "conflicting chip profiles for one device ID"));
                    }
                }
            }
            if let Some(chip) = super::chip_for(entry.device) {
                if chip.name != entry.properties.chip.name {
                    return Err(fail(line, "entry conflicts with a built-in chip range"));
                }
            }
            for (_, record) in lines(BUILTIN) {
                let builtin = parse_entry(record).expect("bundled catalog record");
                if entry.device == builtin.device
                    && (entry.subsystem.is_none()
                        || builtin.subsystem.is_none()
                        || entry.subsystem == builtin.subsystem)
                    && (entry.properties.chip.name != builtin.properties.chip.name
                        || entry.properties.attachment != builtin.properties.attachment)
                {
                    return Err(fail(line, "entry conflicts with a built-in identity"));
                }
            }
            previous = Some(entry);
        }
        Ok(Self {
            text,
            extension: true,
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

#[derive(Clone, Copy)]
struct Entry {
    device: u16,
    subsystem: Option<u16>,
    properties: Properties,
}

fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines().enumerate().filter_map(|(line, text)| {
        let text = text.trim_ascii();
        (!text.is_empty() && !text.starts_with('#')).then_some((line + 1, text))
    })
}

fn parse_entry(text: &str) -> Result<Entry, &'static str> {
    let mut fields = text.split_ascii_whitespace();
    let device = hex_id(fields.next())?;
    let subsystem = match fields.next() {
        Some("*") => None,
        value => Some(hex_id(value)?),
    };
    let name = fields.next().ok_or("missing chip profile")?;
    let chip = CHIPS
        .iter()
        .find(|chip| chip.name == name)
        .ok_or("unknown chip profile")?;
    let attachment = match fields.next() {
        Some("pcie") => Attachment::Pcie,
        Some("sxm") => Attachment::Sxm,
        Some("coherent") => Attachment::Coherent,
        _ => return Err("invalid attachment"),
    };
    if fields.next().is_some() {
        return Err("extra record fields");
    }
    Ok(Entry {
        device,
        subsystem,
        properties: Properties { chip, attachment },
    })
}

fn hex_id(value: Option<&str>) -> Result<u16, &'static str> {
    let value = value.ok_or("missing PCI identity")?;
    let value = value.strip_prefix("0x").unwrap_or(value);
    if value.len() != 4 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("PCI identity must contain four hexadecimal digits");
    }
    u16::from_str_radix(value, 16).map_err(|_| "invalid PCI identity")
}

#[cfg(test)]
mod tests {
    use super::*;
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
