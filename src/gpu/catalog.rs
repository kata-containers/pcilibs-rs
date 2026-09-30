// Copyright (c) 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA GPU identity extensions, independent of the library release.
//!
//! The first non-comment line is `pcilibs-nvidia-gpus 1 REVISION`. Records are
//! `DEVICE SUBSYSTEM_DEVICE CHIP ATTACHMENT`, sorted by numeric device/subsystem
//! ID. IDs are four hexadecimal digits (optional `0x`); subsystem `*` explicitly
//! covers every variant. Attachments are `pcie`, `sxm`, or `coherent`. Chip names
//! must name an existing [`super::CHIPS`] profile. Blank lines and whole-line
//! `#` comments are allowed. Wildcards cannot overlap exact records.
//!
//! External records extend the bundled mappings. Conflicting built-in facts are
//! rejected; identical records remain valid after a library update incorporates
//! them. Identities absent from both catalogs stay unknown.
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
        let mut lines = lines(text);
        let (line, header) = lines.next().ok_or(fail(0, "missing header"))?;
        let mut fields = header.split_ascii_whitespace();
        if fields.next() != Some("pcilibs-nvidia-gpus") || fields.next() != Some("1") {
            return Err(fail(line, "unsupported catalog format or version"));
        }
        let revision = fields.next().unwrap_or("");
        if revision.is_empty()
            || revision.len() > 128
            || !revision
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-@/".contains(&b))
            || fields.next().is_some()
        {
            return Err(fail(line, "invalid revision or extra header fields"));
        }
        let mut previous: Option<Entry> = None;
        let mut count = 0;
        for (line, record) in lines {
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
            for (_, record) in self::lines(BUILTIN).skip(1) {
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

    pub fn revision(self) -> &'a str {
        lines(self.text)
            .next()
            .and_then(|(_, header)| header.split_ascii_whitespace().nth(2))
            .expect("validated catalog header")
    }

    /// Device-only callers cannot distinguish coherent and non-coherent variants.
    pub fn may_be_coherent(self, device: u16) -> bool {
        lines(self.text).skip(1).any(|(_, record)| {
            let entry = parse_entry(record).expect("validated catalog record");
            entry.device == device && entry.properties.attachment == Attachment::Coherent
        }) || (self.extension && Catalog::builtin().may_be_coherent(device))
    }

    /// NVIDIA vendor and GPU class must be established by the caller.
    pub fn lookup(self, device: u16, subsystem_device: u16) -> Option<Properties> {
        for (_, record) in lines(self.text).skip(1) {
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
        assert_eq!(catalog.revision(), Catalog::builtin().revision());
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
            "# synthetic identities for this test\npcilibs-nvidia-gpus 1 test@2\n0x0001 * GH100 coherent\nffff 1234 GR100 coherent\n",
        ).unwrap();
        assert_eq!(catalog.revision(), "test@2");
        assert_eq!(catalog.lookup(1, 0).unwrap().chip.name, "GH100");
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
    #[case::hopper_coherent("pcilibs-nvidia-gpus 1 test\n0001 * GH100 coherent", false)]
    #[case::blackwell_coherent("pcilibs-nvidia-gpus 1 test\n0001 * GB100 coherent", false)]
    #[case::rubin_coherent("pcilibs-nvidia-gpus 1 test\n0001 * GR100 coherent", true)]
    #[case::blackwell_sxm("pcilibs-nvidia-gpus 1 test\n0001 * GB100 sxm", true)]
    #[case::hopper_pcie("pcilibs-nvidia-gpus 1 test\n0001 * GH100 pcie", true)]
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
    #[case::empty("", 0)]
    #[case::comments("# nothing\n\n", 0)]
    #[case::version("pcilibs-nvidia-gpus 2 test", 1)]
    #[case::format("wrong 1 test", 1)]
    #[case::missing_revision("pcilibs-nvidia-gpus 1", 1)]
    #[case::invalid_revision("pcilibs-nvidia-gpus 1 bad!", 1)]
    #[case::extra_header("pcilibs-nvidia-gpus 1 test extra", 1)]
    #[case::builtin_attachment_conflict("pcilibs-nvidia-gpus 1 test\n3041 221a GR100 pcie", 2)]
    #[case::builtin_wildcard_conflict("pcilibs-nvidia-gpus 1 test\n29bc * GB102 coherent", 2)]
    #[case::builtin_chip_conflict("pcilibs-nvidia-gpus 1 test\n3043 * GH100 coherent", 2)]
    #[case::missing_identity("pcilibs-nvidia-gpus 1 test\n0001", 2)]
    #[case::wide_id("pcilibs-nvidia-gpus 1 test\n10000 * GR100 sxm", 2)]
    #[case::bad_hex("pcilibs-nvidia-gpus 1 test\nzzzz * GR100 sxm", 2)]
    #[case::short_hex("pcilibs-nvidia-gpus 1 test\n001 * GR100 sxm", 2)]
    #[case::bad_subsystem("pcilibs-nvidia-gpus 1 test\n0001 xxxx GR100 sxm", 2)]
    #[case::missing_chip("pcilibs-nvidia-gpus 1 test\n0001 *", 2)]
    #[case::unknown_chip("pcilibs-nvidia-gpus 1 test\n0001 * FUTURE sxm", 2)]
    #[case::bad_attachment("pcilibs-nvidia-gpus 1 test\n0001 * GR100 unknown", 2)]
    #[case::missing_attachment("pcilibs-nvidia-gpus 1 test\n0001 * GR100", 2)]
    #[case::extra_field("pcilibs-nvidia-gpus 1 test\n0001 * GR100 sxm extra", 2)]
    #[case::duplicate("pcilibs-nvidia-gpus 1 test\n0001 * GR100 sxm\n0001 * GR100 sxm", 3)]
    #[case::duplicate_conflict(
        "pcilibs-nvidia-gpus 1 test\n0001 0001 GR100 sxm\n0001 0001 GR100 coherent",
        3
    )]
    #[case::unsorted("pcilibs-nvidia-gpus 1 test\n0002 * GR100 sxm\n0001 * GR100 sxm", 3)]
    #[case::wildcard_overlap(
        "pcilibs-nvidia-gpus 1 test\n0001 * GR100 sxm\n0001 0001 GR100 coherent",
        3
    )]
    #[case::conflicting_chip(
        "pcilibs-nvidia-gpus 1 test\n0001 0001 GR100 sxm\n0001 0002 GH100 sxm",
        3
    )]
    fn rejects_invalid_snapshots(#[case] text: &str, #[case] line: usize) {
        let error = Catalog::parse(text).unwrap_err();
        assert_eq!(error.line, line);
        assert!(!error.message.is_empty());
    }

    #[test]
    fn whitespace_comments_and_hex_case_are_unambiguous() {
        let catalog = Catalog::parse(
            "\r\n # comment\r\n pcilibs-nvidia-gpus\t1 test\r\n\n 00FF\t0xABCD GR100 sxm\r\n",
        )
        .unwrap();
        assert_eq!(
            catalog.lookup(0xff, 0xabcd).unwrap().attachment,
            Attachment::Sxm
        );
    }
    #[test]
    fn extensions_remain_valid_when_the_library_learns_the_same_identity() {
        let catalog =
            Catalog::parse("pcilibs-nvidia-gpus 1 test\n3041 221a GR100 coherent\n").unwrap();
        assert_eq!(
            catalog.lookup(0x3041, 0x221a).unwrap().attachment,
            Attachment::Coherent
        );
        assert_eq!(
            catalog.lookup(0x29bc, 0x1985).unwrap().attachment,
            Attachment::Sxm
        );
        let empty = Catalog::parse("pcilibs-nvidia-gpus 1 empty\n# no extensions\n").unwrap();
        assert_eq!(
            empty.lookup(0x3041, 0x221a).unwrap().attachment,
            Attachment::Coherent
        );
        let wildcard =
            Catalog::parse("pcilibs-nvidia-gpus 1 verified\n3041 * GR100 coherent\n").unwrap();
        assert_eq!(
            wildcard.lookup(0x3041, 0xffff).unwrap().attachment,
            Attachment::Coherent
        );
    }
}
