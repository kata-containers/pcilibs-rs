// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

use core::fmt;

pub const MAX_BYTES: usize = 64 * 1024;
pub const MAX_ENTRIES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Error {
    /// One-based line number; zero denotes a whole-input error.
    pub line: usize,
    pub message: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PCI catalog line {}: {}", self.line, self.message)
    }
}
impl core::error::Error for Error {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Gpu,
    Nic,
    Switch,
    Bridge,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attachment {
    Pcie,
    Sxm,
    Coherent,
    Unknown,
}

/// Descriptive metadata. A profile name does not authorize register access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Properties<'a> {
    pub kind: DeviceKind,
    pub profile: &'a str,
    pub attachment: Attachment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciIdentity {
    pub vendor: u16,
    pub device: u16,
    pub subsystem_vendor: u16,
    pub subsystem_device: u16,
}

impl PciIdentity {
    pub const fn new(
        vendor: u16,
        device: u16,
        subsystem_vendor: u16,
        subsystem_device: u16,
    ) -> Self {
        Self {
            vendor,
            device,
            subsystem_vendor,
            subsystem_device,
        }
    }
}

/// A catalog identity pattern. Only subsystem fields permit wildcards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry<'a> {
    pub vendor: u16,
    pub device: u16,
    pub subsystem_vendor: Option<u16>,
    pub subsystem_device: Option<u16>,
    pub properties: Properties<'a>,
}

impl Entry<'_> {
    pub(super) fn key(self) -> (u16, u16, Option<u16>, Option<u16>) {
        (
            self.vendor,
            self.device,
            self.subsystem_vendor,
            self.subsystem_device,
        )
    }

    pub(super) fn matches(self, identity: PciIdentity) -> bool {
        self.vendor == identity.vendor
            && self.device == identity.device
            && self
                .subsystem_vendor
                .is_none_or(|id| id == identity.subsystem_vendor)
            && self
                .subsystem_device
                .is_none_or(|id| id == identity.subsystem_device)
    }

    pub(super) fn overlaps(self, other: Self) -> bool {
        self.vendor == other.vendor
            && self.device == other.device
            && (self.subsystem_vendor.is_none()
                || other.subsystem_vendor.is_none()
                || self.subsystem_vendor == other.subsystem_vendor)
            && (self.subsystem_device.is_none()
                || other.subsystem_device.is_none()
                || self.subsystem_device == other.subsystem_device)
    }
}

pub(super) fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines().enumerate().filter_map(|(line, text)| {
        let text = text.trim_ascii();
        (!text.is_empty() && !text.starts_with('#')).then_some((line + 1, text))
    })
}

pub(super) fn parse_entry(text: &str) -> Result<Entry<'_>, &'static str> {
    let mut fields = text.split_ascii_whitespace();
    let vendor = hex_id(fields.next())?;
    let device = hex_id(fields.next())?;
    let subsystem_vendor = subsystem_id(fields.next())?;
    let subsystem_device = subsystem_id(fields.next())?;
    let kind = match fields.next() {
        Some("gpu") => DeviceKind::Gpu,
        Some("nic") => DeviceKind::Nic,
        Some("switch") => DeviceKind::Switch,
        Some("bridge") => DeviceKind::Bridge,
        Some("other") => DeviceKind::Other,
        _ => return Err("invalid device kind"),
    };
    let profile = fields.next().ok_or("missing device profile")?;
    if profile.len() > 64
        || !profile
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err("invalid device profile");
    }
    let attachment = match fields.next() {
        Some("pcie") => Attachment::Pcie,
        Some("sxm") => Attachment::Sxm,
        Some("coherent") => Attachment::Coherent,
        Some("unknown") => Attachment::Unknown,
        _ => return Err("invalid attachment"),
    };
    if fields.next().is_some() {
        return Err("extra record fields");
    }
    Ok(Entry {
        vendor,
        device,
        subsystem_vendor,
        subsystem_device,
        properties: Properties {
            kind,
            profile,
            attachment,
        },
    })
}

fn subsystem_id(value: Option<&str>) -> Result<Option<u16>, &'static str> {
    match value {
        Some("*") => Ok(None),
        value => hex_id(value).map(Some),
    }
}

fn hex_id(value: Option<&str>) -> Result<u16, &'static str> {
    let value = value.ok_or("missing PCI identity")?;
    let value = value.strip_prefix("0x").unwrap_or(value);
    if value.len() != 4 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("PCI identity must contain four hexadecimal digits");
    }
    u16::from_str_radix(value, 16).map_err(|_| "invalid PCI identity")
}

/// Validate sorted records without imposing vendor-specific device support.
pub(super) fn validate(text: &str) -> Result<(), Error> {
    let fail = |line, message| Error { line, message };
    if text.len() > MAX_BYTES {
        return Err(fail(0, "catalog exceeds byte limit"));
    }
    let mut previous = None;
    for (count, (line, record)) in lines(text).enumerate() {
        if count >= MAX_ENTRIES {
            return Err(fail(line, "catalog exceeds entry limit"));
        }
        let entry = parse_entry(record).map_err(|message| fail(line, message))?;
        if previous.is_some_and(|key| entry.key() <= key) {
            return Err(fail(line, "duplicate or unsorted identity"));
        }
        // Wildcards on two subsystem axes can overlap nonadjacent records.
        for (_, record) in lines(text).take(count) {
            let prev = parse_entry(record).expect("previous record was validated");
            if entry.overlaps(prev) {
                return Err(fail(line, "overlapping subsystem identities"));
            }
        }
        previous = Some(entry.key());
    }
    Ok(())
}
