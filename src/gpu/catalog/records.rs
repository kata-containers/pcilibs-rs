// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

use crate::gpu::{Attachment, Chip, CHIPS};
use core::fmt;

pub const MAX_BYTES: usize = 64 * 1024;
pub const MAX_ENTRIES: usize = 1024;

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

#[derive(Clone, Copy)]
pub(super) struct Entry {
    pub device: u16,
    pub subsystem: Option<u16>,
    pub properties: Properties,
}

pub(super) fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines().enumerate().filter_map(|(line, text)| {
        let text = text.trim_ascii();
        (!text.is_empty() && !text.starts_with('#')).then_some((line + 1, text))
    })
}

pub(super) fn parse_entry(text: &str) -> Result<Entry, &'static str> {
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

/// Validate one sorted catalog independently of the bundled identities.
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
        if let Some(prev) = previous {
            check_next(prev, entry).map_err(|message| fail(line, message))?;
        }
        if let Some(chip) = crate::gpu::chip_for_range(entry.device) {
            if chip.name != entry.properties.chip.name {
                return Err(fail(line, "entry conflicts with a built-in chip range"));
            }
        }
        previous = Some(entry);
    }
    Ok(())
}

pub(super) fn check_next(prev: Entry, entry: Entry) -> Result<(), &'static str> {
    if (entry.device, entry.subsystem) <= (prev.device, prev.subsystem) {
        return Err("duplicate or unsorted identity");
    }
    if entry.device == prev.device {
        if prev.subsystem.is_none() || entry.subsystem.is_none() {
            return Err("wildcard overlaps a subsystem identity");
        }
        if entry.properties.chip.name != prev.properties.chip.name {
            return Err("conflicting chip profiles for one device ID");
        }
    }
    Ok(())
}
