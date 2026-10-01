// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

//! Shared by the build script and the Linux catalog loader.

use super::records::{self, Entry, MAX_BYTES, MAX_ENTRIES};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};

fn context(error: io::Error, path: &Path) -> io::Error {
    io::Error::new(error.kind(), format!("{}: {error}", path.display()))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Read one file or the immediate `.catalog` files of a directory.
/// Limits apply to the whole input, before deduplication.
pub(super) fn read(path: &Path) -> io::Result<String> {
    let metadata = fs::metadata(path).map_err(|e| context(e, path))?;
    if !metadata.is_dir() {
        return read_file(path, MAX_BYTES);
    }

    let mut paths = Vec::new();
    for entry in fs::read_dir(path).map_err(|e| context(e, path))? {
        let entry = entry.map_err(|e| context(e, path))?;
        let candidate = entry.path();
        if candidate.extension().is_none_or(|ext| ext != "catalog") {
            continue;
        }
        let metadata = fs::metadata(&candidate).map_err(|e| context(e, &candidate))?;
        if metadata.is_dir() {
            continue;
        }
        if paths.len() >= MAX_ENTRIES {
            return Err(context(
                invalid("catalog directory exceeds file limit"),
                path,
            ));
        }
        paths.push(candidate);
    }
    paths.sort();

    let mut remaining = MAX_BYTES;
    let mut sources = Vec::new();
    for path in paths {
        let text = read_file(&path, remaining)?;
        remaining -= text.len();
        sources.push((path, text));
    }
    merge(&sources).map_err(|e| context(e, path))
}

fn read_file(path: &Path, max_bytes: usize) -> io::Result<String> {
    if !fs::metadata(path).map_err(|e| context(e, path))?.is_file() {
        return Err(context(invalid("catalog is not a regular file"), path));
    }
    let file = File::open(path).map_err(|e| context(e, path))?;
    let mut text = String::new();
    file.take((max_bytes + 1) as u64)
        .read_to_string(&mut text)
        .map_err(|e| context(e, path))?;
    if text.len() > max_bytes {
        return Err(context(invalid("catalog exceeds byte limit"), path));
    }
    records::validate(&text).map_err(|e| context(invalid(e.to_string()), path))?;
    Ok(text)
}

fn merge(sources: &[(PathBuf, String)]) -> io::Result<String> {
    let mut entries = Vec::new();
    for (path, text) in sources {
        for (line, record) in records::lines(text) {
            if entries.len() >= MAX_ENTRIES {
                return Err(context(invalid("catalog exceeds entry limit"), path));
            }
            let entry = records::parse_entry(record).expect("validated catalog record");
            entries.push((entry, path, line, record));
        }
    }
    entries.sort_by_key(|(entry, ..)| entry.key());

    let mut accepted: Vec<(Entry<'_>, &Path, usize)> = Vec::new();
    let mut merged = String::new();
    for (entry, path, line, record) in entries {
        let mut duplicate = false;
        for &(prev, prev_path, prev_line) in &accepted {
            if entry.key() == prev.key() && entry.properties == prev.properties {
                duplicate = true;
                break;
            }
            if entry.overlaps(prev) {
                return Err(invalid(format!(
                    "{}:{line}: overlapping subsystem identities; previous record at {}:{prev_line}",
                    path.display(), prev_path.display(),
                )));
            }
        }
        if duplicate {
            continue;
        }
        merged.push_str(record);
        merged.push('\n');
        accepted.push((entry, path, line));
    }
    // Normalizing a missing final newline can add one byte per file.
    records::validate(&merged).map_err(|e| invalid(e.to_string()))?;
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Attachment, Catalog, PciIdentity};
    use rstest::rstest;

    #[test]
    fn directory_merges_by_identity_and_ignores_unrelated_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("a.catalog"),
            "10de ffff * * gpu GR100 coherent",
        )
        .unwrap();
        fs::write(
            dir.path().join("z.catalog"),
            "10de 0001 * * gpu GH100 pcie\n",
        )
        .unwrap();
        fs::write(dir.path().join("notes.txt"), "not a catalog").unwrap();
        fs::create_dir(dir.path().join("nested.catalog")).unwrap();
        fs::write(dir.path().join("nested.catalog/bad.catalog"), "invalid").unwrap();
        let text = read(dir.path()).unwrap();
        assert!(text.starts_with("10de 0001"));
        let catalog = Catalog::parse(&text).unwrap();
        assert_eq!(
            catalog
                .lookup(PciIdentity::new(0x10de, 1, 0x10de, 0))
                .unwrap()
                .attachment,
            Attachment::Pcie
        );
        assert_eq!(
            catalog
                .lookup(PciIdentity::new(0x10de, 0xffff, 0x10de, 0))
                .unwrap()
                .attachment,
            Attachment::Coherent
        );
    }

    #[test]
    fn identical_records_in_different_files_are_accepted() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("a.catalog"),
            "10de 0001 * 000a gpu GR100 sxm",
        )
        .unwrap();
        fs::write(
            dir.path().join("b.catalog"),
            "10de 0x0001 * 0x000A gpu GR100 sxm\n",
        )
        .unwrap();
        assert_eq!(
            read(dir.path()).unwrap(),
            "10de 0001 * 000a gpu GR100 sxm\n"
        );
    }

    #[rstest]
    #[case::attachment(
        "10de 0001 * 0001 gpu GR100 sxm",
        "10de 0001 * 0001 gpu GR100 coherent"
    )]
    #[case::wildcard("10de 0001 * * gpu GR100 sxm", "10de 0001 * 0001 gpu GR100 sxm")]
    #[case::profile("10de 0001 * 0001 gpu GR100 sxm", "10de 0001 * 0001 gpu GH100 sxm")]
    #[case::subsystem_axes(
        "1234 0001 * 0001 nic FutureNIC pcie\n1234 0001 * 0002 nic FutureNIC pcie",
        "1234 0001 0001 * nic FutureNIC pcie"
    )]
    fn conflicts_report_both_files(#[case] first: &str, #[case] second: &str) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.catalog"), first).unwrap();
        fs::write(dir.path().join("b.catalog"), second).unwrap();
        let error = read(dir.path()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("a.catalog:1"));
        assert!(error.to_string().contains("b.catalog:1"));
    }

    #[rstest]
    #[case::malformed("invalid")]
    #[case::invalid_profile("10de 0001 * * gpu bad! sxm")]
    #[case::invalid_kind("10de 3041 * * invalid GH100 coherent")]
    fn invalid_directory_member_is_an_error(#[case] text: &str) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("bad.catalog"), text).unwrap();
        let error = read(dir.path()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("bad.catalog"));
    }

    #[test]
    fn limits_apply_across_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.catalog"), " ".repeat(MAX_BYTES / 2)).unwrap();
        fs::write(dir.path().join("b.catalog"), " ".repeat(MAX_BYTES / 2 + 1)).unwrap();
        assert!(read(dir.path())
            .unwrap_err()
            .to_string()
            .contains("byte limit"));
        for (name, start, end) in [("a.catalog", 0, 512), ("b.catalog", 512, MAX_ENTRIES + 1)] {
            let text: String = (start..end)
                .map(|id| format!("10de {id:04x} * * gpu GR100 sxm\n"))
                .collect();
            fs::write(dir.path().join(name), text).unwrap();
        }
        assert!(read(dir.path())
            .unwrap_err()
            .to_string()
            .contains("entry limit"));
    }

    #[test]
    fn empty_directory_adds_no_records() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read(dir.path()).unwrap(), "");
    }
}
