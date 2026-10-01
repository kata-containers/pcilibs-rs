// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

use super::{input, Catalog};
use std::{io, path::Path};

/// Owns a validated snapshot from a file or directory; callers choose when to reload it.
#[derive(Debug)]
pub struct CatalogFile {
    text: String,
}

impl CatalogFile {
    /// Read a file, or merge the immediate `.catalog` files in a directory.
    /// Identical records across files are accepted; conflicts are errors.
    /// Explicit loading prevents a missing or invalid update from becoming a fallback.
    pub fn read(path: &Path) -> io::Result<Self> {
        let text = input::read(path)?;
        Catalog::parse(&text).map_err(|error| {
            crate::context(
                io::Error::new(io::ErrorKind::InvalidData, error),
                path.display(),
            )
        })?;
        Ok(Self { text })
    }

    pub fn catalog(&self) -> Catalog<'_> {
        Catalog {
            text: &self.text,
            extension: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{PciIdentity, MAX_BYTES};
    use rstest::{fixture, rstest};
    use tempfile::TempDir;

    #[fixture]
    fn directory() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[rstest]
    fn loaded_extension_is_a_snapshot(directory: TempDir) {
        let path = directory.path().join("gpus.catalog");
        std::fs::write(&path, "10de ffff * * gpu GR100 coherent\n").unwrap();
        let first = CatalogFile::read(&path).unwrap();
        std::fs::write(&path, "10de ffff * * gpu GH100 sxm\n").unwrap();
        let second = CatalogFile::read(&path).unwrap();
        assert_eq!(
            first
                .catalog()
                .lookup(PciIdentity::new(0x10de, 0xffff, 0x10de, 0))
                .unwrap()
                .profile,
            "GR100"
        );
        assert_eq!(
            second
                .catalog()
                .lookup(PciIdentity::new(0x10de, 0xffff, 0x10de, 0))
                .unwrap()
                .profile,
            "GH100"
        );
    }

    #[rstest]
    fn directory_reload_picks_up_added_and_removed_files(directory: TempDir) {
        let first_path = directory.path().join("first.catalog");
        std::fs::write(&first_path, "10de fffe * * gpu GR100 coherent\n").unwrap();
        let first = CatalogFile::read(directory.path()).unwrap();
        std::fs::write(
            directory.path().join("second.catalog"),
            "10de ffff * * gpu GH100 sxm\n",
        )
        .unwrap();
        let second = CatalogFile::read(directory.path()).unwrap();
        std::fs::remove_file(first_path).unwrap();
        let third = CatalogFile::read(directory.path()).unwrap();
        assert!(first
            .catalog()
            .lookup(PciIdentity::new(0x10de, 0xffff, 0x10de, 0))
            .is_none());
        assert!(second
            .catalog()
            .lookup(PciIdentity::new(0x10de, 0xfffe, 0x10de, 0))
            .is_some());
        assert!(second
            .catalog()
            .lookup(PciIdentity::new(0x10de, 0xffff, 0x10de, 0))
            .is_some());
        assert!(third
            .catalog()
            .lookup(PciIdentity::new(0x10de, 0xfffe, 0x10de, 0))
            .is_none());
        assert!(third
            .catalog()
            .lookup(PciIdentity::new(0x10de, 0xffff, 0x10de, 0))
            .is_some());
        assert!(third
            .catalog()
            .lookup(PciIdentity::new(0x10de, 0x3041, 0x10de, 0x221a))
            .is_some());
    }

    #[rstest]
    fn directory_cannot_override_a_builtin_identity(directory: TempDir) {
        std::fs::write(
            directory.path().join("bad.catalog"),
            "10de 3041 * 221a gpu GR100 pcie\n",
        )
        .unwrap();
        let error = CatalogFile::read(directory.path()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("built-in identity"));
    }

    #[rstest]
    fn missing_file_never_uses_the_builtin_catalog(directory: TempDir) {
        assert_eq!(
            CatalogFile::read(&directory.path().join("missing"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
    }

    #[rstest]
    #[case::malformed(b"bad catalog")]
    #[case::invalid_utf8(b"\xff")]
    fn invalid_files_are_rejected(directory: TempDir, #[case] bytes: &[u8]) {
        let path = directory.path().join("gpus.catalog");
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(
            CatalogFile::read(&path).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[rstest]
    fn size_limits_apply_before_a_snapshot_is_accepted(directory: TempDir) {
        let path = directory.path().join("gpus.catalog");
        let oversized = " ".repeat(MAX_BYTES + 1);
        std::fs::write(&path, &oversized).unwrap();
        let error = CatalogFile::read(&path).unwrap_err();
        assert!(error.to_string().contains("byte limit"));
        let mut text = String::new();
        for device in 0..=super::super::MAX_ENTRIES {
            text.push_str(&format!("10de {device:04x} * * gpu GR100 sxm\n"));
        }
        assert!(Catalog::parse(&text)
            .unwrap_err()
            .message
            .contains("entry limit"));
    }
}
