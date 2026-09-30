// Copyright (c) 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

use super::{Catalog, MAX_BYTES};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

/// Owns one validated snapshot; callers choose when to replace it.
#[derive(Debug)]
pub struct CatalogFile {
    text: String,
}

impl CatalogFile {
    /// Explicit loading prevents a missing or invalid update from becoming a fallback.
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut text = String::new();
        File::open(path)
            .and_then(|file| file.take((MAX_BYTES + 1) as u64).read_to_string(&mut text))
            .map_err(|error| crate::context(error, path.display()))?;
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
    use rstest::{fixture, rstest};
    use tempfile::TempDir;

    #[fixture]
    fn directory() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[rstest]
    fn loaded_extension_is_a_snapshot(directory: TempDir) {
        let path = directory.path().join("gpus.catalog");
        std::fs::write(
            &path,
            "pcilibs-nvidia-gpus 1 first\nffff * GR100 coherent\n",
        )
        .unwrap();
        let first = CatalogFile::read(&path).unwrap();
        std::fs::write(&path, "pcilibs-nvidia-gpus 1 second\nffff * GH100 sxm\n").unwrap();
        let second = CatalogFile::read(&path).unwrap();
        assert_eq!(first.catalog().revision(), "first");
        assert_eq!(second.catalog().revision(), "second");
        assert_eq!(
            first.catalog().lookup(0xffff, 0).unwrap().chip.name,
            "GR100"
        );
        assert_eq!(
            second.catalog().lookup(0xffff, 0).unwrap().chip.name,
            "GH100"
        );
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
        let mut text = String::from("pcilibs-nvidia-gpus 1 test\n");
        for device in 0..=super::super::MAX_ENTRIES {
            text.push_str(&format!("{device:04x} * GR100 sxm\n"));
        }
        assert!(Catalog::parse(&text)
            .unwrap_err()
            .message
            .contains("entry limit"));
    }
}
