// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

use pcilibs_rs::{
    gpu::catalog::{Catalog, CatalogFile},
    platform, Sysfs,
};

fn main() -> std::io::Result<()> {
    // Accept one file or a directory of .catalog files, e.g. /etc/pcilibs/gpus.d.
    let extension = std::env::args_os()
        .nth(1)
        .map(|path| CatalogFile::read(std::path::Path::new(&path)))
        .transpose()?;
    let catalog = extension
        .as_ref()
        .map_or(Catalog::builtin(), CatalogFile::catalog);
    let detected = platform::discover_with_catalog(&Sysfs::default(), catalog)?;
    println!("{:?}", detected.platform);
    Ok(())
}
