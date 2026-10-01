// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

use pcilibs_rs::{
    catalog::{self, Catalog, CatalogFile},
    platform, Sysfs,
};

fn main() -> std::io::Result<()> {
    // Accept one file or a directory of .catalog files, e.g. /etc/pcilibs/devices.d.
    let extension = std::env::args_os()
        .nth(1)
        .map(|path| CatalogFile::read(std::path::Path::new(&path)))
        .transpose()?;
    let catalog = extension
        .as_ref()
        .map_or(Catalog::builtin(), CatalogFile::catalog);
    let sysfs = Sysfs::default();
    for device in catalog::discover(&sysfs, catalog)? {
        println!(
            "{} {:?} {:?}",
            device.bdf, device.identity, device.properties
        );
    }
    let detected = platform::discover_with_catalog(&sysfs, catalog)?;
    println!("{:?}", detected.platform);
    Ok(())
}
