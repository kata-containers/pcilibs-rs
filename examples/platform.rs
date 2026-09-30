// SPDX-License-Identifier: Apache-2.0

use pcilibs_rs::{
    gpu::catalog::{Catalog, CatalogFile},
    platform, Sysfs,
};

fn main() -> std::io::Result<()> {
    let extension = std::env::args_os()
        .nth(1)
        .map(|path| CatalogFile::read(std::path::Path::new(&path)))
        .transpose()?;
    let catalog = extension
        .as_ref()
        .map_or(Catalog::builtin(), CatalogFile::catalog);
    println!("Built-in GPU catalog: {}", Catalog::builtin().revision());
    if extension.is_some() {
        println!("GPU catalog extension: {}", catalog.revision());
    }
    let detected = platform::discover_with_catalog(&Sysfs::default(), catalog)?;
    println!("{:?}", detected.platform);
    Ok(())
}
