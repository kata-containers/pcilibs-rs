// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Reuse runtime validation without pulling target-side std dependencies into
// no_std builds. Build scripts always run on the host.
#[path = "src/catalog/input.rs"]
mod input;
#[allow(dead_code)] // The host validates records but does not perform lookups.
#[path = "src/catalog/records.rs"]
mod records;

#[cfg(feature = "std")]
#[path = "build_support.rs"]
mod linux;

fn main() {
    println!("cargo:rerun-if-changed=data");
    let catalog = input::read(std::path::Path::new("data"))
        .unwrap_or_else(|error| panic!("failed to bundle PCI catalogs: {error}"));
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(output.join("pci-devices.catalog"), catalog).unwrap();

    #[cfg(feature = "std")]
    linux::generate();
}
