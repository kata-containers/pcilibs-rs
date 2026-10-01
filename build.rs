// Copyright (c) NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Reuse runtime validation without pulling target-side std dependencies into
// no_std builds. Build scripts always run on the host.
#[allow(dead_code)]
#[path = "src/gpu/chips.rs"]
mod gpu;
#[path = "src/gpu/catalog/input.rs"]
mod input;
#[path = "src/gpu/catalog/records.rs"]
mod records;

#[cfg(feature = "std")]
#[path = "build_support.rs"]
mod linux;

fn main() {
    println!("cargo:rerun-if-changed=data");
    let catalog = input::read(std::path::Path::new("data"))
        .unwrap_or_else(|error| panic!("failed to bundle GPU catalogs: {error}"));
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(output.join("nvidia-gpus.catalog"), catalog).unwrap();

    #[cfg(feature = "std")]
    linux::generate();
}
