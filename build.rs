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

    #[cfg(feature = "fabric")]
    generate_fabric();
}

/// The fabric view wire types, from the proto with serde derives, so the
/// JSON a client prints and the message it streams are one schema.  The
/// names are fmpm's, so the HGX guide's shell snippets keep working.
#[cfg(feature = "fabric")]
fn generate_fabric() {
    println!("cargo:rerun-if-changed=proto");
    let descriptors = protox::compile(["proto/kata/fabric/v1/fabric.proto"], ["proto"])
        .unwrap_or_else(|error| panic!("failed to compile the fabric proto: {error}"));
    let mut config = prost_build::Config::new();
    config
        .type_attribute(
            ".kata.fabric.v1",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .type_attribute(".kata.fabric.v1", "#[serde(rename_all = \"camelCase\")]")
        .field_attribute(
            ".kata.fabric.v1.PartitionGpu.num_nvlinks_available",
            "#[serde(rename = \"numNvLinksAvailable\")]",
        )
        .field_attribute(
            ".kata.fabric.v1.PartitionGpu.max_num_nvlinks",
            "#[serde(rename = \"maxNumNvLinks\")]",
        )
        .field_attribute(
            ".kata.fabric.v1.PartitionGpu.nvlink_line_rate_mbps",
            "#[serde(rename = \"nvlinkLineRateMBps\")]",
        );
    // One generated file either way; with the stubs it also carries the
    // Publish service's two ends.
    #[cfg(feature = "fabric-grpc")]
    let generated = tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_fds_with_config(descriptors, config);
    #[cfg(not(feature = "fabric-grpc"))]
    let generated = config.compile_fds(descriptors);
    generated.unwrap_or_else(|error| panic!("failed to generate the fabric types: {error}"));
}
