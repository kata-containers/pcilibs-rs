# pcilibs-rs

PCI device enumeration and diagnostics for Linux container runtimes.

This crate provides helpers for reading PCI devices from sysfs, detecting PCIe
devices, checking VFIO driver types, and capturing InfiniBand diagnostics
snapshots.

## Features

- PCI device enumeration via sysfs (`PCIDeviceManager`)
- PCIe detection from config space size
- VFIO driver type matching
- InfiniBand / uverbs diagnostic snapshots

## Testing

The PCI ID database is a submodule, so `git submodule update --init` before
the first build.

```bash
cargo test --all-features -- --include-ignored

# Coverage, as CI gates it: 90% of lines, over the tree and per file.
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --locked --version 0.8.7
cargo llvm-cov --all-features --workspace \
    --fail-under-lines 90 --fail-under-file-lines 90 -- --include-ignored
```

## License

Apache-2.0
