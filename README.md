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

Apache-2.0, except for the in-band confidential computing code, which is MIT
under NVIDIA's copyright:

| Path | License |
| --- | --- |
| `src/cc/` | MIT — a Rust port of the CC subset of NVIDIA's [`gpu-admin-tools`](https://github.com/NVIDIA/gpu-admin-tools) |
| `src/pci_dev.rs` | MIT — the generic PCI register access the port needed, which this crate did not have |
| everything else | Apache-2.0 |

Per-file SPDX headers are authoritative; `LICENSE` and `LICENSE-MIT` hold the
two texts. When adding a register, a PRC knob id or a chip to `src/cc/`, say in
a comment which `gpu-admin-tools` file it came from, so the port stays checkable
against its source.
