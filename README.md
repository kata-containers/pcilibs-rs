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
- Live register access to a device: BAR0 mapping, function-level reset, and
  forcing a runtime-suspended device out of D3 (`PciDev`)

### Minimal classification and Linux access

`default-features = false` builds `gpu` and `platform` as `no_std`, without an
allocator, runtime dependencies, build dependencies, or the PCI-name database.
The default `std` feature preserves the Linux sysfs, VFIO and BAR APIs. Consumers
that disable defaults must enable `std` explicitly to use those APIs. `cc` and
`testfs` imply `std`.

```toml
# Pure classification only.
pcilibs-rs = { git = "...", default-features = false }
# Linux discovery, without firmware CC access.
pcilibs-rs = { git = "...", default-features = false, features = ["std"] }
```

### Accelerator platform detection

`platform::discover(&sysfs)` returns the PCI topology and its platform
classification. PCIe-only GPUs report `Pcie(Family)` with no fabric; neither
NVSwitch devices nor RDMA drivers are required. A ServiceVM can select the H100/H200-style driver/FM path from
`FabricInterface::DirectNvSwitch`, or the Bx00/Rx00-style RDMA/NVLSM/FM path from
`FabricInterface::ConnectX`, even without GPUs or four visible management PFs.

When GPUs are visible, NVIDIA device and subsystem-device IDs distinguish SXM,
PCIe and coherent attachment. The pure `platform::classify` function combines
that evidence with the management interface to report `HgxHx00`, `HgxBx00`,
`HgxRx00`, `Pcie(Family)`, or `Coherent(Family)`. Mixed and unknown evidence remain explicit.
The attachment facts come from NVIDIA
[gpu-admin-tools v2026.09.29](https://github.com/NVIDIA/gpu-admin-tools/blob/44f261a7ebff96559488230b420e4a3035b30d58/gpu/devid_properties.py).

These are accelerator hardware profiles, also used in HGX-based OEM and DGX
systems. They do not prove an exact chassis model or NVL72 rack membership.
CPU identity is independent of GPU attachment and fabric management: a Vera CPU
alone implies neither C2C attachment nor NVL72 membership. Coherent Rubin GPUs
can coexist with ConnectX management; both facts remain in the result.
No SMBIOS strings or OEM model mappings are used. A switch-only ConnectX
assignment cannot prove Bx00 versus Rx00; its interface remains known while
its platform kind is `Unknown`. Exact family information would need additional
switch evidence or a trusted host-provided identity. Rx00 service compatibility
still needs hardware validation.

`cargo run --example platform` performs read-only discovery; VPD may require
root. GPU identities are shared with `cc` so classification needs no BAR access.
The device-only `cc::is_c2c` compatibility API remains conservative for aliased
IDs. VFIO driver selection uses kernel aliases independently of this property.

### GPU device-ID extensions

New variants of supported chips should not require rebuilding every consumer.
`gpu::catalog::Catalog` extends the bundled
[`data/nvidia-gpus.catalog`](data/nvidia-gpus.catalog) with caller-supplied records.
Parsing and lookup are allocation-free and available with `default-features = false`.
Linux callers can load a bounded file with `gpu::catalog::CatalogFile::read`.
Neither path downloads data or changes a process-global catalog.

The format has a version and a caller-assigned revision, followed by device ID,
subsystem device ID, existing chip profile, and attachment. For example, this
**already bundled** identity illustrates the format:

```text
pcilibs-nvidia-gpus 1 deployment-2026-09-30
3041 221a GR100 coherent
```

An extension normally contains only new mappings verified against NVIDIA's
`gpu-admin-tools` or hardware documentation. IDs are hexadecimal and records must
be sorted numerically by device/subsystem ID. Subsystem `*` is permitted only as
an explicit assertion that the mapping applies to every variant; it cannot
overlap exact records in the extension or contradict a built-in variant.
The vendor is implicitly NVIDIA (`10de`); this is a GPU catalog, not a general
PCI or VFIO driver database.

Unlisted identities keep their built-in mappings. Conflicts with built-in
identities or chip ranges, duplicate/overlapping extension records, unknown chip
profiles, malformed fields, and unsupported format versions are errors. Identical
built-in records are accepted so an extension survives a library update that
incorporates those IDs. Files are limited to 64 KiB and 1,024 extension records.
A missing or invalid requested file is an error, never an implicit fallback.

```rust,no_run
use pcilibs_rs::{gpu::catalog::CatalogFile, platform, Sysfs};

let extension = CatalogFile::read(std::path::Path::new("/etc/pcilibs/gpus.catalog"))?;
let catalog = extension.catalog();
let detected = platform::discover_with_catalog(&Sysfs::default(), catalog)?;
println!("extension={} platform={:?}", catalog.revision(), detected.platform);
# Ok::<(), std::io::Error>(())
```

Pure callers use `platform::classify_with_catalog`; CC consumers use
`cc::Gpu::open_in_with_catalog` to apply the same exact mapping before BAR access.
Existing discovery/opening APIs continue using built-in knowledge. Keep the
loaded file alive for its borrowed catalog; replace it explicitly between runs
and record both the built-in and extension revisions. The example accepts an
optional extension path: `cargo run --example platform -- /path/to/gpus.catalog`.

Catalogs are trusted hardware configuration: an incorrect new mapping can select
the wrong existing register profile. They cannot supply registers, firmware
commands, or arbitrary driver names. Coherent attachment and in-band CC capability
remain independent; new chip protocols still require code, and VFIO support still
requires a device-specific kernel alias. Deploying extensions into NVRC or the
provisioner is a separate consumer change.

### `cc` — in-band NVIDIA confidential computing

Off by default. Enables `pcilibs_rs::cc`: query and set a GPU's confidential
computing mode, and Protected PCIe across an HGX baseboard including its
NVSwitches.

```toml
pcilibs-rs = { git = "...", features = ["cc"] }
```

Everything goes through sysfs and a BAR0 mapping, so no NVIDIA kernel driver
has to be present — which is what lets a mode be set on a GPU that is already
bound to `vfio-pci`. Enabling the feature adds `anyhow`; a consumer that only
enumerates devices pulls neither it nor this code.

### NVLink discovery for init processes

`pcilibs_rs::nvlink` is available without the `cc` feature. It reads PCI
identity and VPD before GPU/RDMA drivers are loaded, so an init process can
choose which drivers and services to start:

```rust,no_run
use pcilibs_rs::{nvlink, Sysfs};

fn inspect_hardware() -> std::io::Result<()> {
    let sysfs = Sysfs::default();
    let topology = nvlink::discover(&sysfs)?;
    println!("GPUs: {:?}", topology.gpus);
    println!("Direct NVSwitches: {:?}", topology.switches);
    println!("Management PFs: {:?}", topology.management_functions);

    // Port GUIDs exist only after RDMA drivers register their devices.
    let ports = nvlink::discover_management_ports(&sysfs)?;
    if let Some(port) = ports.first() {
        let guid = format!("0x{:016x}", port.guid);
        println!("{} port {}: {guid}", port.ib_device, port.port);
    }
    Ok(())
}
```

For NVRC, `platform::discover()` supplies mode selection,
`platform::discover_topology()` supplies GPU/fabric presence for driver options,
and `nvlink::discover_management_ports()` supplies fabric startup with its GUID.
NVRC retains its mode policy and daemon/module startup. No external discovery
commands or new dependencies are required.

H100/H200 switches are NVIDIA Other Bridge PCI functions. B200/B300 management
ConnectX PFs are identified by the exact `SMDL=SW_MNG` field in PCI VPD, then
associated with present Mellanox PFs sharing the same domain/bus/device.
VFs are excluded. Count the returned functions as PFs, not physical switches:
only some PFs carry the marker, and firmware or VM assignment changes which
functions are visible. VPD reads may require root.

Management ports must belong to those PCI functions, use InfiniBand, and have
`isSMdisabled` (capability-mask bit 10) clear. Results include the BDF, IB device,
port number and GUID, sorted by BDF and numeric port. The GUID parser accepts
compressed and full GIDs and emits a numeric 64-bit GUID. Missing trees,
unreadable attributes and malformed data are errors; an existing tree with no
eligible devices returns an empty list. Missing optional PCI VPD means that
function has no marker.

These rules follow NVIDIA's [HGX integration guide, release 19.0,
§2.5.2](https://docs.nvidia.com/hgx-platforms/shared-nvswitch-gpu-passthrough-virtualization-integration-guide.pdf).
Discovery does not infer a switch generation from the marker or implement
in-band switch-ASIC enumeration. Rx00 management hardware using the same
interface can use these rules, but its topology has not been hardware-validated.

## Testing

For Linux builds, the PCI ID database is a submodule, so `git submodule update --init` before
the first build.

```bash
cargo test --no-default-features
cargo check --no-default-features --target aarch64-unknown-none
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
| `src/cc/`, `src/gpu.rs`, `data/nvidia-gpus.catalog` | MIT — a Rust port of the CC subset of NVIDIA's [`gpu-admin-tools`](https://github.com/NVIDIA/gpu-admin-tools) |
| `src/pci_dev.rs` | MIT — the generic PCI register access the port needed, which this crate did not have |
| everything else | Apache-2.0 |

Per-file SPDX headers are authoritative; `LICENSE` and `LICENSE-MIT` hold the
two texts. When adding a register, a PRC knob id or a chip to `src/cc/`, say in
a comment which `gpu-admin-tools` file it came from, so the port stays checkable
against its source.
