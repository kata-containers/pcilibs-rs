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

`default-features = false` builds `catalog`, `gpu` and `platform` as `no_std`,
without an allocator, runtime dependencies, build dependencies, or the PCI-name
database.
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
The attachment facts follow NVIDIA gpu-admin-tools `main`.

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

### Extending the PCI device catalog

`catalog` identifies devices by PCI vendor, device, subsystem vendor and
subsystem device IDs. The bundled `data/pci-devices.catalog` contains GPU,
NIC and switch records. File names have no meaning for device matching; a file
can contain records for any vendor and device kind.

There are two ways to extend the database:

- **Bundle records:** add a `.catalog` file directly to `data/` and rebuild.
  The build validates and merges every `data/*.catalog` file into the built-in
  database, including when `default-features = false`. Adding, editing or removing
  a catalog triggers a rebuild. No Rust source changes or file list updates are needed.
- **Load records at runtime:** configure the application to call
  `catalog::CatalogFile::read` with a directory such as `/etc/pcilibs/devices.d`.
  Users can then add `.catalog` files there and restart or reload the application
  without rebuilding it. The same API also accepts a single file.

Directory loading reads immediate `.catalog` files, including symlinks to files;
it ignores other filenames and subdirectories. Each load returns a snapshot.
Applications pass its `catalog()` to discovery, classification or CC opening,
and call `read` again to pick up changes. Existing APIs without a catalog argument
use the bundled database. Neither workflow downloads data or changes global state.

The borrowed `catalog::Catalog::parse` API and lookup remain allocation-free and
available with `default-features = false`. Filesystem loading requires `std`.

Each record has seven whitespace-separated columns. There is no header line;
blank lines and whole-line `#` comments are allowed. These **already bundled**
records illustrate the format:

```text
# VENDOR DEVICE SUBSYSTEM_VENDOR SUBSYSTEM_DEVICE KIND PROFILE ATTACHMENT
10de 3041 * 221a gpu GR100 coherent
15b3 1021 * * nic ConnectX-7 pcie
15b3 d2f4 * * switch Quantum-3 pcie
```

The first four columns are hexadecimal PCI IDs; only the subsystem columns
allow `*`, asserting that the record applies to every value of that field.
`KIND` is `gpu`, `nic`, `switch`, `bridge` or `other`. `PROFILE` is a descriptive
name of up to 64 ASCII letters, digits, dots, underscores or hyphens.
`ATTACHMENT` is `pcie`, `sxm`, `coherent` or `unknown`. Vendor IDs always come
from the record, never the filename or a default vendor.

New vendors and profile names need no compiled registration. Verify new mappings
against hardware documentation. NVIDIA GPU attachment facts follow
`gpu-admin-tools` main; the bundled Mellanox NIC and Quantum PCI IDs follow
`mstflint`. Each file must be sorted numerically by the four identity columns,
with `*` before exact values. Files may cover interleaved ID ranges; the loader
merges them by identity.

Unlisted identities keep their built-in mappings. Overlapping records within an
extension, contradictory built-in mappings and malformed fields are errors.
Identical records across files are deduplicated; duplicates within a file are
errors. Identical built-in mappings are accepted so an extension survives a
library update that incorporates those IDs. Limits of 64 KiB and 1,024 records
apply to each complete input before deduplication, including the combined
contents of a directory. A directory can contain at most 1,024 catalog files.
Missing or invalid requested inputs are errors; empty directories add no mappings.

```rust,no_run
use pcilibs_rs::{catalog::{self, CatalogFile}, platform, Sysfs};

let extension = CatalogFile::read(std::path::Path::new("/etc/pcilibs/devices.d"))?;
let catalog = extension.catalog();
let sysfs = Sysfs::default();
// All PCI vendors and device kinds; unknown identities remain in the result.
for device in catalog::discover(&sysfs, catalog)? {
    println!("{} {:?} {:?}", device.bdf, device.identity, device.properties);
}
let detected = platform::discover_with_catalog(&sysfs, catalog)?;
println!("platform={:?}", detected.platform);
# Ok::<(), std::io::Error>(())
```

Pure callers can use `catalog.lookup(catalog::PciIdentity::new(vendor, device,
subsystem_vendor, subsystem_device))` or `platform::classify_with_catalog`.
CC consumers use `cc::Gpu::open_in_with_catalog` to apply the same exact mapping
before BAR access. Keep the loaded file alive for its borrowed catalog; replace
it explicitly between runs. The example accepts a file or directory:

```sh
cargo run --example platform -- /etc/pcilibs/devices.d
```

Catalog metadata describes identity; hardware operations still need a supported
implementation. NVIDIA platform classification and CC access check vendor,
device kind and a compatible compiled GPU register profile. An unknown vendor
or profile remains available to general discovery but cannot select NVIDIA
register access. Catalogs cannot supply registers, firmware commands or arbitrary
driver names. Coherent attachment and in-band CC capability remain independent;
new chip protocols still require code, and VFIO support still requires a
device-specific kernel alias. Deploying extensions into NVRC or the provisioner
is a separate consumer change.

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
in-band switch-ASIC enumeration. The generic catalog can identify a locally visible
Quantum PCI function; it does not discover remote InfiniBand switches or infer
a Quantum generation from an NVLink management PF. Rx00 management hardware
using the same interface can use these rules, but its topology has not been
hardware-validated.

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
| `src/cc/`, `src/gpu.rs`, `src/gpu/chips.rs`, `data/pci-devices.catalog` | MIT — a Rust port of the CC subset of NVIDIA's [`gpu-admin-tools`](https://github.com/NVIDIA/gpu-admin-tools) |
| `src/pci_dev.rs` | MIT — the generic PCI register access the port needed, which this crate did not have |
| everything else | Apache-2.0 |

Per-file SPDX headers are authoritative; `LICENSE` and `LICENSE-MIT` hold the
two texts. When adding a register, a PRC knob id or a chip to `src/cc/`, say in
a comment which `gpu-admin-tools` file it came from, so the port stays checkable
against its source.
