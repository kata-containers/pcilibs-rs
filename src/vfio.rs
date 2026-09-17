// Copyright (c) 2026 Kata Containers contributors
//
// SPDX-License-Identifier: Apache-2.0

//! Binding a PCI function to a vfio driver, and taking it back.
//!
//! Runtime state only: nothing here survives a reboot.
//!
//! Everything goes through `driver_override` rather than a driver's own
//! `bind`, because the variant drivers match on nothing else.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::{context, failed, Sysfs, DRIVER_VFIO_PCI_TYPE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Already,
    Now,
}

/// A device's directory and the name to write into the control files under
/// `bus/pci`, which look a device up by the name it registered under: the
/// canonical address, not whichever spelling of it got us here.
fn resolve(sysfs: &Sysfs, address: &str) -> io::Result<(PathBuf, String)> {
    let device = sysfs.device(address).ok_or_else(|| {
        failed(
            io::ErrorKind::InvalidInput,
            format!("{address:?} is not a PCI address"),
        )
    })?;
    let name = device
        .file_name()
        .expect("a device directory")
        .to_string_lossy()
        .into_owned();

    Ok((device, name))
}

pub fn current_driver(sysfs: &Sysfs, address: &str) -> Option<String> {
    fs::read_link(sysfs.device(address)?.join("driver"))
        .ok()?
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
}

/// Refuses a device another driver holds rather than stealing it; unbind
/// first if that is what you want.
///
/// Binding is asynchronous, so the write is not read back; see
/// [`verify_bound`].
pub fn bind(sysfs: &Sysfs, address: &str, driver: &str) -> io::Result<Bound> {
    let (device, address) = resolve(sysfs, address)?;
    if !device.is_dir() {
        return Err(failed(
            io::ErrorKind::NotFound,
            format!("{address}: no such PCI device under {}", device.display()),
        ));
    }

    match current_driver(sysfs, &address).as_deref() {
        Some(current) if current == driver => return Ok(Bound::Already),
        Some(other) => {
            return Err(failed(
                io::ErrorKind::ResourceBusy,
                format!("{address}: held by driver {other:?}, refusing to take it"),
            ))
        }
        None => {}
    }

    fs::write(device.join("driver_override"), driver)
        .map_err(|err| context(err, format!("{address}: set driver_override to {driver}")))?;
    fs::write(sysfs.drivers_probe(), &address).map_err(|err| {
        // An override left behind outlives the failure: the next probe, at
        // the latest the next boot, binds a device this said it had not.
        let _ = clear_override(&device);
        context(err, format!("{address}: trigger driver probe"))
    })?;

    Ok(Bound::Now)
}

/// The kernel rejects an empty write; a newline is how it spells "none".
fn clear_override(device: &Path) -> io::Result<()> {
    fs::write(device.join("driver_override"), "\n")
}

pub fn verify_bound(sysfs: &Sysfs, address: &str, driver: &str) -> io::Result<()> {
    match current_driver(sysfs, address).as_deref() {
        Some(current) if current == driver => Ok(()),
        Some(other) => Err(failed(
            io::ErrorKind::Other,
            format!("{address}: bound to {other:?}, expected {driver}"),
        )),
        None => Err(failed(
            io::ErrorKind::Other,
            format!("{address}: still unbound after probe"),
        )),
    }
}

/// Clears the override too, so the next probe matches as it would on an
/// unprovisioned node.
pub fn unbind(sysfs: &Sysfs, address: &str) -> io::Result<()> {
    let (device, address) = resolve(sysfs, address)?;

    if current_driver(sysfs, &address).is_some() {
        fs::write(device.join("driver").join("unbind"), &address)
            .map_err(|err| context(err, format!("{address}: unbind")))?;
    }

    clear_override(&device)
        .map_err(|err| context(err, format!("{address}: clear driver_override")))?;

    Ok(())
}

/// The vfio module for a device, from the kernel's own alias table.
///
/// A variant driver — `nvgrace_gpu_vfio_pci` on Grace, `mlx5_vfio_pci` —
/// registers a `vfio_pci:` alias for exactly the devices it supports. Those
/// aliases are `override_only`: the driver never matches through normal PCI
/// probing, so nothing but this table says which module a device wants, and
/// no list of device ids here could stay correct as the kernel adds them.
/// No alias means plain `vfio-pci`, which takes whatever it is given.
pub fn module_for(modules_alias: &Path, vendor: u16, device: u16) -> io::Result<String> {
    let wanted = format!("vfio_pci:v{vendor:08X}d{device:08X}");

    let table = fs::read_to_string(modules_alias)
        .map_err(|err| context(err, format!("read {}", modules_alias.display())))?;

    let module = table
        .lines()
        .find(|line| line.contains(&wanted))
        .and_then(|line| line.split_whitespace().next_back())
        .unwrap_or(DRIVER_VFIO_PCI_TYPE);

    Ok(module.to_string())
}

/// The PCI driver a loaded module implements. Not the module's own name:
/// modprobe wants `vfio_pci` where `driver_override` wants `vfio-pci`, and
/// only the module knows which is which. The two happen to be identical for
/// `nvgrace_gpu_vfio_pci`, which is why guessing at the punctuation cannot
/// be relied on either way.
pub fn driver_for(sysfs: &Sysfs, module: &str) -> io::Result<String> {
    let drivers = sysfs
        .module(module)
        .ok_or_else(|| {
            failed(
                io::ErrorKind::InvalidInput,
                format!("{module:?} is not a module name"),
            )
        })?
        .join("drivers");

    let entries = fs::read_dir(&drivers).map_err(|err| {
        context(
            err,
            format!("read {} (is {module} loaded?)", drivers.display()),
        )
    })?;

    for entry in entries {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if let Some(driver) = name.strip_prefix("pci:") {
            return Ok(driver.to_string());
        }
    }

    Err(failed(
        io::ErrorKind::NotFound,
        format!("module {module} registers no PCI driver"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs::{self, Fake};
    use rstest::{fixture, rstest};
    use std::path::PathBuf;

    #[fixture]
    fn fake() -> Fake {
        testfs::fake()
    }

    /// Grace GPUs need a variant driver, and it is only ever reachable
    /// through the override this writes.
    #[rstest]
    #[case::vfio_pci("vfio-pci")]
    #[case::grace("nvgrace-gpu-vfio-pci")]
    fn binds_an_unbound_device_to_the_driver_it_asked_for(fake: Fake, #[case] driver: &str) {
        fake.add_device("0000:65:00.0", None);

        assert_eq!(
            bind(&fake.sysfs, "0000:65:00.0", driver).unwrap(),
            Bound::Now
        );

        assert_eq!(fake.driver_override("0000:65:00.0"), driver);
        assert_eq!(fake.drivers_probe(), "0000:65:00.0");
    }

    #[rstest]
    #[case::vfio_pci("vfio-pci")]
    #[case::grace("nvgrace-gpu-vfio-pci")]
    fn binding_an_already_bound_device_succeeds_and_writes_nothing(
        fake: Fake,
        #[case] driver: &str,
    ) {
        fake.add_device("0000:65:00.0", Some(driver));

        assert_eq!(
            bind(&fake.sysfs, "0000:65:00.0", driver).unwrap(),
            Bound::Already
        );

        assert_eq!(fake.drivers_probe(), "");
    }

    /// A GPU sitting on plain vfio-pci when it needs the Grace driver is not
    /// "already bound": it would be handed to a VM with its coherent memory
    /// missing.
    #[rstest]
    fn refuses_a_device_bound_to_the_wrong_vfio_driver(fake: Fake) {
        fake.add_device("0000:65:00.0", Some("vfio-pci"));

        let err = bind(&fake.sysfs, "0000:65:00.0", "nvgrace-gpu-vfio-pci")
            .unwrap_err()
            .to_string();

        assert!(err.contains("vfio-pci"), "{err}");
    }

    #[rstest]
    #[case::host_gpu_driver("nvidia")]
    #[case::open_source_gpu_driver("nouveau")]
    #[case::anything_else("igb")]
    fn refuses_a_device_another_driver_holds(fake: Fake, #[case] driver: &str) {
        fake.add_device("0000:65:00.0", Some(driver));

        let err = bind(&fake.sysfs, "0000:65:00.0", "vfio-pci")
            .unwrap_err()
            .to_string();

        assert!(err.contains(driver), "{err}");
        assert_eq!(fake.drivers_probe(), "");
    }

    /// The control files find a device by the name it registered under, so a
    /// shorthand address has to be spelled out before it is written there, or
    /// the probe lands on nothing with the override already set.
    #[rstest]
    #[case::canonical("0000:65:00.0")]
    #[case::no_domain("65:00.0")]
    #[case::unpadded("0:65:0.0")]
    fn probes_the_address_the_kernel_knows_the_device_by(fake: Fake, #[case] address: &str) {
        fake.add_device(address, None);

        bind(&fake.sysfs, address, "vfio-pci").unwrap();

        assert_eq!(fake.drivers_probe(), "0000:65:00.0");
    }

    #[rstest]
    #[case::canonical("0000:65:00.0")]
    #[case::no_domain("65:00.0")]
    #[case::unpadded("0:65:0.0")]
    fn unbinds_the_address_the_kernel_knows_the_device_by(fake: Fake, #[case] address: &str) {
        fake.add_device(address, Some("vfio-pci"));

        unbind(&fake.sysfs, address).unwrap();

        assert_eq!(fake.driver_unbind("vfio-pci"), "0000:65:00.0");
    }

    #[rstest]
    fn a_refused_bind_takes_its_override_back(fake: Fake) {
        fake.add_device("0000:65:00.0", None);
        // A directory is what makes the kernel's own file reject a write.
        fs::remove_file(fake.sysfs.drivers_probe()).unwrap();
        fs::create_dir(fake.sysfs.drivers_probe()).unwrap();

        bind(&fake.sysfs, "0000:65:00.0", "vfio-pci").unwrap_err();

        assert_eq!(fake.driver_override("0000:65:00.0").trim(), "");
    }

    #[rstest]
    fn refuses_a_device_that_is_not_there(fake: Fake) {
        assert!(bind(&fake.sysfs, "0000:65:00.0", "vfio-pci").is_err());
    }

    /// Refused before the join: an address is joined onto the sysfs root
    /// here as much as anywhere else.
    #[rstest]
    #[case::traversal("../../../etc")]
    #[case::absolute("/etc/shadow")]
    #[case::nonsense("wat")]
    fn refuses_an_address_that_is_not_one(fake: Fake, #[case] address: &str) {
        let err = bind(&fake.sysfs, address, "vfio-pci").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{err}");
        assert!(unbind(&fake.sysfs, address).is_err());
    }

    #[rstest]
    #[case::bound(Some("vfio-pci"), true)]
    #[case::bound_elsewhere(Some("nvidia"), false)]
    #[case::wrong_vfio_driver(Some("nvgrace-gpu-vfio-pci"), false)]
    #[case::unbound(None, false)]
    fn verify_reads_the_driver_back(
        fake: Fake,
        #[case] driver: Option<&str>,
        #[case] expected: bool,
    ) {
        fake.add_device("0000:65:00.0", driver);

        assert_eq!(
            verify_bound(&fake.sysfs, "0000:65:00.0", "vfio-pci").is_ok(),
            expected
        );
    }

    #[rstest]
    fn unbind_releases_and_clears_the_override(fake: Fake) {
        fake.add_device("0000:65:00.0", Some("vfio-pci"));

        unbind(&fake.sysfs, "0000:65:00.0").unwrap();

        assert_eq!(fake.driver_unbind("vfio-pci"), "0000:65:00.0");
        assert_eq!(fake.driver_override("0000:65:00.0").trim(), "");
    }

    #[rstest]
    fn unbind_of_an_unbound_device_only_clears_the_override(fake: Fake) {
        fake.add_device("0000:65:00.0", None);

        unbind(&fake.sysfs, "0000:65:00.0").unwrap();

        assert_eq!(fake.driver_override("0000:65:00.0").trim(), "");
    }

    /// Lines as `depmod` writes them, including vfio-pci's catch-all, which
    /// must not be mistaken for a variant driver's exact match.
    const ALIASES: &str = "\
alias vfio_pci:v000015B3d0000101Esv*sd*bc*sc*i* mlx5_vfio_pci
alias vfio_pci:v000010DEd00002342sv*sd*bc*sc*i* nvgrace_gpu_vfio_pci
alias vfio_pci:v000010DEd00002345sv*sd*bc*sc*i* nvgrace_gpu_vfio_pci
alias vfio_pci:v000010DEd00002941sv*sd*bc*sc*i* nvgrace_gpu_vfio_pci
alias vfio_pci:v*d*sv*sd*bc*sc*i* vfio_pci
alias pci:v000010DEd00002330sv*sd*bc*sc*i* nvidia
";

    /// Not a fixture: the table has to outlive the tree it sits in.
    fn aliases(fake: &Fake) -> PathBuf {
        let path = fake.root().join("modules.alias");
        fs::write(&path, ALIASES).unwrap();
        path
    }

    #[rstest]
    #[case::gh200_120gb(0x2342, "nvgrace_gpu_vfio_pci")]
    #[case::gh200_480gb(0x2345, "nvgrace_gpu_vfio_pci")]
    #[case::gb200(0x2941, "nvgrace_gpu_vfio_pci")]
    #[case::h100_pcie(0x2330, "vfio-pci")]
    #[case::unknown_to_this_kernel(0xffff, "vfio-pci")]
    fn reads_the_module_out_of_the_alias_table(
        fake: Fake,
        #[case] device: u16,
        #[case] expected: &str,
    ) {
        let aliases = aliases(&fake);

        assert_eq!(module_for(&aliases, 0x10de, device).unwrap(), expected);
    }

    /// The vendor is part of the match: another vendor's device with the same
    /// device id must not be handed NVIDIA's driver.
    #[rstest]
    fn the_alias_match_is_on_vendor_and_device(fake: Fake) {
        let aliases = aliases(&fake);

        assert_eq!(module_for(&aliases, 0x8086, 0x2342).unwrap(), "vfio-pci");
        assert_eq!(
            module_for(&aliases, 0x15b3, 0x101e).unwrap(),
            "mlx5_vfio_pci"
        );
    }

    #[rstest]
    #[case::underscores("nvgrace_gpu_vfio_pci")]
    #[case::dashes("nvgrace-gpu-vfio-pci")]
    fn resolves_a_module_to_the_driver_it_registers(fake: Fake, #[case] module: &str) {
        fake.add_module("nvgrace_gpu_vfio_pci", "nvgrace-gpu-vfio-pci");

        assert_eq!(
            driver_for(&fake.sysfs, module).unwrap(),
            "nvgrace-gpu-vfio-pci"
        );
    }

    #[rstest]
    fn a_module_that_is_not_loaded_has_no_driver(fake: Fake) {
        assert!(driver_for(&fake.sysfs, "nvgrace_gpu_vfio_pci").is_err());
    }

    /// Refused before the join: a module name becomes a path here too.
    #[rstest]
    #[case::traversal("../../../etc")]
    #[case::absolute("/etc")]
    #[case::separator("vfio_pci/..")]
    #[case::empty("")]
    fn refuses_a_module_name_that_is_not_one(fake: Fake, #[case] module: &str) {
        let err = driver_for(&fake.sysfs, module).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{err}");
    }

    /// Loaded but registering no PCI driver is not the same as not loaded.
    #[rstest]
    fn a_module_registering_no_pci_driver_has_none(fake: Fake) {
        let drivers = fake
            .sysfs
            .module("vfio")
            .expect("a module name")
            .join("drivers");
        fs::create_dir_all(drivers.join("vfio_group")).unwrap();

        let err = driver_for(&fake.sysfs, "vfio").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound, "{err}");
    }
}
