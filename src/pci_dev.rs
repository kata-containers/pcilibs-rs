// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

//! Live register access to a PCI function: BAR0 mapping, function-level
//! reset, and forcing the device out of runtime suspend so MMIO reads mean
//! something.
//!
//! Ported from NVIDIA's `gpu-admin-tools`, hence the MIT header, but nothing
//! here is vendor-specific: it is the half of sysfs PCI that
//! [`PCIDevice`](crate::PCIDevice) enumeration does not cover.
//!
//! All paths are kernel contracts (`/sys/bus/pci/devices/<bdf>/...`), not
//! configuration.

use std::fs::OpenOptions;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{context, failed, normalize_bdf, Sysfs};

const PCI_COMMAND: u64 = 0x04;
const PCI_COMMAND_MEMORY: u16 = 0x0002;

/// Read a sysfs attribute that prints as hex (`0x10de`).
pub fn attr_hex(dir: &Path, name: &str) -> io::Result<u32> {
    let text = std::fs::read_to_string(dir.join(name))
        .map_err(|err| context(err, format!("read {}/{name}", dir.display())))?;
    let text = text.trim().trim_start_matches("0x");
    u32::from_str_radix(text, 16).map_err(|err| {
        failed(
            io::ErrorKind::InvalidData,
            format!("parse {}/{name}: {err}", dir.display()),
        )
    })
}

/// Poll until `done` returns true, at 1 ms granularity.
pub fn poll(what: &str, timeout: Duration, mut done: impl FnMut() -> bool) -> io::Result<()> {
    let start = Instant::now();
    loop {
        if done() {
            return Ok(());
        }
        if start.elapsed() > timeout {
            return Err(failed(
                io::ErrorKind::TimedOut,
                format!("timed out waiting for {what}"),
            ));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// One PCI function with its BAR0 mapped for register access.
#[derive(Debug)]
pub struct PciDev {
    pub bdf: String,
    pub vendor: u16,
    pub device: u16,
    path: PathBuf,
    bar0: Bar0,
    /// Previous `power/control` policy, restored on drop.
    power_control: Option<String>,
    decoding_enabled: bool,
}

impl PciDev {
    /// Map a device's BAR0 (`resource0`) on the running kernel.  The domain
    /// may be omitted (`65:00.0` means `0000:65:00.0`).  Needs root.
    pub fn open(bdf: &str) -> io::Result<Self> {
        Self::open_in(&Sysfs::default(), bdf)
    }

    /// Open under a given sysfs, so a caller that enumerated under a test
    /// tree does not silently reach into `/sys` here.
    pub fn open_in(sysfs: &Sysfs, bdf: &str) -> io::Result<Self> {
        let bdf = normalize_bdf(bdf).ok_or_else(|| {
            failed(
                io::ErrorKind::InvalidInput,
                format!("{bdf:?} is not a PCI address"),
            )
        })?;
        let path = sysfs.devices().join(&bdf);
        if !path.exists() {
            return Err(failed(
                io::ErrorKind::NotFound,
                format!("no PCI device {bdf} under {}", sysfs.devices().display()),
            ));
        }
        // Drop runs only once there is a Self, so what can fail comes first
        // and the two switches it leaves set come last.
        let vendor = attr_hex(&path, "vendor")? as u16;
        let device = attr_hex(&path, "device")? as u16;
        let bar0 = Bar0::Mapped(Mapping::map(&path.join("resource0"))?);

        let power_control = wake(&path)?;
        let decoding_enabled = enable_memory_decoding(&path)
            .inspect_err(|_| restore_power(&path, power_control.as_deref()))?;

        Ok(Self {
            vendor,
            device,
            bar0,
            bdf,
            path,
            power_control,
            decoding_enabled,
        })
    }

    pub fn read32(&self, offset: u32) -> u32 {
        self.bar0.read32(offset)
    }

    pub fn write32(&self, offset: u32, value: u32) {
        self.bar0.write32(offset, value)
    }

    /// Function-level reset via sysfs.  The kernel saves and restores config
    /// space around it, so the mapping stays valid.
    pub fn sysfs_reset(&self) -> io::Result<()> {
        std::fs::write(self.path.join("reset"), "1")
            .map_err(|err| context(err, format!("{}: reset via sysfs", self.bdf)))
    }
}

/// A device whose registers are modelled rather than mapped: the FSP
/// mailboxes in [`crate::cc`] are a protocol, not a memory, so a `resource0`
/// file can hold register values but cannot answer an RPC.
#[cfg(test)]
impl PciDev {
    pub(crate) fn modelled(
        path: PathBuf,
        bdf: &str,
        vendor: u16,
        device: u16,
        registers: Box<dyn Registers>,
    ) -> Self {
        Self {
            bdf: bdf.to_string(),
            vendor,
            device,
            path,
            bar0: Bar0::Modelled(registers),
            power_control: None,
            decoding_enabled: false,
        }
    }
}

/// See [`PciDev::modelled`].
#[cfg(test)]
pub(crate) trait Registers: std::fmt::Debug + Send {
    fn read32(&self, offset: u32) -> u32;
    fn write32(&self, offset: u32, value: u32);
}

impl Drop for PciDev {
    fn drop(&mut self) {
        if self.decoding_enabled {
            let _ = set_memory_decoding(&self.path, false);
        }
        restore_power(&self.path, self.power_control.as_deref());
    }
}

/// An unbound device answers no memory cycles, yet `resource0` still maps, so
/// every register reads all-ones and the device looks dead.
fn enable_memory_decoding(path: &Path) -> io::Result<bool> {
    if command(path)? & PCI_COMMAND_MEMORY != 0 {
        return Ok(false);
    }

    set_memory_decoding(path, true)?;
    Ok(true)
}

fn command(path: &Path) -> io::Result<u16> {
    let config = path.join("config");
    let mut bytes = [0u8; 2];
    OpenOptions::new()
        .read(true)
        .open(&config)
        .and_then(|file| file.read_exact_at(&mut bytes, PCI_COMMAND))
        .map_err(|err| {
            context(
                err,
                format!("read COMMAND from {} (needs root)", config.display()),
            )
        })?;

    Ok(u16::from_le_bytes(bytes))
}

fn set_memory_decoding(path: &Path, on: bool) -> io::Result<()> {
    let current = command(path)?;
    let updated = if on {
        current | PCI_COMMAND_MEMORY
    } else {
        current & !PCI_COMMAND_MEMORY
    };

    let config = path.join("config");
    OpenOptions::new()
        .write(true)
        .open(&config)
        .and_then(|file| file.write_all_at(&updated.to_le_bytes(), PCI_COMMAND))
        .map_err(|err| {
            context(
                err,
                format!("set memory decoding on {} (needs root)", config.display()),
            )
        })
}

/// A runtime-suspended device (D3) reads all-ones on MMIO.  Force it to D0
/// by switching the runtime-PM policy to `on`; returns the previous policy
/// so drop can restore it.
fn wake(path: &Path) -> io::Result<Option<String>> {
    let control = path.join("power/control");
    // No power/control at all is a device the kernel never suspends: it is
    // already in D0, and there is no policy to put back.
    let Ok(previous) = std::fs::read_to_string(&control) else {
        return Ok(None);
    };
    let previous = previous.trim().to_string();
    std::fs::write(&control, "on")
        .map_err(|err| context(err, format!("set {} to on (needs root)", control.display())))?;
    let status = path.join("power/runtime_status");
    poll("device wakeup from D3", Duration::from_secs(5), || {
        // "unsupported" is runtime PM disabled: in D0, and never "active".
        std::fs::read_to_string(&status).is_ok_and(|s| matches!(s.trim(), "active" | "unsupported"))
    })
    .inspect_err(|_| restore_power(path, Some(&previous)))?;
    Ok(Some(previous))
}

fn restore_power(path: &Path, previous: Option<&str>) {
    if let Some(previous) = previous {
        let _ = std::fs::write(path.join("power/control"), previous);
    }
}

/// A device's BAR0: the mapping on real hardware, a model of one in tests.
#[derive(Debug)]
enum Bar0 {
    Mapped(Mapping),
    #[cfg(test)]
    Modelled(Box<dyn Registers>),
}

impl Bar0 {
    fn read32(&self, offset: u32) -> u32 {
        match self {
            Self::Mapped(mapping) => mapping.read32(offset),
            #[cfg(test)]
            Self::Modelled(registers) => registers.read32(offset),
        }
    }

    fn write32(&self, offset: u32, value: u32) {
        match self {
            Self::Mapped(mapping) => mapping.write32(offset, value),
            #[cfg(test)]
            Self::Modelled(registers) => registers.write32(offset, value),
        }
    }
}

/// MMIO mapping of `resource0`.  Sysfs resource files reject read()/write();
/// mmap is the only access path.
#[derive(Debug)]
struct Mapping {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the pointer is a plain MMIO address; volatile accesses are not
// tied to the owning thread.  Deliberately not Sync — the FSP RPC
// sequences on top of this are not safe to interleave.
unsafe impl Send for Mapping {}

impl Mapping {
    fn map(resource0: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(resource0)
            .map_err(|err| context(err, format!("open {} (needs root)", resource0.display())))?;
        let len = file.metadata()?.len() as usize;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(context(
                io::Error::last_os_error(),
                format!("mmap {}", resource0.display()),
            ));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    /// A misaligned `u32` access is undefined behaviour rather than a slow
    /// one, and every register offset is a constant: this is a caller bug.
    fn dword_at(&self, offset: u32, access: &str) -> usize {
        let offset = offset as usize;
        assert!(
            offset % 4 == 0,
            "BAR0 {access} not dword aligned: {offset:#x}"
        );
        assert!(
            offset.checked_add(4).is_some_and(|end| end <= self.len),
            "BAR0 {access} past end: {offset:#x}"
        );
        offset
    }

    fn read32(&self, offset: u32) -> u32 {
        let offset = self.dword_at(offset, "read");
        unsafe { (self.ptr.add(offset) as *const u32).read_volatile() }
    }

    fn write32(&self, offset: u32, value: u32) {
        let offset = self.dword_at(offset, "write");
        unsafe { (self.ptr.add(offset) as *mut u32).write_volatile(value) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfs::{self, Fake};
    use std::os::unix::fs::PermissionsExt;

    const BDF: &str = "0000:65:00.0";

    /// One page is all these tests reach; a GPU's BAR0 is megabytes.
    const BAR0_LEN: u64 = 4096;

    fn fake_device(runtime_status: &str) -> Fake {
        let fake = testfs::fake();
        fake.add_mappable_device(BDF, 0x10de, 0x2330, 0x030200, BAR0_LEN);
        std::fs::write(
            fake.device(BDF).join("power/runtime_status"),
            format!("{runtime_status}\n"),
        )
        .unwrap();

        fake
    }

    #[rstest::rstest]
    fn an_attribute_that_does_not_read_as_hex_is_refused() {
        let fake = fake_device("active");
        std::fs::write(fake.device(BDF).join("vendor"), "nonsense\n").unwrap();

        let err = attr_hex(&fake.device(BDF), "vendor").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "{err}");
    }

    #[rstest::rstest]
    fn poll_waits_for_what_it_is_told_to() {
        let mut asked = 0;

        poll("a second look", Duration::from_secs(5), || {
            asked += 1;
            asked > 1
        })
        .unwrap();

        assert_eq!(asked, 2);
    }

    /// The deadline is checked after the predicate, so even an expired one
    /// asks once: a register that is already right is not a timeout.
    #[rstest::rstest]
    fn poll_gives_up_once_the_timeout_has_passed() {
        let mut asked = 0;

        let err = poll("nothing at all", Duration::ZERO, || {
            asked += 1;
            false
        })
        .unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::TimedOut, "{err}");
        assert!(asked >= 1);
    }

    #[rstest::rstest]
    #[case::active("active")]
    #[case::runtime_pm_disabled("unsupported")]
    fn opens_a_device_that_is_not_suspended(#[case] status: &str) {
        let fake = fake_device(status);

        let dev = PciDev::open_in(&fake.sysfs, BDF).unwrap();

        assert_eq!((dev.vendor, dev.device), (0x10de, 0x2330));
    }

    /// The only entry point that reads the running kernel, so all a test can
    /// hold it to is refusing what is not an address before it looks.
    #[rstest::rstest]
    fn open_reads_the_running_kernel() {
        let err = PciDev::open("nonsense").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{err}");
    }

    #[rstest::rstest]
    fn open_refuses_a_device_that_is_not_there() {
        let fake = testfs::fake();

        let err = PciDev::open_in(&fake.sysfs, BDF).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound, "{err}");
    }

    #[rstest::rstest]
    fn a_zero_length_bar0_fails_to_map() {
        let fake = fake_device("active");
        std::fs::write(fake.device(BDF).join("resource0"), []).unwrap();

        let err = PciDev::open_in(&fake.sysfs, BDF).unwrap_err().to_string();

        assert!(err.contains("mmap"), "{err}");
    }

    #[rstest::rstest]
    fn a_failed_open_leaves_the_power_policy_alone() {
        let fake = fake_device("active");
        let path = fake.device(BDF);
        // No config file: the COMMAND read fails after wake has written.
        std::fs::remove_file(path.join("config")).unwrap();

        let err = PciDev::open_in(&fake.sysfs, BDF).unwrap_err().to_string();

        assert!(err.contains("COMMAND"), "{err}");
        assert_eq!(
            std::fs::read_to_string(path.join("power/control"))
                .unwrap()
                .trim(),
            "auto"
        );
    }

    /// Root writes through the mode bits, so it cannot see the failure
    /// staged here; CI is unprivileged.
    fn read_only(path: &Path) -> bool {
        if unsafe { libc::geteuid() } == 0 {
            return false;
        }
        let mut mode = std::fs::metadata(path).unwrap().permissions();
        mode.set_mode(0o444);
        std::fs::set_permissions(path, mode).unwrap();

        true
    }

    /// A device left in D3 answers all-ones, so a wakeup that did not happen
    /// has to be an error rather than a device that reads as dead.
    #[rstest::rstest]
    fn a_device_that_cannot_be_woken_fails_to_open() {
        let fake = fake_device("suspended");
        if !read_only(&fake.device(BDF).join("power/control")) {
            return;
        }

        let err = PciDev::open_in(&fake.sysfs, BDF).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied, "{err}");
    }

    /// A device the kernel never suspends has no policy for drop to put back.
    #[rstest::rstest]
    fn a_device_without_runtime_pm_opens_anyway() {
        let fake = fake_device("active");
        std::fs::remove_dir_all(fake.device(BDF).join("power")).unwrap();

        let dev = PciDev::open_in(&fake.sysfs, BDF).unwrap();

        assert!(dev.power_control.is_none());
    }

    fn command_byte(fake: &Fake) -> u8 {
        std::fs::read(fake.device(BDF).join("config")).unwrap()[PCI_COMMAND as usize]
    }

    #[rstest::rstest]
    fn opening_a_silent_device_turns_memory_decoding_on() {
        let fake = fake_device("active");

        let dev = PciDev::open_in(&fake.sysfs, BDF).unwrap();
        assert_eq!(command_byte(&fake), PCI_COMMAND_MEMORY as u8);
        drop(dev);

        assert_eq!(command_byte(&fake), 0);
    }

    /// A device a driver already enabled is left as it was found, so drop
    /// does not switch it off underneath that driver.
    #[rstest::rstest]
    fn a_device_already_decoding_memory_is_left_as_it_was() {
        let fake = fake_device("active");
        let mut config = [0u8; 64];
        config[PCI_COMMAND as usize] = PCI_COMMAND_MEMORY as u8;
        std::fs::write(fake.device(BDF).join("config"), config).unwrap();

        drop(PciDev::open_in(&fake.sysfs, BDF).unwrap());

        assert_eq!(command_byte(&fake), PCI_COMMAND_MEMORY as u8);
    }

    #[rstest::rstest]
    fn a_config_space_that_will_not_take_the_command_bit_fails_the_open() {
        let fake = fake_device("active");
        if !read_only(&fake.device(BDF).join("config")) {
            return;
        }

        let err = PciDev::open_in(&fake.sysfs, BDF).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied, "{err}");
    }

    /// The fake outlives the device: it owns the mapped `resource0`.
    fn mapped_device() -> (Fake, PciDev) {
        let fake = fake_device("active");
        let dev = PciDev::open_in(&fake.sysfs, BDF).unwrap();

        (fake, dev)
    }

    #[rstest::rstest]
    fn a_register_round_trips_through_bar0() {
        let (_fake, dev) = mapped_device();

        dev.write32(0xffc, 0xdead_beef);

        assert_eq!(dev.read32(0xffc), 0xdead_beef);
    }

    #[rstest::rstest]
    fn a_reset_goes_through_sysfs() {
        let (fake, dev) = mapped_device();

        dev.sysfs_reset().unwrap();

        assert_eq!(
            std::fs::read_to_string(fake.device(BDF).join("reset")).unwrap(),
            "1"
        );
    }

    #[rstest::rstest]
    fn a_reset_the_kernel_will_not_take_is_reported() {
        let (fake, dev) = mapped_device();
        if !read_only(&fake.device(BDF).join("reset")) {
            return;
        }

        let err = dev.sysfs_reset().unwrap_err().to_string();

        assert!(err.contains("reset via sysfs"), "{err}");
    }

    #[rstest::rstest]
    #[should_panic(expected = "BAR0 read not dword aligned")]
    fn a_misaligned_read_is_refused() {
        let (_fake, dev) = mapped_device();

        dev.read32(0x201);
    }

    #[rstest::rstest]
    #[should_panic(expected = "BAR0 write not dword aligned")]
    fn a_misaligned_write_is_refused() {
        let (_fake, dev) = mapped_device();

        dev.write32(0x202, 0);
    }

    #[rstest::rstest]
    #[should_panic(expected = "BAR0 read past end")]
    fn a_register_outside_bar0_is_refused() {
        let (_fake, dev) = mapped_device();

        dev.read32(0x1000);
    }

    /// Refused before the join, so the error is about the address rather than
    /// a missing device: it must not have been looked for at all.
    #[rstest::rstest]
    #[case::traversal("../../../etc")]
    #[case::absolute("/etc")]
    #[case::separator("0000:65:00.0/..")]
    fn open_refuses_an_address_that_is_not_one(#[case] bdf: &str) {
        let err = PciDev::open_in(&Sysfs::default(), bdf).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{err}");
    }
}
