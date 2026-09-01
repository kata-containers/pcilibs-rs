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

//! FSP PRC-knob RPC — the in-band interface that owns CC configuration.
//!
//! The FSP (firmware security processor) stores CC settings as persistent
//! "PRC knobs".  Knobs are read and written with single-packet MCTP
//! messages (NVIDIA vendor-defined type) delivered through one of two
//! BAR0 mailboxes:
//!
//! * Hopper — the FSP falcon's EMEM shared memory, channel 2
//! * Blackwell — the FSP MNOC mailbox, port 0
//!
//! Register offsets, framing and command values mirror NVIDIA's
//! gpu-admin-tools (MIT), the reference implementation for out-of-driver
//! CC provisioning.

use crate::poll;
use crate::PciDev;
use anyhow::{bail, ensure, Result};
use std::time::Duration;

/// PRC knob ids (gpu-admin-tools `gpu/prc.py`).
pub const KNOB_2: u32 = 2; // Hopper-only, cleared before enabling CC
pub const KNOB_4: u32 = 4; // Hopper-only, cleared before enabling CC
pub const KNOB_CCD: u32 = 6; // CC devtools mode
pub const KNOB_CCM: u32 = 8; // CC mode
pub const KNOB_BAR0_DECOUPLER: u32 = 10; // Hopper BAR0 filter
pub const KNOB_34: u32 = 34; // Hopper-only, cleared before enabling CC
pub const KNOB_PPCIE: u32 = 45; // protected PCIe, mutually exclusive with CC

/// NVDM (NVIDIA data model) message types.
const NVDM_PRC: u32 = 0x13;
const NVDM_RESPONSE: u32 = 0x15;

/// A non-zero FSP completion code.
#[derive(Debug)]
pub struct FspError {
    pub code: u32,
}

impl std::fmt::Display for FspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FSP RPC failed with completion code {:#x}", self.code)
    }
}

impl std::error::Error for FspError {}

/// Firmware predating a knob answers knob reads with this code.
pub fn is_invalid_knob(err: &anyhow::Error) -> bool {
    err.downcast_ref::<FspError>()
        .is_some_and(|fsp| fsp.code == 0x1e3)
}

/// MCTP transport header: version 0, endpoint ids 0, tag 0, seq 0,
/// som=1 (bit 31), eom=1 (bit 30) — every RPC here fits one packet.
const MCTP_HEADER: u32 = 0xc000_0000;

/// MCTP message header dword: type 0x7e (vendor via PCI), vendor 0x10de,
/// NVDM type in the top byte.
fn mctp_msg_header(nvdm_type: u32) -> u32 {
    0x0010_de7e | nvdm_type << 24
}

fn mctp_packet(nvdm_type: u32, payload: &[u32]) -> Vec<u32> {
    let mut packet = Vec::with_capacity(2 + payload.len());
    packet.push(MCTP_HEADER);
    packet.push(mctp_msg_header(nvdm_type));
    packet.extend_from_slice(payload);
    packet
}

/// One RPC channel to the FSP.  Variant selection is per GPU generation.
pub enum FspRpc<'a> {
    Emem(&'a PciDev),
    Mnoc(&'a PciDev),
}

impl<'a> FspRpc<'a> {
    /// Hopper: EMEM channel 2.  Resets stale queue state left behind by a
    /// crashed client before first use.
    pub fn emem(dev: &'a PciDev) -> Result<Self> {
        emem_reset_queues(dev)?;
        Ok(Self::Emem(dev))
    }

    /// Blackwell: MNOC port 0.
    pub fn mnoc(dev: &'a PciDev) -> Self {
        Self::Mnoc(dev)
    }

    pub fn knob_read(&self, knob: u32) -> Result<u16> {
        let data = self.prc_cmd(&[0xc | 0x2 << 8 | knob << 16])?;
        ensure!(data.len() == 1, "knob {knob} read: bad response {data:x?}");
        // The value is 16 bits; the upper half is not zero-initialized.
        Ok(data[0] as u16)
    }

    pub fn knob_write(&self, knob: u32, value: u16) -> Result<()> {
        let data = self.prc_cmd(&[0xd | 0x2 << 8 | knob << 16, value as u32])?;
        ensure!(data.is_empty(), "knob {knob} write: bad response {data:x?}");
        Ok(())
    }

    pub fn knob_check_and_write(&self, knob: u32, value: u16) -> Result<()> {
        if self.knob_read(knob)? != value {
            self.knob_write(knob, value)?;
        }
        Ok(())
    }

    fn prc_cmd(&self, payload: &[u32]) -> Result<Vec<u32>> {
        let packet = mctp_packet(NVDM_PRC, payload);
        let response = match self {
            Self::Emem(dev) => {
                emem_send(dev, &packet)?;
                emem_receive(dev)?
            }
            Self::Mnoc(dev) => {
                mnoc_send(dev, &packet)?;
                mnoc_receive(dev)?
            }
        };
        // [mctp hdr, msg hdr, seq, request nvdm type, completion, payload...]
        ensure!(response.len() >= 5, "FSP response too short: {response:x?}");
        ensure!(
            response[1] >> 24 == NVDM_RESPONSE,
            "FSP response has wrong nvdm type: {response:x?}"
        );
        ensure!(
            response[3] == NVDM_PRC,
            "FSP response for wrong command: {response:x?}"
        );
        if response[4] != 0 {
            return Err(FspError { code: response[4] }.into());
        }
        Ok(response[5..].to_vec())
    }
}

// --- Hopper: FSP falcon EMEM, channel 2 ------------------------------------
//
// A command is written into the channel's EMEM window and announced by the
// queue head/tail registers; the response arrives in the same window,
// announced by the message queue registers.

const EMEM_CHANNEL: u32 = 2;
const EMEM_BASE: u32 = EMEM_CHANNEL * 1024; // byte offset inside EMEM
const EMEMC: u32 = 0x8f2ac0 + EMEM_CHANNEL * 8; // port control: offset + autoinc
const EMEMD: u32 = EMEMC + 4; // port data window
const QUEUE_HEAD: u32 = 0x8f2c00 + EMEM_CHANNEL * 8; // writing head is the doorbell
const QUEUE_TAIL: u32 = QUEUE_HEAD + 4;
const MSGQ_HEAD: u32 = 0x8f2c80 + EMEM_CHANNEL * 8;
const MSGQ_TAIL: u32 = MSGQ_HEAD + 4;
const EMEMC_AINCW: u32 = 1 << 24;
const EMEMC_AINCR: u32 = 1 << 25;

fn emem_reset_queues(dev: &PciDev) -> Result<()> {
    let empty = |head, tail| dev.read32(head) == dev.read32(tail);
    if empty(QUEUE_HEAD, QUEUE_TAIL) && empty(MSGQ_HEAD, MSGQ_TAIL) {
        return Ok(());
    }
    // Give an in-flight command a chance to produce its response, then
    // point both queues back at this channel's EMEM base.
    let _ = poll("stale FSP response", Duration::from_secs(5), || {
        !empty(MSGQ_HEAD, MSGQ_TAIL)
    });
    dev.write32(QUEUE_TAIL, EMEM_BASE);
    dev.write32(QUEUE_HEAD, EMEM_BASE);
    dev.write32(MSGQ_TAIL, EMEM_BASE);
    dev.write32(MSGQ_HEAD, EMEM_BASE);
    Ok(())
}

fn emem_send(dev: &PciDev, data: &[u32]) -> Result<()> {
    ensure!(data.len() * 4 <= 1024, "FSP command exceeds EMEM channel");
    poll("FSP command queue empty", Duration::from_secs(5), || {
        dev.read32(QUEUE_HEAD) == dev.read32(QUEUE_TAIL)
    })?;
    dev.write32(EMEMC, EMEM_BASE | EMEMC_AINCW | EMEMC_AINCR);
    for &d in data {
        dev.write32(EMEMD, d);
    }
    dev.write32(QUEUE_TAIL, EMEM_BASE + (data.len() as u32 - 1) * 4);
    dev.write32(QUEUE_HEAD, EMEM_BASE);
    Ok(())
}

fn emem_receive(dev: &PciDev) -> Result<Vec<u32>> {
    poll("FSP response", Duration::from_secs(5), || {
        dev.read32(MSGQ_HEAD) != dev.read32(MSGQ_TAIL)
    })?;
    let head = dev.read32(MSGQ_HEAD);
    let tail = dev.read32(MSGQ_TAIL);
    let dwords = tail.wrapping_sub(head) / 4 + 1;
    ensure!(dwords <= 256, "FSP response exceeds EMEM channel");
    dev.write32(EMEMC, EMEM_BASE | EMEMC_AINCW | EMEMC_AINCR);
    let data = (0..dwords).map(|_| dev.read32(EMEMD)).collect();
    dev.write32(MSGQ_TAIL, head); // ack
    Ok(data)
}

// --- Blackwell: FSP MNOC mailbox, port 0 ------------------------------------
//
// Two mailbox register pairs: we push commands through the "receive"
// mailbox (info + data) and pull responses from the "send" mailbox.

const MNOC_INFO_SEND: u32 = 0x8f1e00 + 0x104;
const MNOC_RDATA_SEND: u32 = MNOC_INFO_SEND + 4;
const MNOC_INFO_RECV: u32 = 0x8f1e00 + 0x184;
const MNOC_WDATA_RECV: u32 = MNOC_INFO_RECV + 4;
const MNOC_SIZE_MASK: u32 = 0xfffff;
const MNOC_NEW_MSG: u32 = 1 << 20;
const MNOC_READY: u32 = 1 << 24;
const MNOC_ERROR: u32 = 1 << 25;
const MNOC_CREDITS: u32 = 1 << 26;

fn mnoc_send(dev: &PciDev, data: &[u32]) -> Result<()> {
    poll("FSP MNOC receive ready", Duration::from_secs(5), || {
        dev.read32(MNOC_INFO_RECV) & MNOC_READY != 0
    })?;
    dev.write32(MNOC_INFO_RECV, (data.len() as u32 * 4) | MNOC_NEW_MSG);
    for (i, &d) in data.iter().enumerate() {
        // Credits are granted in 64-byte units.
        if i % 16 == 0 {
            poll("FSP MNOC credits", Duration::from_secs(1), || {
                dev.read32(MNOC_INFO_RECV) & MNOC_CREDITS != 0
            })?;
        }
        dev.write32(MNOC_WDATA_RECV, d);
    }
    let info = dev.read32(MNOC_INFO_RECV);
    ensure!(info & MNOC_ERROR == 0, "FSP MNOC send error: {info:#x}");
    Ok(())
}

fn mnoc_receive(dev: &PciDev) -> Result<Vec<u32>> {
    if let Err(err) = poll("FSP MNOC response", Duration::from_secs(5), || {
        dev.read32(MNOC_INFO_SEND) & MNOC_READY != 0
    }) {
        let info = dev.read32(MNOC_INFO_SEND);
        if info & MNOC_ERROR != 0 {
            bail!("FSP MNOC receive error: {info:#x}");
        }
        return Err(err.into());
    }
    // An inaccessible BAR0 reads all-ones, which passes the poll above and
    // asks for a megabyte here.
    let bytes = dev.read32(MNOC_INFO_SEND) & MNOC_SIZE_MASK;
    ensure!(
        bytes <= 1024,
        "FSP MNOC response too large: {bytes:#x} bytes"
    );
    let data = (0..bytes / 4)
        .map(|_| dev.read32(MNOC_RDATA_SEND))
        .collect();
    let info = dev.read32(MNOC_INFO_SEND);
    ensure!(info & MNOC_ERROR == 0, "FSP MNOC receive error: {info:#x}");
    Ok(data)
}

/// A model of the FSP behind either mailbox, for tests across the crate.
/// It lives here rather than beside them so that it takes its offsets and
/// its framing from the constants above, and the two sides of the protocol
/// cannot drift apart.
#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use crate::pci_dev::Registers;
    use crate::PciDev;
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex};

    /// The PRC subcommand `knob_read` frames; anything else is a write.
    const PRC_READ: u32 = 0xc;
    /// What firmware predating a knob answers a read of it with.
    const INVALID_KNOB: u32 = 0x1e3;

    /// Every knob this crate touches: an absent one is firmware too old to
    /// have it, which is a thing a test asks for deliberately.
    const KNOBS: &[u32] = &[
        KNOB_2,
        KNOB_4,
        KNOB_CCD,
        KNOB_CCM,
        KNOB_BAR0_DECOUPLER,
        KNOB_34,
        KNOB_PPCIE,
    ];

    /// How the FSP should answer, so the checks above it have something to
    /// catch.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) enum Fault {
        /// A non-zero completion code.
        Completion(u32),
        /// Something that is not an NVDM response.
        NvdmType,
        /// An answer to a command that was never sent.
        OtherCommand,
        /// Less than a response header.
        Short,
        /// One payload dword too many: two for a read, one for a write.
        ExtraDword,
        /// More than the channel can carry.
        Oversized,
        /// No answer at all.
        Silent,
        /// The MNOC error bit, on the way in.
        SendError,
        /// The MNOC error bit instead of an answer.
        ReceiveError,
        /// The MNOC error bit, once the payload has been handed over.
        LateError,
    }

    /// EMEM for Hopper and the switches, MNOC for Blackwell.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Mailbox {
        Emem,
        Mnoc,
    }

    #[derive(Debug)]
    pub(crate) struct Fsp {
        mailbox: Mailbox,
        state: Mutex<State>,
    }

    #[derive(Debug)]
    struct State {
        /// Registers neither mailbox owns.
        regs: HashMap<u32, u32>,
        knobs: HashMap<u32, u16>,
        /// So a test can see a write that was not needed.
        writes: usize,
        fault: Option<Fault>,
        /// The knob `fault` applies to; all of them when None.
        faulty_knob: Option<u32>,
        /// The EMEM window by dword, carrying command then response.
        emem: Vec<u32>,
        emem_at: usize,
        queue_head: u32,
        queue_tail: u32,
        msgq_head: u32,
        msgq_tail: u32,
        expected: usize,
        command: Vec<u32>,
        response: VecDeque<u32>,
        info_send: u32,
        send_error: bool,
        late_error: bool,
    }

    impl Fsp {
        /// Hopper and the NVSwitches: EMEM channel 2.
        pub(crate) fn emem() -> Arc<Self> {
            Self::new(Mailbox::Emem)
        }

        /// Blackwell: the MNOC mailbox.
        pub(crate) fn mnoc() -> Arc<Self> {
            Self::new(Mailbox::Mnoc)
        }

        fn new(mailbox: Mailbox) -> Arc<Self> {
            Arc::new(Self {
                mailbox,
                state: Mutex::new(State {
                    regs: HashMap::new(),
                    knobs: KNOBS.iter().map(|knob| (*knob, 0)).collect(),
                    writes: 0,
                    fault: None,
                    faulty_knob: None,
                    emem: vec![0; 1024],
                    emem_at: 0,
                    queue_head: 0,
                    queue_tail: 0,
                    msgq_head: 0,
                    msgq_tail: 0,
                    expected: 0,
                    command: Vec::new(),
                    response: VecDeque::new(),
                    info_send: 0,
                    send_error: false,
                    late_error: false,
                }),
            })
        }

        /// So a test of what is common to both can be written once.
        pub(crate) fn rpc<'a>(&self, dev: &'a PciDev) -> Result<FspRpc<'a>> {
            match self.mailbox {
                Mailbox::Emem => FspRpc::emem(dev),
                Mailbox::Mnoc => Ok(FspRpc::mnoc(dev)),
            }
        }

        pub(crate) fn set_register(&self, offset: u32, value: u32) {
            self.state.lock().unwrap().regs.insert(offset, value);
        }

        pub(crate) fn set_knob(&self, knob: u32, value: u16) {
            self.state.lock().unwrap().knobs.insert(knob, value);
        }

        /// Firmware that does not have the knob at all.
        pub(crate) fn forget_knob(&self, knob: u32) {
            self.state.lock().unwrap().knobs.remove(&knob);
        }

        pub(crate) fn knob(&self, knob: u32) -> Option<u16> {
            self.state.lock().unwrap().knobs.get(&knob).copied()
        }

        pub(crate) fn writes(&self) -> usize {
            self.state.lock().unwrap().writes
        }

        pub(crate) fn fail(&self, fault: Fault) {
            let mut state = self.state.lock().unwrap();
            state.fault = Some(fault);
            state.faulty_knob = None;
        }

        /// Fault only this knob's command, so the ones before it still get
        /// through.
        pub(crate) fn fail_knob(&self, knob: u32, fault: Fault) {
            let mut state = self.state.lock().unwrap();
            state.fault = Some(fault);
            state.faulty_knob = Some(knob);
        }

        /// A response left in the queue by a client that died mid-RPC.
        pub(crate) fn leave_a_stale_response(&self) {
            self.state.lock().unwrap().msgq_tail = 4;
        }
    }

    impl Registers for Arc<Fsp> {
        fn read32(&self, offset: u32) -> u32 {
            let mut state = self.state.lock().unwrap();
            match (self.mailbox, offset) {
                (Mailbox::Emem, QUEUE_HEAD) => state.queue_head,
                (Mailbox::Emem, QUEUE_TAIL) => state.queue_tail,
                (Mailbox::Emem, MSGQ_HEAD) => state.msgq_head,
                (Mailbox::Emem, MSGQ_TAIL) => state.msgq_tail,
                (Mailbox::Emem, EMEMD) => {
                    state.emem_at += 1;
                    state.emem[state.emem_at - 1]
                }
                (Mailbox::Mnoc, MNOC_INFO_RECV) => {
                    let error = if state.send_error { MNOC_ERROR } else { 0 };
                    MNOC_READY | MNOC_CREDITS | error
                }
                (Mailbox::Mnoc, MNOC_INFO_SEND) => {
                    let drained = state.late_error && state.response.is_empty();
                    state.info_send | if drained { MNOC_ERROR } else { 0 }
                }
                (Mailbox::Mnoc, MNOC_RDATA_SEND) => state.response.pop_front().unwrap_or(0),
                _ => state.regs.get(&offset).copied().unwrap_or(0),
            }
        }

        fn write32(&self, offset: u32, value: u32) {
            let mut state = self.state.lock().unwrap();
            match (self.mailbox, offset) {
                (Mailbox::Emem, EMEMC) => state.emem_at = (value & 0x00ff_ffff) as usize / 4,
                (Mailbox::Emem, EMEMD) => {
                    let at = state.emem_at;
                    state.emem[at] = value;
                    state.emem_at += 1;
                }
                (Mailbox::Emem, QUEUE_TAIL) => state.queue_tail = value,
                (Mailbox::Emem, QUEUE_HEAD) => {
                    state.queue_head = value;
                    // Head meeting tail is the queue being reset, not a
                    // doorbell: no command here is one dword long.
                    if state.queue_head != state.queue_tail {
                        state.answer_over_emem();
                    }
                }
                (Mailbox::Emem, MSGQ_HEAD) => state.msgq_head = value,
                (Mailbox::Emem, MSGQ_TAIL) => state.msgq_tail = value,
                (Mailbox::Mnoc, MNOC_INFO_RECV) => {
                    state.expected = (value & MNOC_SIZE_MASK) as usize / 4;
                    state.command.clear();
                }
                (Mailbox::Mnoc, MNOC_WDATA_RECV) => {
                    state.command.push(value);
                    if state.command.len() == state.expected {
                        state.answer_over_mnoc();
                    }
                }
                _ => {
                    state.regs.insert(offset, value);
                }
            }
        }
    }

    impl State {
        fn fault(&self, command: &[u32]) -> Option<Fault> {
            match self.faulty_knob {
                Some(knob) if knob != command[2] >> 16 => None,
                _ => self.fault,
            }
        }

        fn answer_over_emem(&mut self) {
            let base = EMEM_BASE as usize / 4;
            let dwords = (self.queue_tail.wrapping_sub(self.queue_head) / 4 + 1) as usize;
            let command = self.emem[base..base + dwords].to_vec();
            self.queue_tail = self.queue_head; // consumed
            let fault = self.fault(&command);
            if fault == Some(Fault::Silent) {
                return;
            }
            if fault == Some(Fault::Oversized) {
                // The queue registers alone say how much there is to read.
                self.msgq_head = EMEM_BASE;
                self.msgq_tail = EMEM_BASE + 256 * 4;
                return;
            }
            let response = self.answer(&command, fault);
            self.emem[base..base + response.len()].copy_from_slice(&response);
            self.msgq_head = EMEM_BASE;
            self.msgq_tail = EMEM_BASE + (response.len() as u32 - 1) * 4;
        }

        fn answer_over_mnoc(&mut self) {
            let command = std::mem::take(&mut self.command);
            let fault = self.fault(&command);
            match fault {
                Some(Fault::Silent) => return,
                Some(Fault::SendError) => {
                    self.send_error = true;
                    return;
                }
                Some(Fault::ReceiveError) => {
                    self.info_send = MNOC_ERROR;
                    return;
                }
                Some(Fault::Oversized) => {
                    self.info_send = MNOC_READY | 2048;
                    return;
                }
                Some(Fault::LateError) => self.late_error = true,
                _ => {}
            }
            let response = self.answer(&command, fault);
            self.info_send = MNOC_READY | (response.len() as u32 * 4);
            self.response = response.into();
        }

        fn answer(&mut self, command: &[u32], fault: Option<Fault>) -> Vec<u32> {
            let payload = &command[2..];
            let knob = payload[0] >> 16;
            let mut completion = 0;
            let mut body = Vec::new();
            if payload[0] & 0xff == PRC_READ {
                match self.knobs.get(&knob) {
                    // Firmware leaves the dword's upper half as it found it.
                    Some(value) => body.push(0xdead_0000 | u32::from(*value)),
                    None => completion = INVALID_KNOB,
                }
            } else {
                self.knobs.insert(knob, payload[1] as u16);
                self.writes += 1;
            }

            match fault {
                Some(Fault::Completion(code)) => completion = code,
                Some(Fault::ExtraDword) => body.push(0),
                _ => {}
            }
            let nvdm = if fault == Some(Fault::NvdmType) {
                NVDM_PRC
            } else {
                NVDM_RESPONSE
            };
            let answered = if fault == Some(Fault::OtherCommand) {
                NVDM_RESPONSE
            } else {
                NVDM_PRC
            };
            let mut response = vec![MCTP_HEADER, mctp_msg_header(nvdm), 0, answered, completion];
            response.extend(body);
            if fault == Some(Fault::Short) {
                response.truncate(4);
            }
            response
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{Fault, Fsp};
    use super::*;
    use crate::PciDev;
    use rstest::rstest;
    use std::path::PathBuf;
    use std::sync::Arc;

    const BDF: &str = "0000:65:00.0";

    /// Nothing here resets, so the sysfs path is never read.
    fn device(fsp: &Arc<Fsp>) -> PciDev {
        PciDev::modelled(
            PathBuf::from("/nonexistent"),
            BDF,
            0x10de,
            0x2330,
            Box::new(fsp.clone()),
        )
    }

    #[test]
    fn knob_read_packet_matches_reference() {
        // gpu-admin-tools framing for a CCM knob read over PRC (nvdm 0x13).
        let payload = 0xc | 0x2 << 8 | KNOB_CCM << 16;
        assert_eq!(
            mctp_packet(NVDM_PRC, &[payload]),
            vec![0xc000_0000, 0x1310_de7e, 0x0008_020c]
        );
    }

    #[test]
    fn knob_write_packet_matches_reference() {
        let payload = 0xd | 0x2 << 8 | KNOB_CCD << 16;
        assert_eq!(
            mctp_packet(NVDM_PRC, &[payload, 0x1]),
            vec![0xc000_0000, 0x1310_de7e, 0x0006_020d, 0x1]
        );
    }

    /// The state and boot-complete reads in [`crate::cc`] stand on this.
    #[rstest]
    fn a_register_neither_mailbox_owns_reads_back_what_was_written() {
        let fsp = Fsp::emem();
        let dev = device(&fsp);

        dev.write32(0x200bc, 0xff);

        assert_eq!(dev.read32(0x200bc), 0xff);
    }

    #[test]
    fn an_fsp_error_names_its_completion_code() {
        assert_eq!(
            FspError { code: 0x1e3 }.to_string(),
            "FSP RPC failed with completion code 0x1e3"
        );
    }

    /// The framing above the two transports is the same, so a knob is asked
    /// of both.
    #[rstest]
    #[case::hopper_and_the_switches(Fsp::emem as fn() -> Arc<Fsp>)]
    #[case::blackwell(Fsp::mnoc)]
    fn a_knob_written_reads_back(#[case] new: fn() -> Arc<Fsp>) {
        let fsp = new();
        let dev = device(&fsp);
        let rpc = fsp.rpc(&dev).unwrap();

        rpc.knob_write(KNOB_CCM, 1).unwrap();

        assert_eq!(rpc.knob_read(KNOB_CCM).unwrap(), 1);
        assert_eq!(fsp.knob(KNOB_CCM), Some(1));
    }

    /// The dword's upper half is not zero-initialized, so a knob is its low
    /// 16 bits and nothing more.
    #[rstest]
    #[case::hopper_and_the_switches(Fsp::emem as fn() -> Arc<Fsp>)]
    #[case::blackwell(Fsp::mnoc)]
    fn a_knob_read_keeps_only_the_low_half(#[case] new: fn() -> Arc<Fsp>) {
        let fsp = new();
        fsp.set_knob(KNOB_CCM, 3);
        let dev = device(&fsp);

        assert_eq!(fsp.rpc(&dev).unwrap().knob_read(KNOB_CCM).unwrap(), 3);
    }

    /// These knobs live in flash, so one already at the value asked for is
    /// read and left alone.
    #[rstest]
    fn a_knob_already_at_the_value_asked_for_is_not_written() {
        let fsp = Fsp::emem();
        fsp.set_knob(KNOB_CCM, 1);
        let dev = device(&fsp);
        let rpc = FspRpc::emem(&dev).unwrap();

        rpc.knob_check_and_write(KNOB_CCM, 1).unwrap();
        assert_eq!(fsp.writes(), 0);

        rpc.knob_check_and_write(KNOB_CCM, 0).unwrap();
        assert_eq!((fsp.writes(), fsp.knob(KNOB_CCM)), (1, Some(0)));
    }

    /// Firmware that predates a knob is not a failure to pass on as one:
    /// the caller carries on without it.
    #[rstest]
    fn a_knob_the_firmware_does_not_have_reads_as_invalid() {
        let fsp = Fsp::emem();
        fsp.forget_knob(KNOB_PPCIE);
        let dev = device(&fsp);

        let err = FspRpc::emem(&dev)
            .unwrap()
            .knob_read(KNOB_PPCIE)
            .unwrap_err();

        assert!(is_invalid_knob(&err), "{err}");
    }

    #[rstest]
    #[case::another_completion_code(anyhow::Error::from(FspError { code: 0x5 }))]
    #[case::not_an_fsp_error(anyhow::anyhow!("BAR0 not accessible"))]
    fn only_that_one_code_means_an_invalid_knob(#[case] err: anyhow::Error) {
        assert!(!is_invalid_knob(&err), "{err}");
    }

    #[rstest]
    fn a_non_zero_completion_code_is_the_error() {
        let fsp = Fsp::emem();
        fsp.fail(Fault::Completion(0x2a));
        let dev = device(&fsp);

        let err = FspRpc::emem(&dev).unwrap().knob_read(KNOB_CCM).unwrap_err();

        assert_eq!(err.downcast_ref::<FspError>().unwrap().code, 0x2a);
    }

    #[rstest]
    #[case::not_a_response(Fault::NvdmType, "wrong nvdm type")]
    #[case::a_response_to_something_else(Fault::OtherCommand, "for wrong command")]
    #[case::no_header(Fault::Short, "response too short")]
    #[case::a_dword_too_many(Fault::ExtraDword, "read: bad response")]
    fn a_malformed_response_to_a_read_is_refused(#[case] fault: Fault, #[case] expected: &str) {
        let fsp = Fsp::emem();
        fsp.fail(fault);
        let dev = device(&fsp);

        let err = FspRpc::emem(&dev)
            .unwrap()
            .knob_read(KNOB_CCM)
            .unwrap_err()
            .to_string();

        assert!(err.contains(expected), "{err}");
    }

    #[rstest]
    fn a_write_answered_with_a_payload_is_refused() {
        let fsp = Fsp::emem();
        fsp.fail(Fault::ExtraDword);
        let dev = device(&fsp);

        let err = FspRpc::emem(&dev)
            .unwrap()
            .knob_write(KNOB_CCM, 1)
            .unwrap_err()
            .to_string();

        assert!(err.contains("write: bad response"), "{err}");
    }

    /// A client that died mid-RPC leaves the queues pointing into the window.
    #[rstest]
    fn a_stale_emem_queue_is_reset_before_first_use() {
        let fsp = Fsp::emem();
        fsp.leave_a_stale_response();
        let dev = device(&fsp);

        let rpc = FspRpc::emem(&dev).unwrap();

        assert_eq!(rpc.knob_read(KNOB_CCM).unwrap(), 0);
    }

    /// The channel is 1 KiB and nothing above builds a command that big;
    /// this is the floor under that.
    #[rstest]
    fn a_command_larger_than_the_emem_channel_is_refused() {
        let fsp = Fsp::emem();
        let dev = device(&fsp);

        let err = emem_send(&dev, &vec![0; 257]).unwrap_err().to_string();

        assert!(err.contains("command exceeds EMEM channel"), "{err}");
    }

    #[rstest]
    fn an_emem_response_larger_than_the_channel_is_refused() {
        let fsp = Fsp::emem();
        fsp.fail(Fault::Oversized);
        let dev = device(&fsp);

        let err = FspRpc::emem(&dev)
            .unwrap()
            .knob_read(KNOB_CCM)
            .unwrap_err()
            .to_string();

        assert!(err.contains("response exceeds EMEM channel"), "{err}");
    }

    #[rstest]
    fn an_mnoc_response_larger_than_the_mailbox_is_refused() {
        let fsp = Fsp::mnoc();
        fsp.fail(Fault::Oversized);
        let dev = device(&fsp);

        let err = FspRpc::mnoc(&dev)
            .knob_read(KNOB_CCM)
            .unwrap_err()
            .to_string();

        assert!(err.contains("MNOC response too large"), "{err}");
    }

    #[rstest]
    fn an_mnoc_error_taking_the_command_is_reported() {
        let fsp = Fsp::mnoc();
        fsp.fail(Fault::SendError);
        let dev = device(&fsp);

        let err = FspRpc::mnoc(&dev)
            .knob_read(KNOB_CCM)
            .unwrap_err()
            .to_string();

        assert!(err.contains("MNOC send error"), "{err}");
    }

    /// The mailbox can fail after handing the payload over, which is why its
    /// status is read a second time.
    #[rstest]
    fn an_mnoc_error_after_the_payload_is_reported() {
        let fsp = Fsp::mnoc();
        fsp.fail(Fault::LateError);
        let dev = device(&fsp);

        let err = FspRpc::mnoc(&dev)
            .knob_read(KNOB_CCM)
            .unwrap_err()
            .to_string();

        assert!(err.contains("MNOC receive error"), "{err}");
    }

    /// Five seconds each: a mailbox that says nothing is only knowable by
    /// the timeout.
    #[rstest]
    #[case::error_bit(Fault::ReceiveError, "MNOC receive error")]
    #[case::nothing_at_all(Fault::Silent, "timed out waiting for FSP MNOC response")]
    fn an_mnoc_mailbox_that_does_not_answer_is_reported(
        #[case] fault: Fault,
        #[case] expected: &str,
    ) {
        let fsp = Fsp::mnoc();
        fsp.fail(fault);
        let dev = device(&fsp);

        let err = FspRpc::mnoc(&dev)
            .knob_read(KNOB_CCM)
            .unwrap_err()
            .to_string();

        assert!(err.contains(expected), "{err}");
    }

    #[rstest]
    fn an_emem_mailbox_that_does_not_answer_is_reported() {
        let fsp = Fsp::emem();
        fsp.fail(Fault::Silent);
        let dev = device(&fsp);

        let err = FspRpc::emem(&dev)
            .unwrap()
            .knob_read(KNOB_CCM)
            .unwrap_err()
            .to_string();

        assert!(err.contains("timed out waiting for FSP response"), "{err}");
    }
}
