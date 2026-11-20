// Copyright (c) Ant Group
// Copyright (c) NVIDIA CORPORATION
//
// SPDX-License-Identifier: Apache-2.0
//

#![cfg_attr(not(feature = "std"), no_std)]

pub mod catalog;
pub mod gpu;
pub mod platform;

#[cfg(feature = "cc")]
pub mod cc;
#[cfg(feature = "std")]
mod iommufd;
#[cfg(feature = "std")]
pub mod nvlink;
#[cfg(feature = "std")]
mod pci_dev;
#[cfg(feature = "std")]
mod pci_ids;
#[cfg(feature = "std")]
mod pci_manager;
#[cfg(feature = "std")]
mod sysfs;
#[cfg(all(feature = "std", any(test, feature = "testfs")))]
pub mod testfs;
#[cfg(feature = "std")]
pub mod vfio;

#[cfg(feature = "std")]
mod linux;
#[cfg(feature = "std")]
pub use linux::*;
#[cfg(feature = "std")]
pub(crate) use linux::{context, failed};

pub fn parse_hex_id(text: &str) -> Option<u32> {
    let text = text.trim();
    let text = text.strip_prefix("0x").unwrap_or(text);
    u32::from_str_radix(text, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::parse_hex_id;

    #[test]
    fn hex_ids_parse_with_and_without_prefix_and_whitespace() {
        assert_eq!(parse_hex_id("0x10de"), Some(0x10de));
        assert_eq!(parse_hex_id("10de"), Some(0x10de));
        assert_eq!(parse_hex_id("0x10DE"), Some(0x10de));
        assert_eq!(parse_hex_id("0x030200\n"), Some(0x030200));
        assert_eq!(parse_hex_id(" 0xffffffff "), Some(u32::MAX));
    }

    #[test]
    fn junk_is_not_an_id() {
        for text in ["", " ", "0x", "zz", "0xzz", "-0x1", "0x100000000"] {
            assert_eq!(parse_hex_id(text), None, "{text:?}");
        }
    }
}
