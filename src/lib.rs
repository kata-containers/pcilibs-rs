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
