//! # memory
//!
//! Memory management subsystem for OmniCompute.
//!
//! Provides:
//! - **DUVAS** (`allocator`) — Distributed Unified Virtual Address Space allocator
//!   managing memory across multiple heterogeneous device domains.
//! - **Virtual Sliding Pager** (`pager`) — Handles asynchronous memory prefetching
//!   and page replacement to run model layers larger than the physical device memory.

pub mod allocator;
pub mod pager;

pub use allocator::DuvasAllocator;
pub use pager::VirtualSlidingPager;
