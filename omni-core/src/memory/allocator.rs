//! # memory::allocator
//!
//! Distributed Unified Virtual Address Space (DUVAS) allocator.
//!
//! Provides a unified 64-bit virtual memory address space across CPU system RAM,
//! discrete GPU VRAM (NVIDIA/AMD), Apple Silicon unified memory (UMA), and remote P2P nodes.
//! Tracks memory residence, coordinates physical allocations, and raises alerts under pressure.

use anyhow::{bail, Result};
use std::collections::BTreeMap;
use std::sync::Mutex;
use tracing::{debug, info, warn};

/// Represents the physical residence domain of a virtual memory block.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryDomain {
    /// Host System CPU memory (RAM)
    Host,
    /// Local discrete GPU memory (VRAM)
    LocalDevice,
    /// Memory residing on another cluster node over the P2P network
    RemotePeer,
    /// Evicted / swapped out to fast disk storage (SSD)
    Swap,
}

/// A contiguous chunk of virtual memory allocated in the DUVAS space.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Starting virtual address
    pub virtual_addr: u64,
    /// Size of allocation in bytes
    pub size_bytes: usize,
    /// Current physical residency domain
    pub domain: MemoryDomain,
    /// Flag indicating whether this block contains active/frequently-read weights (read-only)
    pub is_static: bool,
}

/// Distributed Unified Virtual Address Space (DUVAS) allocator implementation.
pub struct DuvasAllocator {
    /// Mutex protecting allocation maps
    state: Mutex<AllocatorState>,
    /// Capacity of the local device memory in bytes
    device_capacity_bytes: usize,
}

struct AllocatorState {
    /// Map of start_addr -> MemoryBlock
    allocations: BTreeMap<u64, MemoryBlock>,
    /// Total virtual memory allocated
    allocated_virtual_bytes: usize,
    /// Total memory physically resident on the local device
    allocated_device_bytes: usize,
    /// Counter to generate unique virtual addresses (starting at a standard base e.g., 0x10_0000_0000)
    next_addr: u64,
}

impl DuvasAllocator {
    /// Creates a new DUVAS allocator with a specified physical device capacity.
    pub fn new(device_capacity_bytes: usize) -> Self {
        Self {
            state: Mutex::new(AllocatorState {
                allocations: BTreeMap::new(),
                allocated_virtual_bytes: 0,
                allocated_device_bytes: 0,
                next_addr: 0x10_0000_0000u64, // 64 GB boundary offset
            }),
            device_capacity_bytes,
        }
    }

    /// Allocates a virtual memory block of the given size.
    ///
    /// The block is initially allocated in the local device domain. If physical memory pressure
    /// is high, the allocator allows overallocation of virtual space, setting up for paging.
    pub fn alloc(&self, size_bytes: usize, is_static: bool) -> Result<u64> {
        let mut state = self.state.lock().unwrap();

        // Round up to standard 4KB page alignment
        let aligned_size = (size_bytes + 4095) & !4095;
        let addr = state.next_addr;
        state.next_addr += aligned_size as u64;

        // Check if allocating this block directly on device exceeds capacity
        let domain = if state.allocated_device_bytes + aligned_size <= self.device_capacity_bytes {
            state.allocated_device_bytes += aligned_size;
            MemoryDomain::LocalDevice
        } else {
            // Allocate virtually, but residency defaults to Host RAM (needs sliding pager)
            debug!(
                "DuvasAllocator: VRAM pressure. Allocating {} bytes to Host RAM",
                aligned_size
            );
            MemoryDomain::Host
        };

        let block = MemoryBlock {
            virtual_addr: addr,
            size_bytes: aligned_size,
            domain,
            is_static,
        };

        state.allocations.insert(addr, block);
        state.allocated_virtual_bytes += aligned_size;

        debug!(
            "DuvasAllocator: alloc virtual_addr=0x{:x}, size={} bytes, domain={:?}",
            addr, aligned_size, domain
        );

        Ok(addr)
    }

    /// Frees an allocated virtual memory block.
    pub fn free(&self, virtual_addr: u64) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        
        if let Some(block) = state.allocations.remove(&virtual_addr) {
            state.allocated_virtual_bytes -= block.size_bytes;
            if block.domain == MemoryDomain::LocalDevice {
                state.allocated_device_bytes -= block.size_bytes;
            }
            debug!(
                "DuvasAllocator: freed virtual_addr=0x{:x}, size={} bytes",
                virtual_addr, block.size_bytes
            );
            Ok(())
        } else {
            bail!("DuvasAllocator: invalid free at address 0x{:x}", virtual_addr)
        }
    }

    /// Migrates a block of memory to a different physical domain.
    pub fn migrate(&self, virtual_addr: u64, target_domain: MemoryDomain) -> Result<()> {
        let mut state = self.state.lock().unwrap();

        let (prev_domain, size_bytes) = if let Some(block) = state.allocations.get(&virtual_addr) {
            (block.domain, block.size_bytes)
        } else {
            bail!("DuvasAllocator: cannot migrate unallocated address 0x{:x}", virtual_addr)
        };

        if prev_domain == target_domain {
            return Ok(());
        }

        // Adjust physical allocation counters
        if prev_domain == MemoryDomain::LocalDevice {
            state.allocated_device_bytes -= size_bytes;
        }
        if target_domain == MemoryDomain::LocalDevice {
            if state.allocated_device_bytes + size_bytes > self.device_capacity_bytes {
                warn!("DuvasAllocator: device memory overcommitted during migration of address 0x{:x}", virtual_addr);
            }
            state.allocated_device_bytes += size_bytes;
        }

        // Apply new domain
        if let Some(block) = state.allocations.get_mut(&virtual_addr) {
            block.domain = target_domain;
        }

        debug!(
            "DuvasAllocator: migrated virtual_addr=0x{:x} ({:?} -> {:?})",
            virtual_addr, prev_domain, target_domain
        );
        Ok(())
    }

    /// Returns current utilization metrics [allocated_device, capacity, allocated_virtual]
    pub fn get_metrics(&self) -> (usize, usize, usize) {
        let state = self.state.lock().unwrap();
        (
            state.allocated_device_bytes,
            self.device_capacity_bytes,
            state.allocated_virtual_bytes,
        )
    }

    /// Returns the physical memory domain associated with a virtual address, if it exists.
    pub fn get_domain(&self, virtual_addr: u64) -> Option<MemoryDomain> {
        let state = self.state.lock().unwrap();
        state.allocations.get(&virtual_addr).map(|b| b.domain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_free() {
        let allocator = DuvasAllocator::new(1024 * 1024); // 1 MB
        let addr1 = allocator.alloc(100 * 1024, false).unwrap();
        let addr2 = allocator.alloc(200 * 1024, false).unwrap();

        assert!(addr2 > addr1);
        assert_eq!(allocator.get_domain(addr1), Some(MemoryDomain::LocalDevice));

        let (used, cap, virt) = allocator.get_metrics();
        assert_eq!(used, 300 * 1024);
        assert_eq!(cap, 1024 * 1024);
        assert_eq!(virt, 300 * 1024);

        allocator.free(addr1).unwrap();
        let (used, _, _) = allocator.get_metrics();
        assert_eq!(used, 200 * 1024);
    }

    #[test]
    fn test_overallocation_fallback() {
        let allocator = DuvasAllocator::new(100 * 1024); // tiny 100 KB budget
        let addr1 = allocator.alloc(60 * 1024, false).unwrap(); // fits on local device
        let addr2 = allocator.alloc(60 * 1024, false).unwrap(); // overallocated, goes to Host RAM

        assert_eq!(allocator.get_domain(addr1), Some(MemoryDomain::LocalDevice));
        assert_eq!(allocator.get_domain(addr2), Some(MemoryDomain::Host));
    }

    #[test]
    fn test_migration() {
        let allocator = DuvasAllocator::new(100 * 1024);
        let addr = allocator.alloc(40 * 1024, false).unwrap();
        assert_eq!(allocator.get_domain(addr), Some(MemoryDomain::LocalDevice));

        allocator.migrate(addr, MemoryDomain::Host).unwrap();
        assert_eq!(allocator.get_domain(addr), Some(MemoryDomain::Host));
        assert_eq!(allocator.get_metrics().0, 0); // 0 device memory used
    }
}
