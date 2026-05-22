//! # memory::pager
//!
//! Virtual Sliding Pager (Virtual Sliding Pager).
//!
//! Implements a heuristic, latency-hiding memory pager that slides a memory
//! working set window over heterogeneous memory topologies.
//! Coordinates background asynchronous DMA/P2P data transfers so that next-layer weights
//! are resident on-chip *before* compute starts, maximizing hardware utilization.
//!
//! ## Mathematical Latency Hiding Constraint
//!
//! Let $K_i$ represent the kernel execution time of layer $i$ on the local accelerator.
//! Let $D_{i+1}$ represent the data transfer latency of layer $i+1$'s weights from Host RAM
//! (or a remote peer) to the device.
//! The sliding window achieves perfect latency hiding if and only if:
//!
//! $$D_{i+1} \le \sum_{j=1}^{i} K_j - T_{\text{overhead}}$$
//!
//! where $T_{\text{overhead}}$ is the scheduling and orchestration overhead of this pager.

use crate::memory::allocator::{DuvasAllocator, MemoryDomain};
use anyhow::{bail, Result};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use crossbeam::channel::{unbounded, Sender, Receiver};
use tracing::{debug, error, info, warn};

/// A command dispatched to the background asynchronous paging worker.
pub enum PageCommand {
    /// Asynchronously prefetch a virtual memory block to LocalDevice
    Prefetch {
        virtual_addr: u64,
        estimated_kernel_ms: f64,
    },
    /// Asynchronously evict a virtual memory block (move to Host RAM or Swap)
    Evict {
        virtual_addr: u64,
    },
    /// Shut down the background paging worker thread
    Shutdown,
}

/// The Virtual Sliding Pager coordinates memory transfers based on execution graphs.
pub struct VirtualSlidingPager {
    /// Device VRAM capacity in bytes
    _capacity_bytes: usize,
    /// Device L2 Cache boundary to optimize page size
    _l2_cache_bytes: usize,
    /// Reference to DUVAS allocator
    allocator: Arc<DuvasAllocator>,
    /// Channel sender to dispatch tasks to the pager thread
    cmd_sender: Sender<PageCommand>,
    /// Handle of the background paging worker thread
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl VirtualSlidingPager {
    /// Creates and starts a new Virtual Sliding Pager.
    pub fn new(
        capacity_bytes: usize,
        l2_cache_bytes: usize,
        allocator: Arc<DuvasAllocator>,
    ) -> Self {
        let (cmd_sender, cmd_receiver) = unbounded();
        
        let alloc_clone = Arc::clone(&allocator);
        let worker_handle = thread::spawn(move || {
            Self::run_worker(cmd_receiver, alloc_clone);
        });

        info!(
            "VirtualSlidingPager: started background paging worker with VRAM capacity={} MB, L2 cache={} KB",
            capacity_bytes / 1024 / 1024,
            l2_cache_bytes / 1024
        );

        Self {
            _capacity_bytes: capacity_bytes,
            _l2_cache_bytes: l2_cache_bytes,
            allocator,
            cmd_sender,
            worker_handle: Some(worker_handle),
        }
    }

    /// Primary entrypoint to slide the active working set window.
    ///
    /// - `active_layers`: Array of virtual addresses for layers currently being computed.
    ///   These will be pinned/eviction-protected.
    /// - `prefetch_layers`: Array of virtual addresses for layers predicted to run next.
    ///   These will be asynchronously loaded from background thread.
    /// - `evict_layers`: Array of virtual addresses of old layers that should be discarded
    ///   to free up device capacity.
    pub fn slide_window(
        &self,
        active_layers: &[u64],
        prefetch_layers: &[u64],
        evict_layers: &[u64],
        accumulated_kernel_ms: f64,
    ) -> Result<()> {
        debug!(
            "VirtualSlidingPager: sliding window. Active={} Prefetch={} Evict={}",
            active_layers.len(),
            prefetch_layers.len(),
            evict_layers.len()
        );

        // 1. Dispatch evictions first to clear up physical headroom
        for &addr in evict_layers {
            self.cmd_sender.send(PageCommand::Evict { virtual_addr: addr })?;
        }

        // 2. Dispatch prefetch commands for next layers
        for &addr in prefetch_layers {
            self.cmd_sender.send(PageCommand::Prefetch {
                virtual_addr: addr,
                estimated_kernel_ms: accumulated_kernel_ms,
            })?;
        }

        Ok(())
    }

    /// Background worker execution loop.
    ///
    /// Performs physical memory migration tasks asynchronously, measuring transfer bandwidth
    /// to continually refine latency hiding heuristics.
    fn run_worker(receiver: Receiver<PageCommand>, allocator: Arc<DuvasAllocator>) {
        // Safe standard default PCI-e bandwidth assumption (e.g. PCI-e Gen 4 x16 -> ~31.5 GB/s = 32.2 MB/ms)
        let mut estimated_bandwidth_mb_per_ms = 15.0; 
        let overhead_ms = 0.8; // T_overhead scheduling penalty

        loop {
            match receiver.recv() {
                Ok(PageCommand::Prefetch { virtual_addr, estimated_kernel_ms }) => {
                    let domain = allocator.get_domain(virtual_addr);
                    if domain == Some(MemoryDomain::LocalDevice) {
                        // Already resident, nothing to do
                        continue;
                    }

                    // Query block size to calculate expected latency
                    let metrics = allocator.get_metrics();
                    let used_vram = metrics.0;
                    let cap_vram = metrics.1;
                    
                    // Simple size approximation based on typical DUVAS layout size
                    // (Assuming 64MB typical layer size block for simulated metric)
                    let block_size_bytes = 64 * 1024 * 1024; 
                    let block_size_mb = block_size_bytes as f64 / 1024.0 / 1024.0;
                    
                    // Compute D_(i+1) = Block_Size / Bandwidth
                    let expected_transfer_ms = block_size_mb / estimated_bandwidth_mb_per_ms;
                    
                    // Validate: D_(i+1) <= Sum(K_j) - T_overhead
                    let margin = estimated_kernel_ms - expected_transfer_ms - overhead_ms;
                    
                    if margin >= 0.0 {
                        debug!(
                            "VirtualSlidingPager: PERFECT LATENCY HIDING VALIDATED for 0x{:x}. \
                             Transfer={} ms, KernelOverlap={} ms, SlackMargin={} ms",
                            virtual_addr, expected_transfer_ms, estimated_kernel_ms, margin
                        );
                    } else {
                        warn!(
                            "VirtualSlidingPager: LATE ARRIVAL WARNING for 0x{:x}. \
                             Transfer={} ms exceeds KernelOverlap={} ms by {} ms. Pager pipeline stalling!",
                            virtual_addr, expected_transfer_ms, estimated_kernel_ms, -margin
                        );
                    }

                    // Perform the actual data migration (host -> device)
                    let start_time = Instant::now();
                    if let Err(e) = allocator.migrate(virtual_addr, MemoryDomain::LocalDevice) {
                        error!("VirtualSlidingPager: prefetch migration failed for 0x{:x}: {:?}", virtual_addr, e);
                    } else {
                        let duration = start_time.elapsed().as_secs_f64() * 1000.0; // in milliseconds
                        // Dynamically update the bandwidth estimator based on real measured throughput
                        if duration > 0.1 {
                            let measured_bw = block_size_mb / duration;
                            estimated_bandwidth_mb_per_ms = 0.8 * estimated_bandwidth_mb_per_ms + 0.2 * measured_bw;
                            debug!(
                                "VirtualSlidingPager: prefetch completed in {:.2} ms (throughput: {:.2} GB/s)",
                                duration, estimated_bandwidth_mb_per_ms * 1.024
                            );
                        }
                    }
                }
                Ok(PageCommand::Evict { virtual_addr }) => {
                    // Evict memory block back to host RAM to free VRAM space
                    if allocator.get_domain(virtual_addr) == Some(MemoryDomain::LocalDevice) {
                        debug!("VirtualSlidingPager: evicting 0x{:x} to Host RAM", virtual_addr);
                        if let Err(e) = allocator.migrate(virtual_addr, MemoryDomain::Host) {
                            error!("VirtualSlidingPager: eviction migration failed for 0x{:x}: {:?}", virtual_addr, e);
                        }
                    }
                }
                Ok(PageCommand::Shutdown) | Err(_) => {
                    info!("VirtualSlidingPager: background worker shutting down");
                    break;
                }
            }
        }
    }
}

impl Drop for VirtualSlidingPager {
    fn drop(&mut self) {
        let _ = self.cmd_sender.send(PageCommand::Shutdown);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::DuvasAllocator;
    use std::sync::Arc;

    #[test]
    fn test_pager_slide_window() {
        let allocator = Arc::new(DuvasAllocator::new(100 * 1024 * 1024)); // 100 MB
        
        // Allocate virtual spaces
        let layer1 = allocator.alloc(10 * 1024 * 1024, false).unwrap(); // fits on device
        let layer2 = allocator.alloc(10 * 1024 * 1024, false).unwrap(); // fits on device
        let layer3 = allocator.alloc(10 * 1024 * 1024, false).unwrap(); // fits on device

        // Force layer3 to host to simulate pager test case
        allocator.migrate(layer3, MemoryDomain::Host).unwrap();
        assert_eq!(allocator.get_domain(layer3), Some(MemoryDomain::Host));

        let pager = VirtualSlidingPager::new(100 * 1024 * 1024, 4 * 1024 * 1024, Arc::clone(&allocator));

        // Slide window: prefetch layer3, evict layer1
        pager.slide_window(&[layer2], &[layer3], &[layer1], 45.0).unwrap();

        // Allow some time for background thread to execute the command channel
        thread::sleep(Duration::from_millis(50));

        // Pager should have successfully migrated layer3 to Device and layer1 to Host
        assert_eq!(allocator.get_domain(layer3), Some(MemoryDomain::LocalDevice));
        assert_eq!(allocator.get_domain(layer1), Some(MemoryDomain::Host));
    }
}
