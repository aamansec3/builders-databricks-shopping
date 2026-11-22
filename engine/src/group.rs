//! Partitioning records as they are produced, and grouping equal strings
//! exactly within a partition.
//!
//! Each stage hands records straight to the partition the next stage needs:
//! every thread appends to its own set of partition lists (Buckets), so no two
//! threads ever write to the same list. Partition p of the next stage is then
//! the p-th list of every thread's set.
//!
//! Records are partitioned by the top bits of a string's hash, so equal strings
//! always share a partition. Within a partition, a LocalTable groups them;
//! records with equal hashes are compared exactly, so a hash collision can never
//! merge two different strings.

pub const GROUP_BITS: u32 = 8;
pub const GROUP_PARTS: usize = 1 << GROUP_BITS;

/// The hash partition of a string with hash h.
#[inline]
pub fn hash_part(h: u64) -> usize {
    (h >> (64 - GROUP_BITS)) as usize
}

/// One thread's records, by partition.
pub type Buckets<T> = Vec<Vec<T>>;

pub fn buckets<T>(parts: usize) -> Buckets<T> {
    (0..parts).map(|_| Vec::new()).collect()
}

/// Buckets sized for about `total` records spread over every thread's buckets,
/// so they rarely need to grow.
pub fn buckets_for<T>(parts: usize, total: usize) -> Buckets<T> {
    let each = total / (parts * rayon::current_num_threads()) * 5 / 4 + 16;
    (0..parts).map(|_| Vec::with_capacity(each)).collect()
}

/// One value per worker thread of the rayon pool, which only that thread
/// touches: a thread's records go to its own buckets without any locking.
pub struct PerThread<T> {
    cells: Vec<std::cell::UnsafeCell<T>>,
}

// Each cell is only ever used by one thread: the worker whose index it has, or main.
unsafe impl<T: Send> Sync for PerThread<T> {}

impl<T> PerThread<T> {
    /// One value per worker, plus one for the main thread: a parallel loop over
    /// a single item (a tiny input) runs on the calling thread, not in the pool.
    pub fn new(make: impl Fn() -> T) -> PerThread<T> {
        PerThread { cells: (0..=rayon::current_num_threads()).map(|_| std::cell::UnsafeCell::new(make())).collect() }
    }

    /// f on the calling thread's value. Must be called from a pool worker or
    /// the main thread, and f must not start more parallel work (which could
    /// run another task on this thread while f holds its value).
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let i = rayon::current_thread_index().unwrap_or(self.cells.len() - 1);
        f(unsafe { &mut *self.cells[i].get() })
    }

    pub fn into_inner(self) -> Vec<T> {
        self.cells.into_iter().map(std::cell::UnsafeCell::into_inner).collect()
    }
}

/// Every thread's buckets regrouped by partition: [partition][thread].
pub fn by_partition<T>(threads: Vec<Buckets<T>>, parts: usize) -> Vec<Vec<Vec<T>>> {
    let mut out: Vec<Vec<Vec<T>>> = (0..parts).map(|_| Vec::with_capacity(threads.len())).collect();
    for b in threads {
        for (p, v) in b.into_iter().enumerate() {
            if !v.is_empty() {
                out[p].push(v);
            }
        }
    }
    out
}

/// An open-addressing hash table from strings to small group numbers, for one
/// partition. It is sized for `cap` strings up front and never grows.
pub struct LocalTable {
    slots: Vec<u32>,
    hashes: Vec<u64>,
    mask: usize,
}

impl LocalTable {
    pub fn new(cap: usize) -> LocalTable {
        let size = (2 * cap).next_power_of_two().max(16);
        LocalTable { slots: vec![u32::MAX; size], hashes: Vec::with_capacity(cap), mask: size - 1 }
    }

    /// The group of a string with hash h: an existing group g with the same
    /// hash for which same(g) holds, or else a new group. Returns (group, is_new);
    /// groups are numbered 0, 1, ... in order of first appearance.
    #[inline]
    pub fn get(&mut self, h: u64, same: impl Fn(u32) -> bool) -> (u32, bool) {
        let mut slot = h as usize & self.mask;
        loop {
            let g = self.slots[slot];
            if g == u32::MAX {
                let g = self.hashes.len() as u32;
                self.slots[slot] = g;
                self.hashes.push(h);
                return (g, true);
            }
            if self.hashes[g as usize] == h && same(g) {
                return (g, false);
            }
            slot = (slot + 1) & self.mask;
        }
    }
}
