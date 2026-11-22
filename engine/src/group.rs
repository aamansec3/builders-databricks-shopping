//! Partitioning rows, and grouping equal strings exactly, in parallel.
//!
//! Rows are partitioned by the top bits of their hash, so equal strings always
//! share a partition. The partition step copies each row's record (its hash
//! and whatever else the next step needs) into its partition's slot, so each
//! partition's records are contiguous and later steps read them front to back.
//! Each partition, on its own thread, then builds a small hash table; records
//! with equal hashes are compared exactly, so a collision can never merge two
//! different strings.

use crate::parse::{Out, zeroed};
use rayon::prelude::*;

pub const GROUP_BITS: u32 = 10;
pub const GROUP_PARTS: usize = 1 << GROUP_BITS;

/// The hash partition of a string with hash h.
#[inline]
pub fn hash_part(h: u64) -> u16 {
    (h >> (64 - GROUP_BITS)) as u16
}

/// Stable partitioning of rows 0..part.len() by part[i] in 0..nparts (nparts =
/// dropped): recs[off[p]..off[p + 1]] are record(i) for the rows i of
/// partition p, in row order. T must be valid when all-zero.
pub fn scatter<T, R>(part: &[u16], nparts: usize, record: R) -> (Vec<T>, Vec<usize>)
where
    T: Copy + Send + Sync,
    R: Fn(usize) -> T + Sync,
{
    let n = part.len();
    let nch = n.div_ceil(1 << 16).max(1);
    let size = n.div_ceil(nch).max(1);
    let width = nparts + 1;
    let mut hist = vec![0usize; nch * width];
    hist.par_chunks_mut(width).zip(part.par_chunks(size)).for_each(|(h, rows)| {
        for &p in rows {
            h[p as usize] += 1;
        }
    });
    let mut off = vec![0usize; nparts + 2];
    let mut run = 0;
    for p in 0..width {
        off[p] = run;
        for c in 0..nch {
            let k = hist[c * width + p];
            hist[c * width + p] = run;
            run += k;
        }
    }
    off[width] = run;
    let mut recs: Vec<T> = zeroed(off[nparts]);
    let out = Out::new(&mut recs);
    hist.par_chunks_mut(width).zip(part.par_chunks(size)).enumerate().for_each(|(c, (pos, rows))| {
        for (i, &p) in rows.iter().enumerate() {
            if (p as usize) < nparts {
                unsafe { out.set(pos[p as usize], record(c * size + i)) };
                pos[p as usize] += 1;
            }
        }
    });
    (recs, off)
}

/// Groups of equal strings among records partitioned by hash_part.
pub struct Groups {
    /// The group of each record: gid[j] is the group of recs[j].
    pub gid: Vec<u32>,
    /// The position in recs of each group's first record.
    pub rep: Vec<u32>,
    /// Partition p's groups are goff[p]..goff[p + 1].
    pub goff: Vec<usize>,
}

/// Group records whose strings are equal: hash(r) is the string's hash, and
/// same(a, b) tells, for two records with equal hashes, whether the strings are
/// equal.
pub fn group<T, H, S>(recs: &[T], off: &[usize], hash: H, same: S) -> Groups
where
    T: Sync,
    H: Fn(&T) -> u64 + Sync,
    S: Fn(&T, &T) -> bool + Sync,
{
    let nparts = off.len() - 2;
    let mut gid = vec![0u32; recs.len()];
    let out = Out::new(&mut gid);
    let firsts: Vec<Vec<u32>> = (0..nparts)
        .into_par_iter()
        .map(|p| {
            let part = &recs[off[p]..off[p + 1]];
            let size = (2 * part.len()).next_power_of_two().max(2);
            let mask = size - 1;
            let mut table = vec![u32::MAX; size]; // slot -> local group
            let mut first: Vec<u32> = Vec::new(); // local position of each group's first record
            let mut hashes: Vec<u64> = Vec::new(); // and its hash
            for (j, r) in part.iter().enumerate() {
                let h = hash(r);
                let mut slot = h as usize & mask;
                let g = loop {
                    let g = table[slot];
                    if g == u32::MAX {
                        table[slot] = first.len() as u32;
                        first.push(j as u32);
                        hashes.push(h);
                        break table[slot];
                    }
                    if hashes[g as usize] == h && same(&part[first[g as usize] as usize], r) {
                        break g;
                    }
                    slot = (slot + 1) & mask;
                };
                unsafe { out.set(off[p] + j, g) };
            }
            first.iter().map(|&f| (off[p] + f as usize) as u32).collect()
        })
        .collect();
    let mut goff = vec![0usize; nparts + 1];
    for p in 0..nparts {
        goff[p + 1] = goff[p] + firsts[p].len();
    }
    let out = Out::new(&mut gid);
    (0..nparts).into_par_iter().for_each(|p| {
        for j in off[p]..off[p + 1] {
            unsafe { *out.at(j) += goff[p] as u32 };
        }
    });
    Groups { gid, rep: firsts.concat(), goff }
}
