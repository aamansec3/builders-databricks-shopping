//! Grouping equal strings exactly, in parallel.
//!
//! Rows are partitioned by the top bits of their hash, so equal strings always
//! share a partition; each partition, on its own thread, then builds a small
//! hash table that compares the bytes whenever two hashes are equal, so a hash
//! collision can never merge two different strings.

use crate::parse::Out;
use rayon::prelude::*;

pub const GROUP_BITS: u32 = 10;
pub const GROUP_PARTS: usize = 1 << GROUP_BITS;

/// The hash partition of a string with hash h.
#[inline]
pub fn hash_part(h: u64) -> u16 {
    (h >> (64 - GROUP_BITS)) as u16
}

/// Stable counting sort of row numbers by part[i] in 0..nparts (nparts = dropped).
/// order[off[p]..off[p + 1]] are the rows of partition p, in their original order.
pub fn partition(part: &[u16], nparts: usize) -> (Vec<u32>, Vec<usize>) {
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
    let mut order = vec![0u32; n];
    let out = Out::new(&mut order);
    hist.par_chunks_mut(width).zip(part.par_chunks(size)).enumerate().for_each(|(c, (pos, rows))| {
        for (i, &p) in rows.iter().enumerate() {
            unsafe { out.set(pos[p as usize], (c * size + i) as u32) };
            pos[p as usize] += 1;
        }
    });
    (order, off)
}

/// Groups of equal strings among the rows partitioned by hash_part.
pub struct Groups {
    /// The group of each row in partition order: gidp[j] is the group of row order[j].
    pub gidp: Vec<u32>,
    /// The first row of each group.
    pub rep: Vec<u32>,
    /// Partition p's groups are goff[p]..goff[p + 1].
    pub goff: Vec<usize>,
}

pub fn group<'a, H, S>(order: &[u32], off: &[usize], hash: H, string: S) -> Groups
where
    H: Fn(u32) -> u64 + Sync,
    S: Fn(u32) -> &'a [u8] + Sync,
{
    let nparts = off.len() - 2;
    let mut gidp = vec![0u32; off[nparts]];
    let out = Out::new(&mut gidp);
    // Per partition: first copy each row's hash and string pointer into local
    // arrays (independent loads the CPU overlaps), then probe a table over them.
    let firsts: Vec<Vec<u32>> = (0..nparts)
        .into_par_iter()
        .map(|p| {
            let rows = &order[off[p]..off[p + 1]];
            let hl: Vec<u64> = rows.iter().map(|&r| hash(r)).collect();
            let sl: Vec<&[u8]> = rows.iter().map(|&r| string(r)).collect();
            let size = (2 * rows.len()).next_power_of_two().max(2);
            let mask = size - 1;
            let mut table = vec![u32::MAX; size]; // slot -> local group
            let mut first: Vec<u32> = Vec::new(); // local position of each group's first row
            for j in 0..rows.len() {
                let mut slot = hl[j] as usize & mask;
                let g = loop {
                    let g = table[slot];
                    if g == u32::MAX {
                        table[slot] = first.len() as u32;
                        first.push(j as u32);
                        break table[slot];
                    }
                    let f = first[g as usize] as usize;
                    if hl[f] == hl[j] && sl[f] == sl[j] {
                        break g;
                    }
                    slot = (slot + 1) & mask;
                };
                unsafe { out.set(off[p] + j, g) };
            }
            first.iter().map(|&f| rows[f as usize]).collect()
        })
        .collect();
    let mut goff = vec![0usize; nparts + 1];
    for p in 0..nparts {
        goff[p + 1] = goff[p] + firsts[p].len();
    }
    let out = Out::new(&mut gidp);
    (0..nparts).into_par_iter().for_each(|p| {
        for j in off[p]..off[p + 1] {
            unsafe { *out.at(j) += goff[p] as u32 };
        }
    });
    let rep = firsts.concat();
    Groups { gidp, rep, goff }
}

/// The group of each of n rows (u32::MAX for dropped rows).
pub fn ungroup(g: &Groups, order: &[u32], n: usize) -> Vec<u32> {
    let mut gid = vec![u32::MAX; n];
    let out = Out::new(&mut gid);
    g.gidp.par_iter().zip(order.par_iter()).for_each(|(&gi, &r)| unsafe { out.set(r as usize, gi) });
    gid
}
