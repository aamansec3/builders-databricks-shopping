//! The rules.md CSV dialect and timestamp forms, and splitting the input files
//! into line-aligned chunks that threads parse in parallel.
//!
//! Files are memory-mapped, never copied: a parsed string is a (pointer, length)
//! pair into its file's mapping, and the mappings live until the program exits.
//! UTF-8 is safe to scan byte by byte: every byte of a multi-byte character is
//! >= 0x80, so none can be mistaken for a comma, quote or newline.

use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

pub const NULL: i64 = i64::MIN; // a bad timestamp
pub const DAY: i64 = 86_400_000;

/// A raw pointer that parallel tasks may write through, each to its own rows.
#[derive(Clone, Copy)]
pub struct Out<T>(*mut T);
unsafe impl<T> Send for Out<T> {}
unsafe impl<T> Sync for Out<T> {}

impl<T> Out<T> {
    pub fn new(v: &mut [T]) -> Self {
        Out(v.as_mut_ptr())
    }
    /// Safety: i is in bounds and no other task writes index i.
    #[inline]
    pub unsafe fn set(self, i: usize, x: T) {
        unsafe { *self.0.add(i) = x }
    }
    /// Safety: as for set.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn at<'a>(self, i: usize) -> &'a mut T {
        unsafe { &mut *self.0.add(i) }
    }
}

/// The bytes a parsed string points at. The mappings outlive every use.
#[inline]
pub fn bytes<'a>(p: u64, n: u32) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(p as *const u8, n as usize) }
}

/// 64-bit hash of a short string: 8 bytes at a time, then a mixer so the top
/// bits are good for partitioning and the low bits for hash table slots.
#[inline]
pub fn hash(b: &[u8]) -> u64 {
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15 ^ b.len() as u64;
    let mut words = b.chunks_exact(8);
    for w in &mut words {
        let v = u64::from_le_bytes(w.try_into().unwrap());
        h = (h ^ v).wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(31);
    }
    let rest = words.remainder();
    if !rest.is_empty() {
        let mut t = [0u8; 8];
        t[..rest.len()].copy_from_slice(rest);
        h = (h ^ u64::from_le_bytes(t)).wrapping_mul(0x94D0_49BB_1331_11EB).rotate_left(29);
    }
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// Split a line into fields; fb[f] = [start, end) of field f, quotes excluded.
/// True when the line has exactly nf fields and every quote wraps a whole field.
#[inline]
pub fn fields(line: &[u8], fb: &mut [(usize, usize)]) -> bool {
    let nf = fb.len();
    let e = line.len();
    let mut p = 0;
    let mut k = 0;
    loop {
        let (cs, ce);
        if p < e && line[p] == b'"' {
            let mut q = p + 1;
            while q < e && line[q] != b'"' && line[q] != b',' {
                q += 1;
            }
            if q >= e || line[q] != b'"' {
                return false;
            }
            cs = p + 1;
            ce = q;
            p = q + 1;
        } else {
            let mut q = p;
            while q < e && line[q] != b',' && line[q] != b'"' {
                q += 1;
            }
            if q < e && line[q] == b'"' {
                return false;
            }
            cs = p;
            ce = q;
            p = q;
        }
        if k == nf {
            return false;
        }
        fb[k] = (cs, ce);
        k += 1;
        if p == e {
            return k == nf;
        }
        if line[p] != b',' {
            return false;
        }
        p += 1;
    }
}

/// Value of a field of ASCII digits only, or -1 if it is empty, has any other
/// character, or has more than max_sig significant digits (leading zeros are fine).
#[inline]
pub fn digits(f: &[u8], max_sig: usize) -> i64 {
    if f.is_empty() || !f.iter().all(|c| c.is_ascii_digit()) {
        return -1;
    }
    let zeros = f.iter().take_while(|&&c| c == b'0').count();
    let sig = &f[zeros..];
    if sig.len() > max_sig {
        return -1;
    }
    sig.iter().fold(0i64, |v, &c| v * 10 + (c - b'0') as i64)
}

#[inline]
fn two(f: &[u8], p: usize) -> i64 {
    let a = f[p].wrapping_sub(b'0');
    let b = f[p + 1].wrapping_sub(b'0');
    if a > 9 || b > 9 { -1 } else { (a * 10 + b) as i64 }
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// A ts field in form F1, F2 or F3 -> epoch ms, or NULL when bad. F3 values
/// over 18 significant digits can't fit in an i64 and are far outside any
/// window, so they are bad too.
pub fn ts(f: &[u8]) -> i64 {
    let n = f.len();
    if (n == 20 || n == 25) && f[4] == b'-' && f[7] == b'-' && f[10] == b'T' && f[13] == b':' && f[16] == b':' {
        let (hi, lo) = (two(f, 0), two(f, 2));
        let y = if hi >= 0 && lo >= 0 { hi * 100 + lo } else { -1 };
        let (mo, d) = (two(f, 5), two(f, 8));
        let (h, mi, se) = (two(f, 11), two(f, 14), two(f, 17));
        if y < 0 || !(1..=12).contains(&mo) || d < 1 || !(0..=23).contains(&h) || !(0..=59).contains(&mi)
            || !(0..=59).contains(&se)
        {
            return NULL;
        }
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let dim = match mo {
            2 => if leap { 29 } else { 28 },
            4 | 6 | 9 | 11 => 30,
            _ => 31,
        };
        if d > dim {
            return NULL;
        }
        let off = if n == 20 {
            if f[19] != b'Z' {
                return NULL; // 'Z', never 'z'
            }
            0
        } else {
            let sign = match f[19] {
                b'+' => 1,
                b'-' => -1,
                _ => return NULL,
            };
            let (oh, om) = (two(f, 20), two(f, 23));
            if f[22] != b':' || !(0..=23).contains(&oh) || !(0..=59).contains(&om) {
                return NULL;
            }
            sign * (oh * 3_600_000 + om * 60_000)
        };
        return days_from_civil(y, mo, d) * DAY + h * 3_600_000 + mi * 60_000 + se * 1000 - off;
    }
    let v = digits(f, 18);
    if v < 0 { NULL } else { v }
}

/// The non-empty *.csv files of one kind, mapped, and split into chunks of
/// whole lines. Each chunk knows its first row, so threads can parse chunks in
/// any order and still write every line to its own row.
pub struct Input {
    maps: Vec<Mmap>,
    chunks: Vec<Chunk>,
    pub rows: usize,
    pub bytes: usize,
}

struct Chunk {
    file: usize,
    start: usize,
    end: usize,
    row0: usize,
}

const CHUNK: usize = 1 << 20;

impl Input {
    pub fn open(dir: &Path) -> Input {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        paths.retain(|p| p.extension().is_some_and(|x| x == "csv"));
        paths.sort();
        let maps: Vec<Mmap> = paths
            .iter()
            .filter_map(|p| {
                let f = File::open(p).expect("open input file");
                if f.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
                    return None;
                }
                Some(unsafe { Mmap::map(&f).expect("map input file") })
            })
            .collect();
        // Chunk boundaries: every CHUNK bytes, moved forward to just after a \n.
        let mut chunks = Vec::new();
        for (file, m) in maps.iter().enumerate() {
            let mut start = 0;
            while start < m.len() {
                let mut end = (start + CHUNK).min(m.len());
                if end < m.len() {
                    end = match memchr::memchr(b'\n', &m[end..]) {
                        Some(i) => end + i + 1,
                        None => m.len(),
                    };
                }
                chunks.push(Chunk { file, start, end, row0: 0 });
                start = end;
            }
        }
        // A chunk's lines: its \n count, plus a last line with no \n at the end of a file.
        let counts: Vec<usize> = chunks
            .par_iter()
            .map(|c| {
                let data = &maps[c.file][c.start..c.end];
                memchr::memchr_iter(b'\n', data).count() + usize::from(data.last() != Some(&b'\n'))
            })
            .collect();
        let mut rows = 0;
        for (c, n) in chunks.iter_mut().zip(counts) {
            c.row0 = rows;
            rows += n;
        }
        let bytes = maps.iter().map(|m| m.len()).sum();
        Input { maps, chunks, rows, bytes }
    }

    /// Call f(row, line, is_header) for every line, in parallel over chunks.
    /// Line 0 of each file is its header.
    pub fn for_each_line<F: Fn(usize, &[u8], bool) + Sync>(&self, f: F) {
        self.chunks.par_iter().for_each(|c| {
            let data = &self.maps[c.file][c.start..c.end];
            let mut row = c.row0;
            let mut s = 0;
            for e in memchr::memchr_iter(b'\n', data) {
                f(row, &data[s..e], c.start == 0 && s == 0);
                row += 1;
                s = e + 1;
            }
            if s < data.len() {
                f(row, &data[s..], c.start == 0 && s == 0);
            }
        });
    }
}
