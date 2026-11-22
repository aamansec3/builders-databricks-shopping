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
    /// Safety: i is in bounds and no other task touches index i.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn at<'a>(self, i: usize) -> &'a mut T {
        unsafe { &mut *self.0.add(i) }
    }
}

/// A vec of n all-zero values. Only for types for which all-zero bytes are a
/// valid value. The memory comes zeroed from the kernel, so untouched pages
/// cost nothing; big ones ask for 2 MB huge pages, which cuts page faults 512x.
pub fn zeroed<T: Copy>(n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    let layout = std::alloc::Layout::array::<T>(n).unwrap();
    unsafe {
        let p = std::alloc::alloc_zeroed(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        if layout.size() >= 4 << 20 {
            libc::madvise(p as *mut libc::c_void, layout.size(), libc::MADV_HUGEPAGE);
        }
        Vec::from_raw_parts(p as *mut T, n, n)
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
///
/// Lines of up to 128 bytes (all but freak ones) first mark every comma and
/// quote in the line, 16 bytes at a time, as bits of a 128-bit mask; then each
/// field is found by jumping to the next marked byte, not by testing each byte.
#[inline]
pub fn fields(line: &[u8], fb: &mut [(usize, usize)]) -> bool {
    if line.len() > 128 {
        return fields_bytewise(line, fb);
    }
    let (special, quote) = marks(line);
    let nf = fb.len();
    let e = line.len();
    // The first comma or quote at or after x, or e. (Bits past e are never set.)
    let next = |x: usize| -> usize {
        let m = if x >= 128 { 0 } else { special >> x };
        if m == 0 { e } else { x + m.trailing_zeros() as usize }
    };
    let is_quote = |x: usize| (quote >> x) & 1 == 1;
    let mut p = 0;
    let mut k = 0;
    loop {
        let (cs, ce);
        if p < e && is_quote(p) {
            let q = next(p + 1);
            if q >= e || !is_quote(q) {
                return false;
            }
            cs = p + 1;
            ce = q;
            p = q + 1;
        } else {
            let q = next(p);
            if q < e && is_quote(q) {
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

/// Bit masks of the commas and quotes (special) and of the quotes alone in a
/// line of up to 128 bytes: bit i is byte i.
#[inline]
fn marks(line: &[u8]) -> (u128, u128) {
    let mut buf = [0u8; 128];
    buf[..line.len()].copy_from_slice(line);
    let (mut special, mut quote) = (0u128, 0u128);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        // SSE2: part of every x86-64 CPU.
        let (comma_v, quote_v) = (_mm_set1_epi8(b',' as i8), _mm_set1_epi8(b'"' as i8));
        for i in 0..line.len().div_ceil(16) {
            let v = _mm_loadu_si128(buf.as_ptr().add(16 * i) as *const __m128i);
            let q = _mm_movemask_epi8(_mm_cmpeq_epi8(v, quote_v)) as u32 as u128;
            let c = _mm_movemask_epi8(_mm_cmpeq_epi8(v, comma_v)) as u32 as u128;
            quote |= q << (16 * i);
            special |= (q | c) << (16 * i);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    for (i, &c) in line.iter().enumerate() {
        quote |= u128::from(c == b'"') << i;
        special |= u128::from(c == b'"' || c == b',') << i;
    }
    (special, quote)
}

/// fields, one byte at a time: for lines over 128 bytes.
fn fields_bytewise(line: &[u8], fb: &mut [(usize, usize)]) -> bool {
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
    if f.is_empty() {
        return -1;
    }
    let (mut v, mut sig) = (0i64, 0usize);
    for &c in f {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return -1;
        }
        if sig > 0 || d != 0 {
            sig += 1;
            if sig > max_sig {
                return -1;
            }
            v = v * 10 + d as i64;
        }
    }
    v
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

/// The non-empty *.csv files of one kind, memory-mapped.
pub struct Input {
    maps: Vec<Mmap>,
    pub bytes: usize,
}

/// Lines are parsed in blocks of this many bytes: a block parses the lines that
/// start inside it (the last one may run into the next block).
const BLOCK: usize = 1 << 20;

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
        let bytes = maps.iter().map(|m| m.len()).sum();
        Input { maps, bytes }
    }

    /// Ask the kernel to start reading every file now, in the background, so
    /// the data is (or is on its way to being) in memory when parsing gets to it.
    pub fn prefetch(&self) {
        for m in &self.maps {
            let _ = m.advise(memmap2::Advice::WillNeed);
        }
    }

    /// The blocks to parse: (file, first byte).
    fn blocks(&self) -> Vec<(usize, usize)> {
        self.maps
            .iter()
            .enumerate()
            .flat_map(|(i, m)| (0..m.len()).step_by(BLOCK).map(move |b| (i, b)))
            .collect()
    }

    /// f(line, is_header) for the lines that start in a block (line 0 of each
    /// file is its header; the block's last line may run into the next one).
    fn block_lines(&self, (i, b0): (usize, usize), mut f: impl FnMut(&[u8], bool)) {
        let m = &self.maps[i][..];
        let b1 = (b0 + BLOCK).min(m.len());
        let first = if b0 == 0 { 0 } else { memchr::memchr(b'\n', &m[b0 - 1..]).map_or(m.len(), |k| b0 + k) };
        let mut s = first;
        for e in memchr::memchr_iter(b'\n', &m[first..]) {
            if s >= b1 {
                return;
            }
            let e = first + e;
            f(&m[s..e], s == 0);
            s = e + 1;
        }
        if s < b1 && s < m.len() {
            f(&m[s..], s == 0); // a last line with no \n
        }
    }

    /// f(line, is_header) for every line of every file, in parallel over
    /// blocks, collected in file order. Each block starts as soon as a thread
    /// is free, so on a cold disk parsing overlaps with reading.
    pub fn parse<T, F>(&self, f: F) -> Vec<T>
    where
        T: Copy + Send + Sync,
        F: Fn(&[u8], bool) -> T + Sync,
    {
        let parts: Vec<Vec<T>> = self
            .blocks()
            .into_par_iter()
            .map(|b| {
                let mut out = Vec::new();
                self.block_lines(b, |line, header| out.push(f(line, header)));
                out
            })
            .collect();
        concat(&parts)
    }

    /// f(state, line, is_header) for every line of every file, in parallel
    /// over blocks, where state is the calling thread's own.
    pub fn for_each_line<A, F>(&self, state: &crate::group::PerThread<A>, f: F)
    where
        A: Send,
        F: Fn(&mut A, &[u8], bool) + Sync + Send,
    {
        self.blocks().into_par_iter().for_each(|b| {
            state.with(|s| self.block_lines(b, |line, header| f(s, line, header)));
        });
    }
}

/// The parts, end to end, copied in parallel.
fn concat<T: Copy + Send + Sync>(parts: &[Vec<T>]) -> Vec<T> {
    let mut offs = Vec::with_capacity(parts.len());
    let mut total = 0;
    for p in parts {
        offs.push(total);
        total += p.len();
    }
    let mut all: Vec<T> = zeroed(total);
    let out = Out::new(&mut all);
    parts.par_iter().zip(offs).for_each(|(p, o)| {
        if !p.is_empty() {
            unsafe { std::ptr::copy_nonoverlapping(p.as_ptr(), out.at(o) as *mut T, p.len()) };
        }
    });
    all
}
