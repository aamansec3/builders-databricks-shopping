//! Shopping Sessions pipeline. Implements rules.md.
//!
//! Reads  $DATA_DIR   (default /data)
//! Writes $OUTPUT_DIR (default /output): user_daily/ and campaign_daily/ as Parquet
//!
//! Each stage hands its records straight to the partition the next stage works
//! on, so every row is written about once per stage:
//!   parse     each line of every file (mapped, not copied); event copies go to
//!             their event_id's hash partition, clicks to their user_id's
//!   events    per event_id partition: keep the earliest copy, validate it, and
//!             send it to its user_id's hash partition
//!   users     per user_id partition: number the IDs of events, clicks and links
//!   people    union-find over the links
//!   sessions  per person partition: sessions, user_daily rows, credit and assists
//!   write     the Parquet files

mod group;
mod output;
mod parse;
mod sessions;

use group::{Buckets, GROUP_PARTS, LocalTable, PerThread, buckets, buckets_for, by_partition, hash_part};
use output::Column;
use parquet::data_type::ByteArray;
use parse::{DAY, Input, NULL, bytes, digits, fields, hash, ts};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use sessions::{SessClick, SessEvent, WEEK};
use std::path::{Path, PathBuf};
use std::time::Instant;

const PERSON_PARTS: usize = 128; // partitions of people for the sessions stage
const VIEW: u8 = 1;
const CART: u8 = 2;
const PURCHASE: u8 = 3;

/// A user ID: its hash, its first 16 bytes packed into two words (with the
/// length, that identifies an ID of up to 16 bytes exactly, without reading the
/// input again), and where it is in the input.
#[derive(Clone, Copy, Default)]
struct UserKey {
    hash: u64,
    w0: u64,
    w1: u64,
    p: u64,
    n: u32,
}

impl UserKey {
    fn of(b: &[u8]) -> UserKey {
        let mut t = [0u8; 16];
        let k = b.len().min(16);
        t[..k].copy_from_slice(&b[..k]);
        UserKey {
            hash: hash(b),
            w0: u64::from_le_bytes(t[..8].try_into().unwrap()),
            w1: u64::from_le_bytes(t[8..].try_into().unwrap()),
            p: b.as_ptr() as u64,
            n: b.len() as u32,
        }
    }
    fn bytes(&self) -> &'static [u8] {
        bytes(self.p, self.n)
    }
    /// Whether two IDs with equal hashes are the same ID.
    fn same(&self, o: &UserKey) -> bool {
        self.n == o.n && self.w0 == o.w0 && self.w1 == o.w1 && (self.n <= 16 || self.bytes() == o.bytes())
    }
}

#[derive(Clone, Copy, Default)]
struct Link {
    a: UserKey,
    b: UserKey,
    ok: bool,
}

/// One copy of an event, in its event_id's hash partition. valid holds every
/// field rule and the late rule; it only matters for the copy dedup keeps
/// (a later valid copy doesn't rescue an invalid kept one).
#[derive(Clone, Copy)]
struct EventCopy {
    eid: u64,
    eid_hash: u64,
    ingest: i64,
    ts: i64,
    /// price << 16 | category << 1 | is_purchase
    pay: i64,
    user: UserKey,
    eid_n: u32,
    valid: bool,
}

/// A kept, valid event in its user_id's hash partition; gid is its user ID's
/// number within the partition.
#[derive(Clone, Copy)]
struct Kept {
    user: UserKey,
    ts: i64,
    pay: i64,
    gid: u32,
}

/// A valid click in its user_id's hash partition. Copies of a click are
/// identical, so they aren't deduped: a copy can't change which campaign a
/// session credits, or the set of campaigns that assist it.
#[derive(Clone, Copy)]
struct ClickRec {
    user: UserKey,
    ts: i64,
    cid: u64,
    cid_n: u32,
    camp: u32,
    gid: u32,
}

/// One side (2j or 2j + 1) of link j, in its user_id's hash partition.
#[derive(Clone, Copy)]
struct LinkSide {
    user: UserKey,
    side: u32,
    gid: u32,
}

/// The value of "key": "..." in a flat JSON object.
fn json_str<'a>(json: &'a str, key: &str) -> &'a str {
    let at = json.find(&format!("\"{key}\"")).unwrap_or_else(|| panic!("manifest has no {key}"));
    let rest = &json[at + key.len() + 2..];
    let open = rest.find('"').unwrap() + 1;
    let close = open + rest[open..].find('"').unwrap();
    &rest[open..close]
}

/// The (key, value) rows of a clean two-column CSV.
fn dimension(text: &[u8]) -> Vec<(&[u8], &[u8])> {
    let mut fb = [(0, 0); 2];
    let mut rows = Vec::new();
    for line in text.split(|&c| c == b'\n').skip(1) {
        if fields(line, &mut fb) {
            rows.push((&line[fb[0].0..fb[0].1], &line[fb[1].0..fb[1].1]));
        }
    }
    rows
}

/// This process's CPU time so far, in seconds: (user, sys). Linux counts it in
/// clock ticks of 1/100 s, fields 14 and 15 of /proc/self/stat.
fn cpu_time() -> (f64, f64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Skip past "pid (command)": the command may hold spaces.
    let rest = stat.rsplit_once(')').map_or("", |r| r.1);
    let f: Vec<f64> = rest.split_whitespace().skip(11).take(2).filter_map(|x| x.parse().ok()).collect();
    if f.len() == 2 { (f[0] / 100.0, f[1] / 100.0) } else { (0.0, 0.0) }
}

/// Logs each stage with its wall time and the CPU time it used, user and sys
/// (kernel). CPU time is a steadier measure than wall time on a busy machine.
struct Stages {
    start: Instant,
    last: std::cell::Cell<(f64, f64, f64)>,
}

impl Stages {
    fn new() -> Stages {
        let (u, s) = cpu_time();
        Stages { start: Instant::now(), last: std::cell::Cell::new((0.0, u, s)) }
    }
    fn log(&self, msg: String) {
        let wall = self.start.elapsed().as_secs_f64();
        let (u, s) = cpu_time();
        let (w0, u0, s0) = self.last.replace((wall, u, s));
        eprintln!("[{wall:6.2}s] {:5.2}s wall {:5.2}s user {:5.2}s sys  {msg}", wall - w0, u - u0, s - s0);
    }
}

fn bits(x: i64) -> u32 {
    64 - (x.max(1) as u64).leading_zeros()
}

fn main() {
    let stages = Stages::new();
    let log = |msg: String| stages.log(msg);
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let data = PathBuf::from(env("DATA_DIR", "/data"));
    let out = PathBuf::from(env("OUTPUT_DIR", "/output"));
    let ncpu = env("NCPU", "").parse().unwrap_or_else(|_| std::thread::available_parallelism().map_or(1, |n| n.get()));
    rayon::ThreadPoolBuilder::new().num_threads(ncpu).build_global().unwrap();

    let manifest = std::fs::read_to_string(data.join("manifest.json")).expect("read manifest.json");
    let (ws, we) = (ts(json_str(&manifest, "window_start").as_bytes()), ts(json_str(&manifest, "window_end").as_bytes()));
    assert!(ws != NULL && we != NULL, "bad window in manifest.json");
    log(format!("window {ws} .. {we}, {ncpu} threads"));

    // Products and campaigns become lookup tables. Categories are numbered in
    // byte order, so the smallest number wins a top_category tie.
    let products = std::fs::read(data.join("products.csv")).expect("read products.csv");
    let products = dimension(&products);
    let mut categories: Vec<&[u8]> = products.iter().map(|r| r.1).chain([&b"unknown"[..]]).collect();
    categories.sort_unstable();
    categories.dedup();
    let cat_of = |c: &[u8]| categories.binary_search(&c).unwrap() as i16;
    let unknown = cat_of(b"unknown");
    let product_cat: FxHashMap<&[u8], i16> = products.iter().map(|&(p, c)| (p, cat_of(c))).collect();
    let campaigns = std::fs::read(data.join("campaigns.csv")).expect("read campaigns.csv");
    let campaigns = dimension(&campaigns);
    let campaign_index: FxHashMap<&[u8], u32> =
        campaigns.iter().enumerate().map(|(i, &(c, _))| (c, i as u32)).collect();
    let ncamp = campaigns.len();

    // --- Parse. A background thread asks the kernel to start reading every
    // input file at once; parsing goes block by block as the data arrives.
    let li = Input::open(&data.join("links"));
    let ei = Input::open(&data.join("events"));
    let ci = Input::open(&data.join("clicks"));
    let (links, copies, clicks) = std::thread::scope(|sc| {
        sc.spawn(|| {
            li.prefetch();
            ei.prefetch();
            ci.prefetch();
        });
        let links: Vec<Link> = li.parse(|line, header| {
            let mut fb = [(0, 0); 2];
            if header || !fields(line, &mut fb) || fb[0].1 == fb[0].0 || fb[1].1 == fb[1].0 {
                return Link::default();
            }
            Link { a: UserKey::of(&line[fb[0].0..fb[0].1]), b: UserKey::of(&line[fb[1].0..fb[1].1]), ok: true }
        });
        log(format!("parsed links: {}", links.len()));

        // An events line is ~95 bytes, a clicks line ~50.
        let copies: PerThread<Buckets<EventCopy>> = PerThread::new(|| buckets_for(GROUP_PARTS, ei.bytes / 90));
        ei.for_each_line(&copies, |acc, line, header| {
                let mut fb = [(0, 0); 7];
                if header || !fields(line, &mut fb) || fb[0].1 == fb[0].0 {
                    return; // malformed: never the kept copy
                }
                let f = |k: usize| &line[fb[k].0..fb[k].1];
                let (eid, uid, pid) = (f(0), f(1), f(5));
                let typ = match f(4) {
                    b"view" => VIEW,
                    b"cart" => CART,
                    b"purchase" => PURCHASE,
                    _ => 0,
                };
                let (t, ingest) = (ts(f(2)), ts(f(3)));
                let price = if typ == PURCHASE { digits(f(6), 8) } else { -1 };
                let cat = if pid.is_empty() { -1 } else { *product_cat.get(pid).unwrap_or(&unknown) as i64 };
                let valid = !uid.is_empty()
                    && cat >= 0
                    && typ != 0
                    && t >= ws
                    && t < we
                    && (typ != PURCHASE || (1..=10_000_000).contains(&price))
                    && ingest != NULL
                    && ingest - t <= DAY;
                let h = hash(eid);
                acc[hash_part(h)].push(EventCopy {
                    eid: eid.as_ptr() as u64,
                    eid_hash: h,
                    ingest,
                    ts: t,
                    pay: if typ == PURCHASE { (price << 16) | (cat << 1) | 1 } else { cat << 1 },
                    user: if valid { UserKey::of(uid) } else { UserKey::default() },
                    eid_n: eid.len() as u32,
                    valid,
                });
        });
        log("parsed events".into());

        let clicks: PerThread<Buckets<ClickRec>> = PerThread::new(|| buckets_for(GROUP_PARTS, ci.bytes / 48));
        ci.for_each_line(&clicks, |acc, line, header| {
                let mut fb = [(0, 0); 4];
                if header || !fields(line, &mut fb) || fb[0].1 == fb[0].0 || fb[1].1 == fb[1].0 {
                    return;
                }
                let f = |k: usize| &line[fb[k].0..fb[k].1];
                let t = ts(f(3));
                if t < ws - WEEK || t >= we {
                    return;
                }
                let Some(&camp) = campaign_index.get(f(2)) else { return };
                let (cid, user) = (f(0), UserKey::of(f(1)));
                acc[hash_part(user.hash)].push(ClickRec {
                    user,
                    ts: t,
                    cid: cid.as_ptr() as u64,
                    cid_n: cid.len() as u32,
                    camp,
                    gid: 0,
                });
        });
        (links, by_partition(copies.into_inner(), GROUP_PARTS), by_partition(clicks.into_inner(), GROUP_PARTS))
    });
    log(format!("parsed clicks ({} MB in all)", (li.bytes + ei.bytes + ci.bytes) >> 20));

    // --- Events: per event_id partition, keep the earliest copy of each event
    // (copies with equal ingest_ts are identical, so any will do; a bad
    // ingest_ts never wins), and send it on if it is valid.
    let ncopies: usize = copies.iter().flatten().map(Vec::len).sum();
    let kept: PerThread<Buckets<Kept>> = PerThread::new(|| buckets_for(GROUP_PARTS, ncopies));
    copies.into_par_iter().for_each(|lists| {
        kept.with(|acc| {
                let n = lists.iter().map(Vec::len).sum();
                let mut table = LocalTable::new(n);
                let mut best: Vec<&EventCopy> = Vec::with_capacity(n);
                for c in lists.iter().flatten() {
                    let (g, new) = table.get(c.eid_hash, |g| {
                        let b = best[g as usize];
                        b.eid_n == c.eid_n && bytes(b.eid, b.eid_n) == bytes(c.eid, c.eid_n)
                    });
                    let key = |e: &EventCopy| if e.ingest == NULL { i64::MAX } else { e.ingest };
                    if new {
                        best.push(c);
                    } else if key(c) < key(best[g as usize]) {
                        best[g as usize] = c;
                    }
                }
                for e in best.into_iter().filter(|e| e.valid) {
                    acc[hash_part(e.user.hash)].push(Kept { user: e.user, ts: e.ts, pay: e.pay, gid: 0 });
                }
        })
    });
    let kept = by_partition(kept.into_inner(), GROUP_PARTS);
    log("events deduped, validated".into());

    // --- Users: per user_id partition, number the IDs of kept events, valid
    // clicks and both sides of each link. A user is goff[p] + its number in p.
    let mut sides: Buckets<LinkSide> = buckets(GROUP_PARTS);
    for (j, l) in links.iter().enumerate().filter(|(_, l)| l.ok) {
        for (s, user) in [(2 * j, l.a), (2 * j + 1, l.b)] {
            sides[hash_part(user.hash)].push(LinkSide { user, side: s as u32, gid: 0 });
        }
    }
    struct UserPart {
        kept: Vec<Vec<Kept>>,
        clicks: Vec<Vec<ClickRec>>,
        sides: Vec<LinkSide>,
        ids: Vec<UserKey>,
    }
    let mut parts: Vec<UserPart> = kept
        .into_iter()
        .zip(clicks)
        .zip(sides)
        .map(|((kept, clicks), sides)| UserPart { kept, clicks, sides, ids: Vec::new() })
        .collect();
    parts.par_iter_mut().for_each(|part| {
        let n = part.kept.iter().map(Vec::len).sum::<usize>()
            + part.clicks.iter().map(Vec::len).sum::<usize>()
            + part.sides.len();
        let mut table = LocalTable::new(n);
        let ids = &mut part.ids;
        let mut number = |user: &UserKey| {
            let (g, new) = table.get(user.hash, |g| ids[g as usize].same(user));
            if new {
                ids.push(*user);
            }
            g
        };
        for k in part.kept.iter_mut().flatten() {
            k.gid = number(&k.user);
        }
        for c in part.clicks.iter_mut().flatten() {
            c.gid = number(&c.user);
        }
        for s in part.sides.iter_mut() {
            s.gid = number(&s.user);
        }
    });
    let mut goff = vec![0u32; GROUP_PARTS + 1];
    for p in 0..GROUP_PARTS {
        goff[p + 1] = goff[p] + parts[p].ids.len() as u32;
    }
    let users = goff[GROUP_PARTS] as usize;
    let user_keys: Vec<UserKey> = parts.iter_mut().flat_map(|p| std::mem::take(&mut p.ids)).collect();
    let mut side_user = vec![0u32; 2 * links.len()];
    for (p, part) in parts.iter().enumerate() {
        for s in &part.sides {
            side_user[s.side as usize] = goff[p] + s.gid;
        }
    }
    log(format!("user IDs numbered: {users}"));

    // --- People: union-find over the links; a person is named by their
    // byte-wise smallest ID.
    let mut parent: Vec<u32> = (0..users as u32).collect();
    fn root(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for (j, l) in links.iter().enumerate() {
        if l.ok {
            let a = root(&mut parent, side_user[2 * j]);
            let b = root(&mut parent, side_user[2 * j + 1]);
            if a < b {
                parent[b as usize] = a;
            } else if b < a {
                parent[a as usize] = b;
            }
        }
    }
    let person: Vec<u32> = (0..users as u32).map(|u| root(&mut parent, u)).collect();
    drop(parent);
    let mut name: Vec<u32> = (0..users as u32).collect();
    for u in 0..users {
        let r = person[u] as usize;
        if r != u && user_keys[u].bytes() < user_keys[name[r] as usize].bytes() {
            name[r] = u as u32;
        }
    }
    log("people joined (union-find)".into());

    // --- Sessions, credit, assists: each user_id partition sends its events and
    // clicks, keyed by person, to the person's partition.
    let eshift = bits(we - ws);
    let cshift = bits(we - ws + WEEK);
    assert!(bits(users as i64) + cshift <= 62, "window too long for the sort keys");
    let nkept: usize = parts.iter().map(|p| p.kept.iter().map(Vec::len).sum::<usize>()).sum();
    let nclicks: usize = parts.iter().map(|p| p.clicks.iter().map(Vec::len).sum::<usize>()).sum();
    #[allow(clippy::type_complexity)]
    let to_people: PerThread<(Buckets<SessEvent>, Buckets<SessClick>)> =
        PerThread::new(|| (buckets_for(PERSON_PARTS, nkept), buckets_for(PERSON_PARTS, nclicks)));
    parts.par_iter().enumerate().for_each(|(p, part)| {
        to_people.with(|(ev, cl)| {
                for k in part.kept.iter().flatten() {
                    let who = person[(goff[p] + k.gid) as usize] as i64;
                    ev[who as usize % PERSON_PARTS].push(((who << eshift) | (k.ts - ws), k.pay));
                }
                for c in part.clicks.iter().flatten() {
                    let who = person[(goff[p] + c.gid) as usize] as i64;
                    cl[who as usize % PERSON_PARTS].push(((who << cshift) | (c.ts - (ws - WEEK)), c.camp, c.cid_n, c.cid));
                }
        })
    });
    let (ev, cl): (Vec<_>, Vec<_>) = to_people.into_inner().into_iter().unzip();
    drop(parts);
    log("records sent to person partitions".into());
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;
    let inp = sessions::Input { eshift, cshift, ws, ncat: categories.len(), ncamp, day0, ndays };
    let (rows, acc) = sessions::run(&inp, by_partition(ev, PERSON_PARTS), by_partition(cl, PERSON_PARTS));
    log(format!("sessions, credit, assists: {} user_daily rows", rows.len()));

    // --- Outputs
    let dir = |name: &str| -> PathBuf {
        let d = out.join(name);
        std::fs::create_dir_all(&d).expect("create output folder");
        d
    };
    // user_daily goes to one file per thread, written in parallel: the rules
    // allow any number of part files.
    let user_daily = || {
        let d = dir("user_daily");
        let per = rows.len().div_ceil(ncpu).max(1);
        let parts: Vec<&[sessions::UserDay]> = if rows.is_empty() { vec![&rows[..]] } else { rows.chunks(per).collect() };
        parts.into_par_iter().enumerate().for_each(|(i, rows)| {
            let user_id = rows.iter().map(|r| ByteArray::from(user_keys[name[r.person as usize] as usize].bytes().to_vec())).collect();
            let present: Vec<i16> = rows.iter().map(|r| i16::from(r.top >= 0)).collect();
            let top =
                rows.iter().filter(|r| r.top >= 0).map(|r| ByteArray::from(categories[r.top as usize].to_vec())).collect();
            write(&d.join(format!("part-{i}.parquet")), output::USER_DAILY, vec![
                Column::Str(user_id, None),
                Column::Int32(rows.iter().map(|r| r.day).collect()),
                Column::Int64(rows.iter().map(|r| r.sessions).collect()),
                Column::Int64(rows.iter().map(|r| r.events).collect()),
                Column::Int64(rows.iter().map(|r| r.revenue).collect()),
                Column::Str(top, Some(present)),
            ]);
        });
    };
    let campaign_daily = || {
        let cells: Vec<(usize, usize)> = (0..=ncamp)
            .flat_map(|c| (0..ndays).map(move |d| (c, d)))
            .filter(|&(c, d)| acc.sessions[c * ndays + d] > 0 || acc.assisted[c * ndays + d] > 0)
            .collect();
        let id = |c: usize| if c == ncamp { &b"organic"[..] } else { campaigns[c].0 };
        let channel = |c: usize| if c == ncamp { &b"organic"[..] } else { campaigns[c].1 };
        let at = |v: &Vec<i64>| cells.iter().map(|&(c, d)| v[c * ndays + d]).collect();
        write(&dir("campaign_daily").join("part-0.parquet"), output::CAMPAIGN_DAILY, vec![
            Column::Str(cells.iter().map(|&(c, _)| ByteArray::from(id(c).to_vec())).collect(), None),
            Column::Str(cells.iter().map(|&(c, _)| ByteArray::from(channel(c).to_vec())).collect(), None),
            Column::Int32(cells.iter().map(|&(_, d)| (day0 + d as i64) as i32).collect()),
            Column::Int64(at(&acc.sessions)),
            Column::Int64(at(&acc.purchasing)),
            Column::Int64(at(&acc.revenue)),
            Column::Int64(at(&acc.assisted)),
        ]);
    };
    rayon::join(user_daily, campaign_daily);
    log("wrote user_daily and campaign_daily".into());
    let (u, s) = cpu_time();
    eprintln!("total: {:.2}s wall, {u:.2}s user, {s:.2}s sys", stages.start.elapsed().as_secs_f64());
    // Exit now: the kernel takes back all the memory at once, faster than
    // freeing gigabytes of vectors and mappings one by one on the way out.
    std::process::exit(0);
}

fn write(path: &Path, schema: &str, columns: Vec<Column>) {
    output::write(path, schema, columns).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}
