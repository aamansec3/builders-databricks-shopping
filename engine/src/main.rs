//! Shopping Sessions pipeline. Implements rules.md.
//!
//! Reads  $DATA_DIR   (default /data)
//! Writes $OUTPUT_DIR (default /output): user_daily/ and campaign_daily/ as Parquet
//!
//! Stages, each parallel over rows or partitions:
//!   parse     every line of every file, mapped not copied, into typed rows
//!   events    dedup on event_id (earliest ingest_ts), then validate the kept copy
//!   people    number every user ID, union-find over the links
//!   sessions  per person: sessions, user_daily rows, click credit and assists
//!   write     two Parquet files

mod group;
mod output;
mod parse;
mod sessions;

use group::{GROUP_PARTS, group, hash_part, partition, ungroup};
use output::Column;
use parquet::data_type::ByteArray;
use parse::{DAY, Input, NULL, Out, bytes, digits, fields, hash, ts};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use sessions::WEEK;
use std::path::{Path, PathBuf};
use std::time::Instant;

const PERSON_PARTS: usize = 128; // partitions of people for the sessions stage
const VIEW: u8 = 1;
const CART: u8 = 2;
const PURCHASE: u8 = 3;

#[derive(Clone, Copy)]
struct Link {
    a: u64,
    b: u64,
    a_n: u32,
    b_n: u32,
    ok: bool,
}

/// One copy of an event. ok when well-formed with a non-empty event_id; the
/// other field rules are checked on the kept copy, after dedup. price is -1
/// unless it's a purchase whose price is digits only with at most 8 significant
/// digits. cat is the product's category (unknown when not in products.csv), or
/// -1 when product_id is empty: only the category is ever needed, not the ID.
#[derive(Clone, Copy)]
struct Event {
    eid: u64,
    uid: u64,
    hash: u64,
    ts: i64,
    ingest: i64,
    eid_n: u32,
    uid_n: u32,
    price: i32,
    cat: i16,
    typ: u8,
    ok: bool,
}

/// One click. ok when it's valid: well-formed, non-empty click_id and user_id,
/// ts in the click window, campaign in campaigns.csv. Copies of a click are
/// identical, so they aren't deduped: a copy can't change which campaign a
/// session credits, or the set of campaigns that assist it.
#[derive(Clone, Copy)]
struct Click {
    cid: u64,
    uid: u64,
    ts: i64,
    cid_n: u32,
    uid_n: u32,
    camp: u32,
    ok: bool,
}

/// A user ID to number, from a link, a kept event or a valid click.
#[derive(Clone, Copy)]
struct Item {
    p: u64,
    n: u32,
}

/// A vec of n all-zero values (calloc: untouched pages cost nothing). Only for
/// types for which all-zero bytes are a valid value.
fn zeroed<T: Copy>(n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    let layout = std::alloc::Layout::array::<T>(n).unwrap();
    unsafe {
        let p = std::alloc::alloc_zeroed(layout) as *mut T;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Vec::from_raw_parts(p, n, n)
    }
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

fn bits(x: i64) -> u32 {
    64 - (x.max(1) as u64).leading_zeros()
}

fn main() {
    let t0 = Instant::now();
    let log = |msg: String| eprintln!("[{:6.2}s] {msg}", t0.elapsed().as_secs_f64());
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

    // --- Parse
    let li = Input::open(&data.join("links"));
    let ei = Input::open(&data.join("events"));
    let ci = Input::open(&data.join("clicks"));

    let mut links: Vec<Link> = zeroed(li.rows);
    let o = Out::new(&mut links);
    li.for_each_line(|r, line, header| {
        let mut fb = [(0, 0); 2];
        let good = !header && fields(line, &mut fb) && fb[0].1 > fb[0].0 && fb[1].1 > fb[1].0;
        if good {
            let (a, b) = (&line[fb[0].0..fb[0].1], &line[fb[1].0..fb[1].1]);
            let l = Link { a: a.as_ptr() as u64, b: b.as_ptr() as u64, a_n: a.len() as u32, b_n: b.len() as u32, ok: true };
            unsafe { o.set(r, l) };
        }
    });

    let mut events: Vec<Event> = zeroed(ei.rows);
    let o = Out::new(&mut events);
    ei.for_each_line(|r, line, header| {
        let mut fb = [(0, 0); 7];
        if header || !fields(line, &mut fb) || fb[0].1 == fb[0].0 {
            return;
        }
        let f = |k: usize| &line[fb[k].0..fb[k].1];
        let (eid, uid, pid) = (f(0), f(1), f(5));
        let typ = match f(4) {
            b"view" => VIEW,
            b"cart" => CART,
            b"purchase" => PURCHASE,
            _ => 0,
        };
        let e = Event {
            eid: eid.as_ptr() as u64,
            uid: uid.as_ptr() as u64,
            hash: hash(eid),
            ts: ts(f(2)),
            ingest: ts(f(3)),
            eid_n: eid.len() as u32,
            uid_n: uid.len() as u32,
            price: if typ == PURCHASE { digits(f(6), 8) as i32 } else { -1 },
            cat: if pid.is_empty() { -1 } else { *product_cat.get(pid).unwrap_or(&unknown) },
            typ,
            ok: true,
        };
        unsafe { o.set(r, e) };
    });

    let mut clicks: Vec<Click> = zeroed(ci.rows);
    let o = Out::new(&mut clicks);
    ci.for_each_line(|r, line, header| {
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
        let (cid, uid) = (f(0), f(1));
        let c = Click {
            cid: cid.as_ptr() as u64,
            uid: uid.as_ptr() as u64,
            ts: t,
            cid_n: cid.len() as u32,
            uid_n: uid.len() as u32,
            camp,
            ok: true,
        };
        unsafe { o.set(r, c) };
    });
    log(format!(
        "parsed {} MB: {} link, {} event, {} click lines",
        (li.bytes + ei.bytes + ci.bytes) >> 20,
        links.len(),
        events.len(),
        clicks.len()
    ));

    // --- Events: keep the earliest copy of each event_id, then validate it
    let part: Vec<u16> =
        events.par_iter().map(|e| if e.ok { hash_part(e.hash) } else { GROUP_PARTS as u16 }).collect();
    let (order, off) = partition(&part, GROUP_PARTS);
    drop(part);
    let g = group(&order, &off, |r| events[r as usize].hash, |r| {
        let e = &events[r as usize];
        bytes(e.eid, e.eid_n)
    });
    let mut chosen = vec![false; events.len()];
    let o = Out::new(&mut chosen);
    (0..GROUP_PARTS).into_par_iter().for_each(|p| {
        // The copy with the smallest ingest_ts. Copies with equal ingest_ts are
        // identical, so any of them will do; a bad ingest_ts never wins.
        let g0 = g.goff[p];
        let mut best = vec![(u32::MAX, i64::MAX); g.goff[p + 1] - g0];
        for j in off[p]..off[p + 1] {
            let r = order[j];
            let ing = events[r as usize].ingest;
            let key = if ing == NULL { i64::MAX } else { ing };
            let b = &mut best[g.gidp[j] as usize - g0];
            if b.0 == u32::MAX || key < b.1 {
                *b = (r, key);
            }
        }
        for (r, _) in best {
            unsafe { o.set(r as usize, true) };
        }
    });
    drop((order, off, g));
    let keep: Vec<bool> = events
        .par_iter()
        .zip(&chosen)
        .map(|(e, &c)| {
            c && e.uid_n > 0
                && e.cat >= 0
                && e.typ != 0
                && e.ts >= ws
                && e.ts < we
                && (e.typ != PURCHASE || (1..=10_000_000).contains(&e.price))
                && e.ingest != NULL
                && e.ingest - e.ts <= DAY
        })
        .collect();
    drop(chosen);
    log("events deduped, validated".into());

    // --- People: number every user ID (both sides of each link, then each
    // kept event, then each valid click), then join linked ones.
    let (nl, ne, nc) = (links.len(), events.len(), clicks.len());
    let m = 2 * nl + ne + nc;
    let mut items: Vec<Item> = zeroed(m);
    let mut ih: Vec<u64> = zeroed(m);
    let mut part: Vec<u16> = zeroed(m);
    let (oi, oh, op) = (Out::new(&mut items), Out::new(&mut ih), Out::new(&mut part));
    let put = |i: usize, p: u64, n: u32, used: bool| unsafe {
        if used {
            let h = hash(bytes(p, n));
            oi.set(i, Item { p, n });
            oh.set(i, h);
            op.set(i, hash_part(h));
        } else {
            op.set(i, GROUP_PARTS as u16);
        }
    };
    links.par_iter().enumerate().for_each(|(j, l)| {
        put(2 * j, l.a, l.a_n, l.ok);
        put(2 * j + 1, l.b, l.b_n, l.ok);
    });
    events.par_iter().zip(&keep).enumerate().for_each(|(j, (e, &k))| put(2 * nl + j, e.uid, e.uid_n, k));
    clicks.par_iter().enumerate().for_each(|(j, c)| put(2 * nl + ne + j, c.uid, c.uid_n, c.ok));
    let (order, off) = partition(&part, GROUP_PARTS);
    drop(part);
    let g = group(&order, &off, |r| ih[r as usize], |r| {
        let it = &items[r as usize];
        bytes(it.p, it.n)
    });
    let ugid = ungroup(&g, &order, m);
    drop((order, off, ih));
    let users = g.rep.len();
    let item_str = |i: u32| {
        let it = &items[i as usize];
        bytes(it.p, it.n)
    };
    // Union-find over the links; a person is named by their byte-wise smallest ID.
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
            let a = root(&mut parent, ugid[2 * j]);
            let b = root(&mut parent, ugid[2 * j + 1]);
            if a < b {
                parent[b as usize] = a;
            } else if b < a {
                parent[a as usize] = b;
            }
        }
    }
    let person: Vec<u32> = (0..users as u32).map(|u| root(&mut parent, u)).collect();
    drop(parent);
    let mut name = g.rep.clone();
    for u in 0..users {
        let r = person[u] as usize;
        if r != u && item_str(g.rep[u]) < item_str(name[r]) {
            name[r] = g.rep[u];
        }
    }
    log(format!("people: {users} user IDs"));

    // --- Sessions, credit, assists
    // Sort keys are person << shift | (ts - earliest ts), and persons < users.
    let eshift = bits(we - ws);
    let cshift = bits(we - ws + WEEK);
    assert!(bits(users as i64) + cshift <= 62, "window too long for the sort keys");
    let mut ekey: Vec<i64> = zeroed(ne);
    let mut epay: Vec<i64> = zeroed(ne);
    let mut epart: Vec<u16> = zeroed(ne);
    let (ok_, op_, opart) = (Out::new(&mut ekey), Out::new(&mut epay), Out::new(&mut epart));
    events.par_iter().zip(&keep).enumerate().for_each(|(j, (e, &k))| unsafe {
        if !k {
            opart.set(j, PERSON_PARTS as u16);
            return;
        }
        let p = person[ugid[2 * nl + j] as usize] as i64;
        ok_.set(j, (p << eshift) | (e.ts - ws));
        let pay = if e.typ == PURCHASE { ((e.price as i64) << 16) | ((e.cat as i64) << 1) | 1 } else { (e.cat as i64) << 1 };
        op_.set(j, pay);
        opart.set(j, (p as usize % PERSON_PARTS) as u16);
    });
    let mut ckey: Vec<i64> = zeroed(nc);
    let mut cpart: Vec<u16> = zeroed(nc);
    let (ok_, opart) = (Out::new(&mut ckey), Out::new(&mut cpart));
    clicks.par_iter().enumerate().for_each(|(j, c)| unsafe {
        if !c.ok {
            opart.set(j, PERSON_PARTS as u16);
            return;
        }
        let p = person[ugid[2 * nl + ne + j] as usize] as i64;
        ok_.set(j, (p << cshift) | (c.ts - (ws - WEEK)));
        opart.set(j, (p as usize % PERSON_PARTS) as u16);
    });
    let camp: Vec<u32> = clicks.par_iter().map(|c| c.camp).collect();
    let (eorder, eoff) = partition(&epart, PERSON_PARTS);
    let (corder, coff) = partition(&cpart, PERSON_PARTS);
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;
    let inp = sessions::Input {
        ekey: &ekey,
        epay: &epay,
        eorder: &eorder,
        eoff: &eoff,
        eshift,
        ckey: &ckey,
        camp: &camp,
        corder: &corder,
        coff: &coff,
        cshift,
        ws,
        ncat: categories.len(),
        ncamp,
        day0,
        ndays,
    };
    let (rows, acc) = sessions::run(&inp, |r| {
        let c = &clicks[r as usize];
        bytes(c.cid, c.cid_n)
    });
    log(format!("sessions, credit, assists: {} user_daily rows", rows.len()));

    // --- Outputs
    let dir = |name: &str| -> PathBuf {
        let d = out.join(name);
        std::fs::create_dir_all(&d).expect("create output folder");
        d.join("part-0.parquet")
    };
    let user_daily = || {
        let user_id = rows.par_iter().map(|r| ByteArray::from(item_str(name[r.person as usize]).to_vec())).collect();
        let present: Vec<i16> = rows.iter().map(|r| i16::from(r.top >= 0)).collect();
        let top = rows.iter().filter(|r| r.top >= 0).map(|r| ByteArray::from(categories[r.top as usize].to_vec())).collect();
        write(&dir("user_daily"), output::USER_DAILY, vec![
            Column::Str(user_id, None),
            Column::Int32(rows.iter().map(|r| r.day).collect()),
            Column::Int64(rows.iter().map(|r| r.sessions).collect()),
            Column::Int64(rows.iter().map(|r| r.events).collect()),
            Column::Int64(rows.iter().map(|r| r.revenue).collect()),
            Column::Str(top, Some(present)),
        ]);
    };
    let campaign_daily = || {
        let cells: Vec<(usize, usize)> = (0..=ncamp)
            .flat_map(|c| (0..ndays).map(move |d| (c, d)))
            .filter(|&(c, d)| acc.sessions[c * ndays + d] > 0 || acc.assisted[c * ndays + d] > 0)
            .collect();
        let id = |c: usize| if c == ncamp { &b"organic"[..] } else { campaigns[c].0 };
        let channel = |c: usize| if c == ncamp { &b"organic"[..] } else { campaigns[c].1 };
        let at = |v: &Vec<i64>| cells.iter().map(|&(c, d)| v[c * ndays + d]).collect();
        write(&dir("campaign_daily"), output::CAMPAIGN_DAILY, vec![
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
}

fn write(path: &Path, schema: &str, columns: Vec<Column>) {
    output::write(path, schema, columns).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}
