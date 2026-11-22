#!/usr/bin/env python3
"""
Shopping Sessions pipeline. Implements rules.md.

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)

Python only moves arrays between compiled Numba kernels (kernels.py):
  parse     one file at a time (the next one is read meanwhile): each line into
            typed values, and the IDs still needed later into a compact heap
  events    dedup on event_id (earliest ingest_ts), then validate the kept copy
  people    number every user ID, union-find over the links
  sessions  per person: sessions, user_daily rows, click credit and assists
  write     two Parquet files with pyarrow

`pipeline.py --warmup` runs on a tiny dataset to compile and cache every kernel
(the Docker build does this, so the timed run doesn't compile).
"""
import csv
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", os.cpu_count() or 1))
WARMUP = "--warmup" in sys.argv

# Numba reads these once, at import. Its thread pool would otherwise size
# itself from the host's CPU count. Its compiled-code cache must be writable
# or it is ignored, and /app is read-only at run time, so the cache the image
# build made (see Dockerfile) is copied to scratch. "generic" matches how that
# cache was compiled, whatever CPU the image was built on.
NUMBA_BUILD_CACHE = "/app/numba_cache"
os.environ["NUMBA_NUM_THREADS"] = str(NCPU)
os.environ["NUMBA_CPU_NAME"] = "generic"
if not WARMUP:
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(SCRATCH, "numba_cache")
    try:
        if os.path.isdir(NUMBA_BUILD_CACHE):
            shutil.copytree(NUMBA_BUILD_CACHE, os.environ["NUMBA_CACHE_DIR"], dirs_exist_ok=True)
    except OSError:
        pass  # local runs without /scratch: Numba falls back to compiling

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import kernels as K  # noqa: E402

PERSON_PARTS = 128  # partitions of people for the sessions kernel
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.2f}s] {msg}", file=sys.stderr, flush=True)


def parse_f1(s):
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


# --- Reading -----------------------------------------------------------------

def csv_files(data, kind):
    """(kind, path, size) of each non-empty *.csv file of a kind."""
    paths = sorted(glob.glob(os.path.join(data, kind, "*.csv")))
    return [(kind, p, os.path.getsize(p)) for p in paths if os.path.getsize(p) > 0]


def read_file(path, size):
    """A file's bytes followed by a \\n, so its last line always ends (when the
    file already ends with \\n that adds an empty line, which is malformed and
    dropped anyway), padded with zeros to whole 8-byte words for K.newlines."""
    buf = np.empty(-(-(size + 1) // 8) * 8, dtype=np.uint8)
    buf[size] = K.NEWLINE
    buf[size + 1:] = 0
    view = memoryview(buf)
    fd = os.open(path, os.O_RDONLY)
    try:
        at = 0
        while at < size:
            got = os.preadv(fd, [view[at:size]], at)
            if got <= 0:
                raise IOError(f"short read on {path}")
            at += got
    finally:
        os.close(fd)
    return buf


def stream(files):
    """Yield (kind, bytes) of each file in turn, reading the next file on a
    background thread while the caller parses this one. (The parse kernels
    release the GIL, so the two really run at the same time.)"""
    with ThreadPoolExecutor(1) as pool:
        ahead = pool.submit(read_file, files[0][1], files[0][2]) if files else None
        for i, (kind, _, _) in enumerate(files):
            buf = ahead.result()
            if i + 1 < len(files):
                ahead = pool.submit(read_file, files[i + 1][1], files[i + 1][2])
            yield kind, buf


class Columns:
    """Named arrays that each file appends its rows to. They start sized for a
    generous estimate of the row count: space no row reaches costs address
    space only, not memory. A file that brings more rows makes them grow."""

    def __init__(self, rows, **dtypes):
        self.used = 0
        self.arrays = {name: np.empty(rows, dtype) for name, dtype in dtypes.items()}

    def append(self, rows):
        """Make room for rows more; returns the first new row."""
        cap = len(next(iter(self.arrays.values())))
        if self.used + rows > cap:
            grown = max(2 * cap, self.used + rows)
            for name, a in self.arrays.items():
                b = np.empty(grown, a.dtype)
                b[:self.used] = a[:self.used]
                self.arrays[name] = b
        base = self.used
        self.used += rows
        return base

    def done(self):
        return {name: a[:self.used] for name, a in self.arrays.items()}


def dimension(path):
    """A clean two-column CSV as (key buffer, starts, lengths, keys, values)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    keys = [r[0].encode() for r in rows]
    lens = np.array([len(k) for k in keys], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64) if keys else lens
    dbuf = np.frombuffer(b"".join(keys) + b"\0", dtype=np.uint8)
    return dbuf, starts, lens, [r[0] for r in rows], [r[1] for r in rows]


def strings(buf, s, n):
    offs, data = K.gather(buf, s, n)
    return pa.LargeStringArray.from_buffers(len(s), pa.py_buffer(offs), pa.py_buffer(data))


# --- The pipeline -------------------------------------------------------------

def run(data, out):
    with open(os.path.join(data, "manifest.json")) as f:
        manifest = json.load(f)
    ws, we = parse_f1(manifest["window_start"]), parse_f1(manifest["window_end"])
    log(f"window {manifest['window_start']} .. {manifest['window_end']}")

    # Products and campaigns become lookup tables. Categories are numbered in
    # byte order, so the smallest number wins a top_category tie.
    pbuf, ps, pn, _, pcats = dimension(os.path.join(data, "products.csv"))
    categories = sorted(set(pcats) | {"unknown"}, key=str.encode)
    cat_index = {c: i for i, c in enumerate(categories)}
    pcat = np.array([cat_index[c] for c in pcats], dtype=np.int16)
    ptab = K.table(pbuf, ps, pn)
    cbuf, cs, cn, campaign_ids, channels = dimension(os.path.join(data, "campaigns.csv"))
    ctab = K.table(cbuf, cs, cn)
    ncamp = len(campaign_ids)

    # --- Read and parse, one file at a time. Only the strings still needed
    # later are kept, copied into heap; each file's bytes are dropped once parsed.
    files = csv_files(data, "links") + csv_files(data, "events") + csv_files(data, "clicks")
    size = {k: sum(f[2] for f in files if f[0] == k) for k in ("links", "events", "clicks")}
    heap = np.empty(sum(size.values()) + 8, np.uint8)  # strings never outgrow the input
    hpos = 0
    # Row estimates: an events line is ~95 bytes, a clicks line ~50, a links line ~25.
    links = Columns(size["links"] // 4 + len(files), hs=np.int64, a_n=np.int32, b_n=np.int32, ok=np.bool_)
    events = Columns(size["events"] // 32 + len(files), hs=np.int64, eid_n=np.int32, eid_h=np.uint64,
                     uid_n=np.int32, ts=np.int64, ingest=np.int64, typ=np.int8, price=np.int32,
                     cat=np.int16, ok=np.bool_)
    clicks = Columns(size["clicks"] // 16 + len(files), hs=np.int64, cid_n=np.int32, uid_n=np.int32,
                     ts=np.int64, camp=np.int32, ok=np.bool_)
    for kind, buf in stream(files):
        nl = K.newlines(buf)
        if kind == "links":
            base = links.append(len(nl))
            c = links.arrays
            hpos = K.parse_links(buf, nl, base, heap, hpos, c["hs"], c["a_n"], c["b_n"], c["ok"])
        elif kind == "events":
            base = events.append(len(nl))
            c = events.arrays
            hpos = K.parse_events(buf, nl, ptab, pbuf, ps, pn, pcat, cat_index["unknown"], base, heap, hpos,
                                  c["hs"], c["eid_n"], c["eid_h"], c["uid_n"], c["ts"], c["ingest"],
                                  c["typ"], c["price"], c["cat"], c["ok"])
        else:
            base = clicks.append(len(nl))
            c = clicks.arrays
            hpos = K.parse_clicks(buf, nl, ws - K.WEEK, we, ctab, cbuf, cs, cn, base, heap, hpos,
                                  c["hs"], c["cid_n"], c["uid_n"], c["ts"], c["camp"], c["ok"])
    del buf, nl, c
    L, E, C = links.done(), events.done(), clicks.done()
    del links, events, clicks
    lhs, a_n, b_n, lok = L["hs"], L["a_n"], L["b_n"], L["ok"]
    ehs, eid_n, eid_h, uid_n, eok = E["hs"], E["eid_n"], E["eid_h"], E["uid_n"], E["ok"]
    ts, ingest, typ, price, cat = E["ts"], E["ingest"], E["typ"], E["price"], E["cat"]
    chs, cid_n, cu_n, cts, camp, cok = C["hs"], C["cid_n"], C["uid_n"], C["ts"], C["camp"], C["ok"]
    del L, E, C
    log(f"read and parsed {sum(size.values()) >> 20} MB in {len(files)} files: {hpos >> 20} MB of IDs kept")

    # --- Events: keep the earliest copy of each event_id, then validate it
    order, off = K.partition(K.hash_parts(eid_h, eok), K.GROUP_PARTS)
    gidp, _, goff = K.group(heap, ehs, eid_n, eid_h, order, off)
    chosen = np.zeros(len(ts), np.bool_)
    K.earliest(gidp, order, off, goff, ingest, chosen)
    del order, off, gidp, goff, eid_h, eok
    keep = np.zeros(len(ts), np.bool_)
    K.validate(chosen, uid_n, ts, ingest, typ, price, cat, ws, we, keep)
    del chosen, ingest
    log("events deduped, validated")

    # From here on, only user IDs and click IDs are needed: copy those into
    # small buffers of their own and free the heap.
    cid_buf, cid_s = K.compact(heap, chs, cid_n, cok)
    ubuf, s, n, h, part = K.user_items(heap, lhs, a_n, b_n, lok, ehs, eid_n, uid_n, keep, chs, cid_n, cu_n, cok)
    del heap, lhs, a_n, b_n, ehs, eid_n, uid_n, chs, cu_n

    # --- People: number every user ID, then join linked ones
    order, off = K.partition(part, K.GROUP_PARTS)
    gidp, urep, _ = K.group(ubuf, s, n, h, order, off)
    ugid = K.ungroup(gidp, order, len(s))
    del gidp, order, off, h, part
    person, name = K.people(ugid, lok, urep, s, n, ubuf)
    del urep
    log(f"people: {len(person)} user IDs")

    # --- Sessions, credit, assists
    # Sort keys are person << shift | (ts - earliest ts), and persons < len(person).
    eshift = max(1, we - ws).bit_length()
    cshift = max(1, we - ws + K.WEEK).bit_length()
    if len(person).bit_length() + cshift > 62:
        raise ValueError("window too long for the sort keys")
    ekey, epay, epart = K.event_keys(ugid, person, 2 * len(lok), keep, ts, typ, price, cat, ws, eshift, PERSON_PARTS)
    ckey, cpart = K.click_keys(ugid, person, 2 * len(lok) + len(keep), cok, cts, ws - K.WEEK, cshift, PERSON_PARTS)
    del ugid, typ, price, cat, cts
    eorder, eoff = K.partition(epart, PERSON_PARTS)
    corder, coff = K.partition(cpart, PERSON_PARTS)
    del epart, cpart
    day0 = ws // K.DAY
    ndays = (we - 1) // K.DAY - day0 + 1
    u_person, u_day, u_sess, u_events, u_rev, u_top = (np.empty(len(ts), np.int64) for _ in range(6))
    u_count = np.zeros(PERSON_PARTS, np.int64)
    cr_sess, cr_purch, cr_rev, assist = (np.zeros((PERSON_PARTS, ncamp + 1, ndays), np.int64) for _ in range(4))
    K.sessions(ekey, epay, eorder, eoff, eshift,
               ckey, camp, cid_s, cid_n, cid_buf, corder, coff, cshift,
               ws, len(categories), ncamp, day0,
               u_person, u_day, u_sess, u_events, u_rev, u_top, u_count,
               cr_sess, cr_purch, cr_rev, assist)
    log("sessions, credit, assists")

    # --- Outputs
    os.makedirs(os.path.join(out, "user_daily"), exist_ok=True)
    os.makedirs(os.path.join(out, "campaign_daily"), exist_ok=True)

    sel = np.concatenate([np.arange(eoff[q], eoff[q] + u_count[q]) for q in range(PERSON_PARTS)])
    who = name[u_person[sel]]
    top = u_top[sel]
    user_daily = pa.table({
        "user_id": strings(ubuf, s[who], n[who]),
        "day": pa.array(u_day[sel].astype(np.int32), type=pa.date32()),
        "sessions": u_sess[sel],
        "events": u_events[sel],
        "revenue_cents": u_rev[sel],
        "top_category": pa.array(categories, type=pa.string()).take(pa.array(top, mask=top < 0)),
    })
    pq.write_table(user_daily, os.path.join(out, "user_daily", "part-0.parquet"))
    log(f"wrote user_daily: {len(sel)} rows")

    sess, purch, rev, asst = cr_sess.sum(0), cr_purch.sum(0), cr_rev.sum(0), assist.sum(0)
    c, d = np.nonzero((sess > 0) | (asst > 0))
    campaign_daily = pa.table({
        "campaign_id": pa.array(campaign_ids + ["organic"], type=pa.string()).take(pa.array(c)),
        "channel": pa.array(channels + ["organic"], type=pa.string()).take(pa.array(c)),
        "day": pa.array((day0 + d).astype(np.int32), type=pa.date32()),
        "sessions": sess[c, d],
        "purchasing_sessions": purch[c, d],
        "revenue_cents": rev[c, d],
        "assisted_sessions": asst[c, d],
    })
    pq.write_table(campaign_daily, os.path.join(out, "campaign_daily", "part-0.parquet"))
    log(f"wrote campaign_daily: {len(c)} rows")


def warmup():
    """Run the whole pipeline on a tiny dataset so every kernel is compiled and cached."""
    files = {
        "manifest.json": '{"window_start": "2026-10-01T00:00:00Z", "window_end": "2026-10-15T00:00:00Z"}',
        "products.csv": "product_id,category\np1,shoes\n",
        "campaigns.csv": "campaign_id,channel\nc_1,email\n",
        "links/l.csv": "user_id_a,user_id_b\nu1,u2\n",
        "events/e.csv": "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n"
                        "e1,u2,2026-10-03T10:00:00Z,2026-10-03T10:00:00Z,purchase,p1,100\n"
                        "e2,u1,2026-10-03T10:10:00Z,2026-10-03T10:10:00Z,view,p1,\n",
        "clicks/c.csv": "click_id,user_id,campaign_id,ts\nk1,u1,c_1,2026-10-03T09:00:00Z\n",
    }
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            path = os.path.join(d, "in", name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        run(os.path.join(d, "in"), os.path.join(d, "out"))


if __name__ == "__main__":
    if WARMUP:
        warmup()
    else:
        run(DATA, OUT)
