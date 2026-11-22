#!/usr/bin/env python3
"""
Shopping Sessions pipeline. Implements rules.md.

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)

Python only moves arrays between compiled Numba kernels (kernels.py):
  load      every CSV file into one byte buffer, read in parallel
  parse     each line into typed values and (start, length) pairs into the buffer
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
READ_CHUNK = 64 << 20
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.2f}s] {msg}", file=sys.stderr, flush=True)


def parse_f1(s):
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


# --- Loading ----------------------------------------------------------------

def load(data):
    """Every links, events and clicks file in one uint8 buffer, each followed by
    a \\n so no line runs into the next file (when a file already ends with \\n
    that adds an empty line, which is malformed and dropped anyway).

    Returns the buffer, the newline positions, which lines are headers, and the
    line range [lo, hi) of each kind.
    """
    kinds = ["links", "events", "clicks"]
    reads, starts, ends = [], [], {}
    off = 0
    for k in kinds:
        for path in sorted(glob.glob(os.path.join(data, k, "*.csv"))):
            size = os.path.getsize(path)
            if size == 0:
                continue
            starts.append(off)
            for at in range(0, size, READ_CHUNK):
                reads.append((path, at, off + at, min(READ_CHUNK, size - at)))
            off += size + 1
        ends[k] = off
    buf = np.empty(-(-off // 8) * 8, dtype=np.uint8)  # whole 8-byte words for K.newlines
    buf[off:] = 0
    view = memoryview(buf)

    def read(job):
        path, at, dst, n = job
        fd = os.open(path, os.O_RDONLY)
        try:
            while n > 0:
                got = os.preadv(fd, [view[dst:dst + n]], at)
                if got <= 0:
                    raise IOError(f"short read on {path}")
                at, dst, n = at + got, dst + got, n - got
        finally:
            os.close(fd)

    with ThreadPoolExecutor(NCPU) as pool:
        list(pool.map(read, reads))
    for nxt in starts[1:] + [off]:
        buf[nxt - 1] = K.NEWLINE
    nl = K.newlines(buf)
    header = np.zeros(len(nl), dtype=np.bool_)
    header[np.searchsorted(nl, np.asarray(starts, dtype=np.int64))] = True
    lines, lo = {}, 0
    for k in kinds:
        hi = int(np.searchsorted(nl, ends[k]))
        lines[k] = (lo, hi)
        lo = hi
    return buf, nl, header, lines


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

    buf, nl, header, lines = load(data)
    log(f"loaded {len(buf) >> 20} MB, {len(nl)} lines")

    # --- Parse
    lo, hi = lines["links"]
    a_s, a_n, b_s, b_n = (np.empty(hi - lo, np.int64) for _ in range(4))
    lok = np.empty(hi - lo, np.bool_)
    K.parse_links(buf, nl, header, lo, hi, a_s, a_n, b_s, b_n, lok)

    lo, hi = lines["events"]
    eid_s, eid_n, uid_s, uid_n, pid_s, pid_n, ts, ingest, price = (np.empty(hi - lo, np.int64) for _ in range(9))
    eid_h = np.empty(hi - lo, np.uint64)
    typ = np.empty(hi - lo, np.int8)
    eok = np.empty(hi - lo, np.bool_)
    K.parse_events(buf, nl, header, lo, hi, eid_s, eid_n, eid_h, uid_s, uid_n, pid_s, pid_n,
                   ts, ingest, typ, price, eok)

    lo, hi = lines["clicks"]
    cid_s, cid_n, cu_s, cu_n, cts = (np.empty(hi - lo, np.int64) for _ in range(5))
    camp = np.empty(hi - lo, np.int32)
    cok = np.empty(hi - lo, np.bool_)
    K.parse_clicks(buf, nl, header, lo, hi, ws - K.WEEK, we, ctab, cbuf, cs, cn,
                   cid_s, cid_n, cu_s, cu_n, cts, camp, cok)
    del nl, header
    log("parsed")

    # --- Events: keep the earliest copy of each event_id, then validate it
    order, off = K.partition(K.hash_parts(eid_h, eok), K.GROUP_PARTS)
    gidp, _, goff = K.group(buf, eid_s, eid_n, eid_h, order, off)
    chosen = np.zeros(len(ts), np.bool_)
    K.earliest(gidp, order, off, goff, ingest, chosen)
    del order, off, gidp, goff, eid_s, eid_n, eid_h, eok
    keep = np.zeros(len(ts), np.bool_)
    cat = np.empty(len(ts), np.int16)
    K.validate(chosen, buf, uid_n, ts, ingest, typ, price, pid_s, pid_n, ws, we,
               ptab, pbuf, ps, pn, pcat, cat_index["unknown"], keep, cat)
    del chosen, ingest, pid_s, pid_n
    log("events deduped, validated")

    # --- People: number every user ID, then join linked ones
    s, n, h, part = K.user_items(buf, a_s, a_n, b_s, b_n, lok, uid_s, uid_n, keep, cu_s, cu_n, cok)
    order, off = K.partition(part, K.GROUP_PARTS)
    gidp, urep, _ = K.group(buf, s, n, h, order, off)
    ugid = K.ungroup(gidp, order, len(s))
    del gidp, order, off, h, part
    person, name = K.people(ugid, lok, urep, s, n, buf)
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
    del ugid, uid_s, uid_n, typ, price, cat, cu_s, cu_n, cts
    eorder, eoff = K.partition(epart, PERSON_PARTS)
    corder, coff = K.partition(cpart, PERSON_PARTS)
    del epart, cpart
    day0 = ws // K.DAY
    ndays = (we - 1) // K.DAY - day0 + 1
    u_person, u_day, u_sess, u_events, u_rev, u_top = (np.empty(len(ts), np.int64) for _ in range(6))
    u_count = np.zeros(PERSON_PARTS, np.int64)
    cr_sess, cr_purch, cr_rev, assist = (np.zeros((PERSON_PARTS, ncamp + 1, ndays), np.int64) for _ in range(4))
    K.sessions(ekey, epay, eorder, eoff, eshift,
               ckey, camp, cid_s, cid_n, buf, corder, coff, cshift,
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
        "user_id": strings(buf, s[who], n[who]),
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
