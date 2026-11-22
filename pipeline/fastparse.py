"""
Compiled CSV parsing for the pipeline (Numba). Implements the rules.md CSV
dialect and timestamp forms, straight from bytes into Arrow columns.

Each file is read whole into a uint8 array and split on \\n only, so a \\r is
ordinary data. UTF-8 is safe to scan byte by byte: every byte of a multi-byte
character is >= 0x80, so none can be mistaken for a comma, quote or newline.

All kernels are cached (cache=True). The Docker build compiles them once via
warmup(), so the timed run loads machine code instead of compiling.
"""
import os
import tempfile

import numba as nb
import numpy as np
import pyarrow as pa

NULL = np.iinfo(np.int64).min  # "no value" in int64 outputs, turned into Arrow nulls
DAY = 86_400_000
QUOTE, COMMA, NEWLINE = 34, 44, 10


# --- Small helpers ---------------------------------------------------------

@nb.njit(cache=True, inline="always")
def _digit(buf, p):
    """ASCII digit value at p, or -1. ('٢' or '２' are multi-byte, so never match.)"""
    v = np.int64(buf[p]) - 48
    return v if 0 <= v <= 9 else -1


@nb.njit(cache=True, inline="always")
def _two(buf, p):
    a = _digit(buf, p)
    b = _digit(buf, p + 1)
    return -1 if a < 0 or b < 0 else a * 10 + b


@nb.njit(cache=True)
def _days_from_civil(y, m, d):
    """Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm)."""
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


@nb.njit(cache=True)
def _digits(buf, s, e, max_sig):
    """Value of a field of ASCII digits only, or -1 if it is empty, has any other
    character, or has more than max_sig significant digits (leading zeros are fine)."""
    if s >= e:
        return -1
    for i in range(s, e):
        if _digit(buf, i) < 0:
            return -1
    while s < e and buf[s] == 48:
        s += 1
    if e - s > max_sig:
        return -1
    v = np.int64(0)
    for i in range(s, e):
        v = v * 10 + (np.int64(buf[i]) - 48)
    return v


@nb.njit(cache=True)
def _ts(buf, s, e):
    """A ts field in form F1, F2 or F3 -> epoch ms, or NULL when bad.

    F3 values over 18 significant digits can't fit in an int64 and are far
    outside any window, so they are bad too.
    """
    n = e - s
    if (n == 20 or n == 25) and buf[s + 4] == 45 and buf[s + 7] == 45 and buf[s + 10] == 84 \
            and buf[s + 13] == 58 and buf[s + 16] == 58:
        hi, lo = _two(buf, s), _two(buf, s + 2)
        y = hi * 100 + lo if hi >= 0 and lo >= 0 else -1
        mo, d = _two(buf, s + 5), _two(buf, s + 8)
        h, mi, se = _two(buf, s + 11), _two(buf, s + 14), _two(buf, s + 17)
        if y < 0 or mo < 1 or mo > 12 or d < 1 or h < 0 or h > 23 or mi < 0 or mi > 59 or se < 0 or se > 59:
            return NULL
        leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
        dim = 31
        if mo == 2:
            dim = 29 if leap else 28
        elif mo == 4 or mo == 6 or mo == 9 or mo == 11:
            dim = 30
        if d > dim:
            return NULL
        off = np.int64(0)
        if n == 20:
            if buf[s + 19] != 90:  # 'Z', never 'z'
                return NULL
        else:
            c = buf[s + 19]
            if c == 43:
                sign = 1
            elif c == 45:
                sign = -1
            else:
                return NULL
            oh, om = _two(buf, s + 20), _two(buf, s + 23)
            if buf[s + 22] != 58 or oh < 0 or oh > 23 or om < 0 or om > 59:
                return NULL
            off = sign * (oh * 3_600_000 + om * 60_000)
        return _days_from_civil(y, mo, d) * DAY + h * 3_600_000 + mi * 60_000 + se * 1000 - off
    v = _digits(buf, s, e, 18)
    return NULL if v < 0 else v


@nb.njit(cache=True, inline="always")
def _is(buf, s, e, word):
    if e - s != len(word):
        return False
    for t in range(len(word)):
        if buf[s + t] != word[t]:
            return False
    return True


VIEW = np.frombuffer(b"view", dtype=np.uint8)
CART = np.frombuffer(b"cart", dtype=np.uint8)
PURCHASE = np.frombuffer(b"purchase", dtype=np.uint8)


# --- Kernels ---------------------------------------------------------------

@nb.njit(cache=True, parallel=True)
def _newlines(buf, nchunks):
    """Positions of every \\n, plus len(buf) when the file doesn't end with one.
    Two parallel passes over chunks: count, then write at each chunk's offset."""
    n = len(buf)
    size = (n + nchunks - 1) // nchunks
    counts = np.zeros(nchunks + 1, np.int64)
    for c in nb.prange(nchunks):
        k = 0
        for i in range(c * size, min(n, (c + 1) * size)):
            if buf[i] == NEWLINE:
                k += 1
        counts[c + 1] = k
    for c in range(nchunks):
        counts[c + 1] += counts[c]
    trailing = n > 0 and buf[n - 1] != NEWLINE
    nl = np.empty(counts[nchunks] + (1 if trailing else 0), np.int64)
    for c in nb.prange(nchunks):
        k = counts[c]
        for i in range(c * size, min(n, (c + 1) * size)):
            if buf[i] == NEWLINE:
                nl[k] = i
                k += 1
    if trailing:
        nl[len(nl) - 1] = n
    return nl


@nb.njit(cache=True, parallel=True)
def _split(buf, nl, required, bounds, valid):
    """Split each line into fields; bounds[i, 2f:2f+2] = [start, end) of field f, quotes excluded.

    Line i >= 1 is buf[nl[i-1]+1 : nl[i]] (line 0 is the header). valid[i] is set
    when the line has exactly nf fields, every quote wraps a whole field, and
    every required field is non-empty.
    """
    nf = len(required)
    for i in nb.prange(1, len(nl)):
        e = nl[i]
        p = nl[i - 1] + 1
        k = 0
        good = True
        while True:
            if p < e and buf[p] == QUOTE:
                q = p + 1
                while q < e and buf[q] != QUOTE and buf[q] != COMMA:
                    q += 1
                if q >= e or buf[q] != QUOTE:
                    good = False
                    break
                cs, ce, p = p + 1, q, q + 1
            else:
                q = p
                while q < e and buf[q] != COMMA and buf[q] != QUOTE:
                    q += 1
                if q < e and buf[q] == QUOTE:
                    good = False
                    break
                cs, ce, p = p, q, q
            bounds[i, 2 * k] = cs
            bounds[i, 2 * k + 1] = ce
            k += 1
            if p == e:
                break
            if buf[p] != COMMA or k == nf:
                good = False
                break
            p += 1
        if not good or k != nf:
            continue
        for f in range(nf):
            if required[f] and bounds[i, 2 * f + 1] == bounds[i, 2 * f]:
                good = False
        valid[i] = good


@nb.njit(cache=True, parallel=True)
def _event_values(buf, idx, b, ts, ingest, is_purchase, price, ok):
    """Typed event fields for the kept lines idx. Fields: 0 event_id, 1 user_id,
    2 ts, 3 ingest_ts, 4 event_type, 5 product_id, 6 price_cents."""
    for j in nb.prange(len(idx)):
        i = idx[j]
        ts[j] = _ts(buf, b[i, 4], b[i, 5])
        ingest[j] = _ts(buf, b[i, 6], b[i, 7])
        p = _is(buf, b[i, 8], b[i, 9], PURCHASE)
        known = p or _is(buf, b[i, 8], b[i, 9], VIEW) or _is(buf, b[i, 8], b[i, 9], CART)
        is_purchase[j] = p
        price[j] = _digits(buf, b[i, 12], b[i, 13], 8) if p else -1
        ok[j] = known and b[i, 3] > b[i, 2] and b[i, 11] > b[i, 10]


@nb.njit(cache=True, parallel=True)
def _click_values(buf, idx, b, ts):
    """Fields: 0 click_id, 1 user_id, 2 campaign_id, 3 ts."""
    for j in nb.prange(len(idx)):
        i = idx[j]
        ts[j] = _ts(buf, b[i, 6], b[i, 7])


@nb.njit(cache=True, parallel=True)
def _offsets(idx, b, cols):
    """Arrow offsets for the string columns: offs[c, j+1] - offs[c, j] = length of field cols[c] in row j."""
    m = len(idx)
    offs = np.zeros((len(cols), m + 1), np.int64)
    for j in nb.prange(m):
        i = idx[j]
        for c in range(len(cols)):
            offs[c, j + 1] = b[i, 2 * cols[c] + 1] - b[i, 2 * cols[c]]
    for c in range(len(cols)):
        for j in range(m):
            offs[c, j + 1] += offs[c, j]
    return offs


@nb.njit(cache=True, parallel=True)
def _gather(buf, idx, b, cols, offs, datas):
    """Copy the bytes of each string field into its column's data buffer."""
    for j in nb.prange(len(idx)):
        i = idx[j]
        for c in range(len(cols)):
            s = b[i, 2 * cols[c]]
            o = offs[c, j]
            for t in range(b[i, 2 * cols[c] + 1] - s):
                datas[c, o + t] = buf[s + t]


# --- Python side ------------------------------------------------------------

def _nchunks():
    return nb.get_num_threads() * 4


def _strings(buf, idx, bounds, cols):
    """Arrow large_string columns for fields cols of the kept lines."""
    cols = np.asarray(cols, dtype=np.int64)
    m = len(idx)
    offs = _offsets(idx, bounds, cols)
    datas = np.empty((len(cols), max(1, int(offs[:, -1].max()))), dtype=np.uint8)
    _gather(buf, idx, bounds, cols, offs, datas)
    return [pa.LargeStringArray.from_buffers(m, pa.py_buffer(offs[c]), pa.py_buffer(datas[c, :offs[c, -1]]))
            for c in range(len(cols))]


def _split_file(path, required):
    """(bytes, kept line numbers, field bounds) of one file, or None for an empty file."""
    if os.path.getsize(path) == 0:
        return None
    buf = np.fromfile(path, dtype=np.uint8)  # one read; cheaper than faulting in a memmap page by page
    nl = _newlines(buf, _nchunks())
    required = np.asarray(required, dtype=np.bool_)
    bounds = np.zeros((len(nl), 2 * len(required)), dtype=np.int64)
    valid = np.zeros(len(nl), dtype=np.bool_)
    _split(buf, nl, required, bounds, valid)
    return buf, np.flatnonzero(valid), bounds


def _nullable(values, missing):
    return pa.array(values, mask=missing)


def _concat(tables, schema):
    return pa.concat_tables(tables) if tables else schema.empty_table()


LINKS_SCHEMA = pa.schema([("a", pa.large_string()), ("b", pa.large_string())])
EVENTS_SCHEMA = pa.schema([("event_id", pa.large_string()), ("user_id", pa.large_string()),
                           ("product_id", pa.large_string()), ("ts", pa.int64()), ("ingest_ms", pa.int64()),
                           ("is_purchase", pa.bool_()), ("price", pa.int32()), ("ok", pa.bool_())])
CLICKS_SCHEMA = pa.schema([("click_id", pa.large_string()), ("user_id", pa.large_string()),
                           ("campaign_id", pa.large_string()), ("ts", pa.int64())])


def read_links(paths):
    """Well-formed links with both sides non-empty: columns a, b."""
    tables = []
    for path in paths:
        r = _split_file(path, [True, True])
        if r is not None:
            buf, idx, b = r
            a, bb = _strings(buf, idx, b, [0, 1])
            tables.append(pa.table([a, bb], schema=LINKS_SCHEMA))
    return _concat(tables, LINKS_SCHEMA)


def read_events(paths):
    """Every well-formed copy of every event with a non-empty event_id, typed.

    ts and ingest_ms are null when bad; price is null unless the event is a
    purchase whose price is digits only (range checked later); ok holds the
    remaining field rules: known event_type, non-empty user_id and product_id.
    """
    tables = []
    for path in paths:
        r = _split_file(path, [True, False, False, False, False, False, False])
        if r is None:
            continue
        buf, idx, b = r
        m = len(idx)
        ts, ingest, price = np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m, np.int64)
        is_purchase, ok = np.empty(m, np.bool_), np.empty(m, np.bool_)
        _event_values(buf, idx, b, ts, ingest, is_purchase, price, ok)
        event_id, user_id, product_id = _strings(buf, idx, b, [0, 1, 5])
        tables.append(pa.table([event_id, user_id, product_id,
                                _nullable(ts, ts == NULL), _nullable(ingest, ingest == NULL),
                                pa.array(is_purchase), _nullable(price.astype(np.int32), price < 0),
                                pa.array(ok)], schema=EVENTS_SCHEMA))
    return _concat(tables, EVENTS_SCHEMA)


def read_clicks(paths):
    """Well-formed clicks with non-empty click_id and user_id; ts null when bad."""
    tables = []
    for path in paths:
        r = _split_file(path, [True, True, False, False])
        if r is None:
            continue
        buf, idx, b = r
        ts = np.empty(len(idx), np.int64)
        _click_values(buf, idx, b, ts)
        click_id, user_id, campaign_id = _strings(buf, idx, b, [0, 1, 2])
        tables.append(pa.table([click_id, user_id, campaign_id, _nullable(ts, ts == NULL)], schema=CLICKS_SCHEMA))
    return _concat(tables, CLICKS_SCHEMA)


def warmup():
    """Compile (and cache) every kernel by parsing tiny sample files. Run at image build time."""
    with tempfile.TemporaryDirectory() as d:
        def write(name, text):
            path = os.path.join(d, name)
            with open(path, "w") as f:
                f.write(text)
            return path
        read_links([write("l.csv", "user_id_a,user_id_b\nu1,u2\n")])
        read_events([write("e.csv", "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n"
                                    "e1,u1,2026-10-03T10:00:00Z,2026-10-03T10:00:00Z,purchase,p1,100\n")])
        read_clicks([write("c.csv", "click_id,user_id,campaign_id,ts\nk1,u1,c_1,1791071890000")])


if __name__ == "__main__":
    warmup()
