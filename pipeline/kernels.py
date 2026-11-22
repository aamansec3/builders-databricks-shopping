"""
Compiled kernels for the pipeline (Numba). pipeline.py calls them in order;
each one is a single pass over plain NumPy arrays, parallel over chunks of
rows or over partitions.

Strings are never copied out of the input: a field is a (start, length) pair
into the one buffer that holds every CSV file. UTF-8 is safe to scan byte by
byte: every byte of a multi-byte character is >= 0x80, so none can be mistaken
for a comma, quote or newline, and byte order is the rules' byte-wise order.

Grouping equal strings (event dedup, user IDs) works in two steps: hash every
string and partition the rows by the top bits of the hash, so equal strings
always share a partition; then each partition, on its own thread, builds an
exact hash table that compares the bytes, so hash collisions can't merge two
different strings.

All kernels are cached (cache=True). The Docker build compiles them once by
running the pipeline on a tiny dataset, so the timed run loads machine code.
"""
import numba as nb
import numpy as np

NULL = np.iinfo(np.int64).min  # a bad timestamp
BIG = np.iinfo(np.int64).max
DAY = 86_400_000
GAP = 1_800_000
WEEK = 7 * DAY
QUOTE, COMMA, NEWLINE = 34, 44, 10
VIEW, CART, PURCHASE = 1, 2, 3  # event_type codes; 0 is anything else
GROUP_BITS = 10
GROUP_PARTS = 1 << GROUP_BITS  # hash partitions for grouping strings

VIEW_W = np.frombuffer(b"view", dtype=np.uint8)
CART_W = np.frombuffer(b"cart", dtype=np.uint8)
PURCHASE_W = np.frombuffer(b"purchase", dtype=np.uint8)


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


@nb.njit(cache=True, inline="always")
def _hash(buf, s, n):
    """64-bit hash of buf[s:s+n]: FNV-1a, then a mixer so the top bits are good for partitioning."""
    h = np.uint64(0xCBF29CE484222325)
    for i in range(s, s + n):
        h = (h ^ np.uint64(buf[i])) * np.uint64(0x100000001B3)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> np.uint64(33)
    return h


@nb.njit(cache=True, inline="always")
def _eq(a, sa, na, b, sb, nb_):
    if na != nb_:
        return False
    for t in range(na):
        if a[sa + t] != b[sb + t]:
            return False
    return True


@nb.njit(cache=True, inline="always")
def _cmp(buf, s1, n1, s2, n2):
    """Negative, 0 or positive as buf[s1:s1+n1] sorts byte-wise before, equal to or after buf[s2:s2+n2]."""
    for t in range(min(n1, n2)):
        a = buf[s1 + t]
        b = buf[s2 + t]
        if a != b:
            return -1 if a < b else 1
    return n1 - n2


@nb.njit(cache=True, inline="always")
def _fields(buf, s, e, fb, nf):
    """Split line buf[s:e] into fields; fb[2f:2f+2] = [start, end) of field f, quotes excluded.
    True when the line has exactly nf fields and every quote wraps a whole field."""
    p = s
    k = 0
    while True:
        if p < e and buf[p] == QUOTE:
            q = p + 1
            while q < e and buf[q] != QUOTE and buf[q] != COMMA:
                q += 1
            if q >= e or buf[q] != QUOTE:
                return False
            cs, ce, p = p + 1, q, q + 1
        else:
            q = p
            while q < e and buf[q] != COMMA and buf[q] != QUOTE:
                q += 1
            if q < e and buf[q] == QUOTE:
                return False
            cs, ce, p = p, q, q
        if k == nf:
            return False
        fb[2 * k] = cs
        fb[2 * k + 1] = ce
        k += 1
        if p == e:
            return k == nf
        if buf[p] != COMMA:
            return False
        p += 1


CHUNKS = 256  # row chunks per parallel loop: plenty for 8 threads to share evenly


@nb.njit(cache=True, inline="always")
def _chunks(n):
    """(number of chunks, chunk size) for splitting n rows across the threads.
    (A fixed count: a kernel that calls nb.get_num_threads() can't be cached.)"""
    nch = max(1, min(n, CHUNKS))
    return nch, (n + nch - 1) // nch


# --- Lookup tables for products.csv and campaigns.csv ----------------------

@nb.njit(cache=True)
def table(dbuf, ds, dn):
    """Open-addressing hash table over unique keys dbuf[ds[k]:ds[k]+dn[k]]: slot -> k, or -1."""
    m = len(ds)
    size = 2
    while size < 2 * m:
        size *= 2
    tab = np.full(size, -1, np.int64)
    mask = np.uint64(size - 1)
    for k in range(m):
        slot = np.int64(_hash(dbuf, ds[k], dn[k]) & mask)
        while tab[slot] >= 0:
            slot = (slot + 1) & (size - 1)
        tab[slot] = k
    return tab


@nb.njit(cache=True, inline="always")
def _find(tab, dbuf, ds, dn, buf, s, n):
    """Index of the key equal to buf[s:s+n], or -1."""
    size = len(tab)
    slot = np.int64(_hash(buf, s, n) & np.uint64(size - 1))
    while True:
        k = tab[slot]
        if k < 0:
            return -1
        if _eq(dbuf, ds[k], dn[k], buf, s, n):
            return k
        slot = (slot + 1) & (size - 1)


# --- Parsing ---------------------------------------------------------------

ONES = np.uint64(0x0101010101010101)
LOW7 = np.uint64(0x7F7F7F7F7F7F7F7F)


@nb.njit(cache=True, inline="always")
def _newline_bits(word):
    """The top bit of each byte of word that is \\n, and no other bits (exact, no false hits)."""
    v = word ^ (ONES * np.uint64(NEWLINE))
    return ~(((v & LOW7) + LOW7) | v | LOW7)


@nb.njit(cache=True, inline="always")
def _newline_count(bits):
    return ((bits >> np.uint64(7)) * ONES) >> np.uint64(56)


@nb.njit(cache=True, parallel=True)
def newlines(buf):
    """Positions of every \\n. Two parallel passes over chunks: count, then write at
    each chunk's offset. Both look at 8 bytes at a time; a line is ~70 bytes, so
    most words hold no newline and are skipped with one test."""
    words = buf.view(np.uint64)  # buf's length is a multiple of 8 (see pipeline.load)
    n = len(words)
    nch, size = _chunks(n)
    counts = np.zeros(nch + 1, np.int64)
    for c in nb.prange(nch):
        found = np.uint64(0)
        for w in range(c * size, min(n, (c + 1) * size)):
            found += _newline_count(_newline_bits(words[w]))
        counts[c + 1] = found
    for c in range(nch):
        counts[c + 1] += counts[c]
    nl = np.empty(counts[nch], np.int64)
    for c in nb.prange(nch):
        k = counts[c]
        for w in range(c * size, min(n, (c + 1) * size)):
            if _newline_bits(words[w]) != 0:
                for i in range(8 * w, 8 * w + 8):
                    if buf[i] == NEWLINE:
                        nl[k] = i
                        k += 1
    return nl


@nb.njit(cache=True, parallel=True)
def parse_links(buf, nl, header, lo, hi, a_s, a_n, b_s, b_n, ok):
    """Lines lo..hi of links: ok when well-formed with both IDs non-empty."""
    n = hi - lo
    nch, size = _chunks(n)
    for c in nb.prange(nch):
        fb = np.empty(4, np.int64)
        for j in range(c * size, min(n, (c + 1) * size)):
            i = lo + j
            s = nl[i - 1] + 1 if i > 0 else 0
            good = not header[i] and _fields(buf, s, nl[i], fb, 2) and fb[1] > fb[0] and fb[3] > fb[2]
            ok[j] = good
            a_s[j], a_n[j] = fb[0], fb[1] - fb[0]
            b_s[j], b_n[j] = fb[2], fb[3] - fb[2]


@nb.njit(cache=True, parallel=True)
def parse_events(buf, nl, header, lo, hi, eid_s, eid_n, eid_h, uid_s, uid_n, pid_s, pid_n,
                 ts, ingest, typ, price, ok):
    """Lines lo..hi of events. ok when well-formed with a non-empty event_id;
    the other field rules are checked on the kept copy, after dedup.

    Fields: 0 event_id, 1 user_id, 2 ts, 3 ingest_ts, 4 event_type, 5 product_id,
    6 price_cents. price is -1 unless the event is a purchase whose price is
    digits only with at most 8 significant digits (range checked later).
    """
    n = hi - lo
    nch, size = _chunks(n)
    for c in nb.prange(nch):
        fb = np.empty(14, np.int64)
        for j in range(c * size, min(n, (c + 1) * size)):
            i = lo + j
            s = nl[i - 1] + 1 if i > 0 else 0
            good = not header[i] and _fields(buf, s, nl[i], fb, 7) and fb[1] > fb[0]
            ok[j] = good
            if not good:
                continue
            eid_s[j], eid_n[j] = fb[0], fb[1] - fb[0]
            eid_h[j] = _hash(buf, fb[0], fb[1] - fb[0])
            uid_s[j], uid_n[j] = fb[2], fb[3] - fb[2]
            ts[j] = _ts(buf, fb[4], fb[5])
            ingest[j] = _ts(buf, fb[6], fb[7])
            t = 0
            if _is(buf, fb[8], fb[9], VIEW_W):
                t = VIEW
            elif _is(buf, fb[8], fb[9], CART_W):
                t = CART
            elif _is(buf, fb[8], fb[9], PURCHASE_W):
                t = PURCHASE
            typ[j] = t
            pid_s[j], pid_n[j] = fb[10], fb[11] - fb[10]
            price[j] = _digits(buf, fb[12], fb[13], 8) if t == PURCHASE else -1


@nb.njit(cache=True, parallel=True)
def parse_clicks(buf, nl, header, lo, hi, ts_lo, ts_hi, ctab, cbuf, cs, cn,
                 cid_s, cid_n, uid_s, uid_n, ts, camp, ok):
    """Lines lo..hi of clicks. ok when the click is valid: well-formed, non-empty
    click_id and user_id, ts in [ts_lo, ts_hi), campaign in campaigns.csv.

    Copies of a click are identical, so they are not deduped: a copy can't change
    which campaign a session credits, or the set of campaigns that assist it.
    """
    n = hi - lo
    nch, size = _chunks(n)
    for c in nb.prange(nch):
        fb = np.empty(8, np.int64)
        for j in range(c * size, min(n, (c + 1) * size)):
            i = lo + j
            s = nl[i - 1] + 1 if i > 0 else 0
            good = not header[i] and _fields(buf, s, nl[i], fb, 4) and fb[1] > fb[0] and fb[3] > fb[2]
            if good:
                t = _ts(buf, fb[6], fb[7])
                good = ts_lo <= t < ts_hi
                ts[j] = t
            if good:
                k = _find(ctab, cbuf, cs, cn, buf, fb[4], fb[5] - fb[4])
                good = k >= 0
                camp[j] = k
            ok[j] = good
            cid_s[j], cid_n[j] = fb[0], fb[1] - fb[0]
            uid_s[j], uid_n[j] = fb[2], fb[3] - fb[2]


# --- Grouping equal strings --------------------------------------------------

@nb.njit(cache=True, parallel=True)
def hash_parts(h, use):
    """Hash partition of each row, or GROUP_PARTS (dropped) when not use."""
    part = np.empty(len(h), np.int64)
    for i in nb.prange(len(h)):
        part[i] = np.int64(h[i] >> np.uint64(64 - GROUP_BITS)) if use[i] else GROUP_PARTS
    return part


@nb.njit(cache=True, parallel=True)
def partition(part, nparts):
    """Stable counting sort of row numbers by part[i] in 0..nparts-1 (nparts = dropped).
    order[off[p]:off[p+1]] are the rows of partition p, in their original order."""
    n = len(part)
    nch, size = _chunks(n)
    hist = np.zeros((nch, nparts + 1), np.int64)
    for c in nb.prange(nch):
        for i in range(c * size, min(n, (c + 1) * size)):
            hist[c, part[i]] += 1
    pos = np.empty((nch, nparts + 1), np.int64)
    off = np.zeros(nparts + 2, np.int64)
    run = 0
    for p in range(nparts + 1):
        off[p] = run
        for c in range(nch):
            pos[c, p] = run
            run += hist[c, p]
    off[nparts + 1] = run
    order = np.empty(n, np.int64)
    for c in nb.prange(nch):
        for i in range(c * size, min(n, (c + 1) * size)):
            p = part[i]
            order[pos[c, p]] = i
            pos[c, p] += 1
    return order, off


@nb.njit(cache=True, parallel=True)
def group(buf, s, n, h, order, off):
    """Number the distinct strings buf[s[i]:s[i]+n[i]] of the rows partitioned by
    hash_parts. Each partition first copies its rows' hashes into a local array
    (independent loads the CPU overlaps), then probes a small table over that.
    Bytes are compared only when two hashes are equal.

    Returns gidp, the group of each row in partition order (gidp[j] is the
    group of row order[j]), rep[g] = the first row of group g, and goff
    (partition p's groups are goff[p]..goff[p+1]).
    """
    nparts = len(off) - 2
    gidp = np.empty(off[nparts], np.int64)
    goff = np.zeros(nparts + 1, np.int64)
    for p in nb.prange(nparts):
        lo = off[p]
        m = off[p + 1] - lo
        hl = np.empty(m, np.uint64)
        sl = np.empty(m, np.int64)
        nl_ = np.empty(m, np.int64)
        for j in range(m):
            r = order[lo + j]
            hl[j], sl[j], nl_[j] = h[r], s[r], n[r]
        size = 2
        while size < 2 * m:
            size *= 2
        tab = np.full(size, -1, np.int32)  # slot -> local group
        gh = np.empty(m, np.uint64)  # each local group's hash and string
        gs = np.empty(m, np.int64)
        gn = np.empty(m, np.int64)
        mask = np.uint64(size - 1)
        k = 0
        for j in range(m):
            slot = np.int64(hl[j] & mask)
            while True:
                g = tab[slot]
                if g < 0:
                    tab[slot] = k
                    gh[k], gs[k], gn[k] = hl[j], sl[j], nl_[j]
                    gidp[lo + j] = k
                    k += 1
                    break
                if gh[g] == hl[j] and _eq(buf, gs[g], gn[g], buf, sl[j], nl_[j]):
                    gidp[lo + j] = g
                    break
                slot = (slot + 1) & (size - 1)
        goff[p + 1] = k
    for p in range(nparts):
        goff[p + 1] += goff[p]
    rep = np.empty(goff[nparts], np.int64)
    for p in nb.prange(nparts):
        g = goff[p]  # groups are numbered in order of first appearance
        for j in range(off[p], off[p + 1]):
            gidp[j] += goff[p]
            if gidp[j] == g:
                rep[g] = order[j]
                g += 1
    return gidp, rep, goff


@nb.njit(cache=True, parallel=True)
def ungroup(gidp, order, n):
    """The group of each of n rows (-1 for dropped rows) from group's partition-order gidp."""
    gid = np.full(n, -1, np.int64)
    for j in nb.prange(len(gidp)):
        gid[order[j]] = gidp[j]
    return gid


# --- Events: dedup and validate ----------------------------------------------

@nb.njit(cache=True, parallel=True)
def earliest(gidp, order, off, goff, ingest, chosen):
    """chosen[r] for the copy r of each event with the smallest ingest_ts. Copies
    with equal ingest_ts are identical, so any of them will do; a bad ingest_ts
    never wins over a good one."""
    nparts = len(off) - 2
    for p in nb.prange(nparts):
        g0 = goff[p]
        ng = goff[p + 1] - g0
        best = np.full(ng, -1, np.int64)
        best_key = np.empty(ng, np.int64)
        for j in range(off[p], off[p + 1]):
            r = order[j]
            key = BIG if ingest[r] == NULL else ingest[r]
            g = gidp[j] - g0
            if best[g] < 0 or key < best_key[g]:
                best[g] = r
                best_key[g] = key
        for g in range(ng):
            chosen[best[g]] = True


@nb.njit(cache=True, parallel=True)
def validate(chosen, buf, uid_n, ts, ingest, typ, price, pid_s, pid_n, ts_lo, ts_hi,
             ptab, pbuf, ps, pn, pcat, unknown, keep, cat):
    """keep[r] for each chosen copy r that passes the field rules and isn't late,
    and cat[r], its category index (unknown when the product isn't in products.csv).
    Rows go in file order, so the arrays and the buffer are read front to back."""
    for r in nb.prange(len(chosen)):
        if not chosen[r]:
            continue
        t = ts[r]
        if uid_n[r] == 0 or pid_n[r] == 0 or typ[r] == 0 or t < ts_lo or t >= ts_hi:
            continue
        if typ[r] == PURCHASE and (price[r] < 1 or price[r] > 10_000_000):
            continue
        if ingest[r] == NULL or ingest[r] - t > DAY:
            continue
        keep[r] = True
        k = _find(ptab, pbuf, ps, pn, buf, pid_s[r], pid_n[r])
        cat[r] = pcat[k] if k >= 0 else unknown


# --- People ------------------------------------------------------------------

@nb.njit(cache=True, inline="always")
def _item(buf, s0, n0, use, i, s, n, h, part):
    s[i] = s0
    n[i] = n0
    if use:
        hv = _hash(buf, s0, n0)
        h[i] = hv
        part[i] = np.int64(hv >> np.uint64(64 - GROUP_BITS))
    else:
        part[i] = GROUP_PARTS


@nb.njit(cache=True, parallel=True)
def user_items(buf, a_s, a_n, b_s, b_n, lok, eu_s, eu_n, keep, cu_s, cu_n, cok):
    """Every user ID to number, as rows: both sides of each link (rows 2j, 2j+1),
    then each kept event, then each valid click. Unused rows are dropped."""
    nl_, ne, nc = len(lok), len(keep), len(cok)
    m = 2 * nl_ + ne + nc
    s = np.empty(m, np.int64)
    n = np.empty(m, np.int64)
    h = np.empty(m, np.uint64)
    part = np.empty(m, np.int64)
    for j in nb.prange(nl_):
        _item(buf, a_s[j], a_n[j], lok[j], 2 * j, s, n, h, part)
        _item(buf, b_s[j], b_n[j], lok[j], 2 * j + 1, s, n, h, part)
    for j in nb.prange(ne):
        _item(buf, eu_s[j], eu_n[j], keep[j], 2 * nl_ + j, s, n, h, part)
    for j in nb.prange(nc):
        _item(buf, cu_s[j], cu_n[j], cok[j], 2 * nl_ + ne + j, s, n, h, part)
    return s, n, h, part


@nb.njit(cache=True, inline="always")
def _root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@nb.njit(cache=True)
def people(gid, lok, rep, s, n, buf):
    """Union-find over the links. person[u] = the person of user u (the root of
    their linked set); name[root] = the row holding the set's byte-wise smallest ID."""
    users = len(rep)
    parent = np.arange(users)
    for j in range(len(lok)):
        if lok[j]:
            a = _root(parent, gid[2 * j])
            b = _root(parent, gid[2 * j + 1])
            if a < b:
                parent[b] = a
            elif b < a:
                parent[a] = b
    person = np.empty(users, np.int64)
    for u in range(users):
        person[u] = _root(parent, u)
    name = rep.copy()
    for u in range(users):
        r = person[u]
        if r != u and _cmp(buf, s[rep[u]], n[rep[u]], s[name[r]], n[name[r]]) < 0:
            name[r] = rep[u]
    return person, name


@nb.njit(cache=True, parallel=True)
def event_keys(gid, person, base, keep, ts, typ, price, cat, ts_lo, shift, nparts):
    """For each kept event (user item base + j): key = person << shift | (ts - ts_lo),
    which sorts by (person, ts); pay = price << 16 | category << 1 | is_purchase;
    and the person's partition (nparts = dropped). Packing these in file order
    means the sessions kernel fetches 2 numbers per event instead of 5."""
    m = len(keep)
    key = np.empty(m, np.int64)
    pay = np.empty(m, np.int64)
    part = np.empty(m, np.int64)
    for j in nb.prange(m):
        if keep[j]:
            p = person[gid[base + j]]
            key[j] = (p << shift) | (ts[j] - ts_lo)
            if typ[j] == PURCHASE:
                pay[j] = (price[j] << 16) | (np.int64(cat[j]) << 1) | 1
            else:
                pay[j] = np.int64(cat[j]) << 1
            part[j] = p % nparts
        else:
            part[j] = nparts
    return key, pay, part


@nb.njit(cache=True, parallel=True)
def click_keys(gid, person, base, ok, ts, ts_lo, shift, nparts):
    """For each valid click (user item base + j): key = person << shift | (ts - ts_lo),
    and the person's partition (nparts = dropped)."""
    m = len(ok)
    key = np.empty(m, np.int64)
    part = np.empty(m, np.int64)
    for j in nb.prange(m):
        if ok[j]:
            p = person[gid[base + j]]
            key[j] = (p << shift) | (ts[j] - ts_lo)
            part[j] = p % nparts
        else:
            part[j] = nparts
    return key, part


# --- Sessions, credit, assists -------------------------------------------------

@nb.njit(cache=True, inline="always")
def _emit(o, p, day, sessions, events, revenue, catrev, u_person, u_day, u_sess, u_events, u_rev, u_top):
    """Write one user_daily row at o and clear the category sums."""
    top = -1
    for c in range(len(catrev)):
        if catrev[c] > 0 and (top < 0 or catrev[c] > catrev[top]):
            top = c  # categories are numbered in byte order, so a tie keeps the smallest
    catrev[:] = 0
    u_person[o], u_day[o], u_sess[o], u_events[o], u_rev[o], u_top[o] = p, day, sessions, events, revenue, top
    return o + 1


@nb.njit(cache=True, parallel=True)
def sessions(ekey, epay, eorder, eoff, eshift,
             ckey, camp, cid_s, cid_n, buf, corder, coff, cshift,
             ts_lo, ncat, ncamp, day0,
             u_person, u_day, u_sess, u_events, u_rev, u_top, u_count,
             cr_sess, cr_purch, cr_rev, assist):
    """One partition of people per iteration: sort their events and clicks by
    (person, ts), then walk each person's events once. Keys and payloads come
    from event_keys and click_keys.

    A session ends where the gap to the next event is over 30 minutes; equal
    timestamps have gap 0, so they always share a session. As sessions advance,
    two pointers keep the person's clicks with start - 7d <= ts <= start in a
    window, with a count per campaign: the campaigns with a count are the
    assists, and the window's latest click (smallest click_id on a ts tie) gets
    the credit; an empty window credits organic (index ncamp).

    user_daily rows of partition q go to u_*[eoff[q]:eoff[q]+u_count[q]]; the
    campaign_daily sums go to [q, campaign, day - day0] of the cr_* and assist arrays.
    """
    nq = len(eoff) - 2
    for q in nb.prange(nq):
        elo = eoff[q]
        m = eoff[q + 1] - elo
        if m == 0:
            u_count[q] = 0
            continue
        # Sort this partition's events by (person, ts), then copy what the walk
        # needs into local arrays in that order, so the walk reads them front to back.
        rows = eorder[elo:elo + m]
        keys = np.empty(m, np.int64)
        pays = np.empty(m, np.int64)
        for k in range(m):
            keys[k], pays[k] = ekey[rows[k]], epay[rows[k]]
        srt = np.argsort(keys)
        emask = (np.int64(1) << eshift) - 1
        e_pe = np.empty(m, np.int64)
        e_ts = np.empty(m, np.int64)
        e_pur = np.empty(m, np.bool_)
        e_price = np.empty(m, np.int64)
        e_cat = np.empty(m, np.int64)
        for k in range(m):
            key, pay = keys[srt[k]], pays[srt[k]]
            e_pe[k], e_ts[k] = key >> eshift, (key & emask) + ts_lo
            e_pur[k], e_cat[k], e_price[k] = (pay & 1) == 1, (pay >> 1) & 0x7FFF, pay >> 16
        crows = corder[coff[q]:coff[q + 1]]
        mc = len(crows)
        ckeys = np.empty(mc, np.int64)
        for k in range(mc):
            ckeys[k] = ckey[crows[k]]
        srt = np.argsort(ckeys)
        cmask = (np.int64(1) << cshift) - 1
        c_pe = np.empty(mc, np.int64)
        c_ts = np.empty(mc, np.int64)
        c_camp = np.empty(mc, np.int64)
        c_row = np.empty(mc, np.int64)
        for k in range(mc):
            key, r = ckeys[srt[k]], crows[srt[k]]
            c_pe[k], c_ts[k], c_camp[k], c_row[k] = key >> cshift, (key & cmask) + ts_lo - WEEK, camp[r], r

        cnt = np.zeros(ncamp, np.int64)  # clicks in the window, per campaign
        active = np.empty(ncamp, np.int64)  # campaigns with cnt > 0
        slot = np.empty(ncamp, np.int64)  # where each is in active
        catrev = np.zeros(ncat, np.int64)
        o = elo
        i = 0
        ci = 0
        while i < m:
            p = e_pe[i]
            j = i
            while j < m and e_pe[j] == p:
                j += 1
            while ci < mc and c_pe[ci] < p:
                ci += 1
            ce = ci
            while ce < mc and c_pe[ce] == p:
                ce += 1
            lo = ci
            hi = ci
            nact = 0
            day = np.int64(-1)
            d_sess = d_events = d_rev = 0
            k = i
            while k < j:
                start = e_ts[k]
                sday = start // DAY
                if sday != day:
                    if day >= 0:
                        o = _emit(o, p, day, d_sess, d_events, d_rev, catrev,
                                  u_person, u_day, u_sess, u_events, u_rev, u_top)
                    day = sday
                    d_sess = d_events = d_rev = 0
                n_ev = 0
                rev = 0
                purch = False
                prev = start
                while k < j and e_ts[k] - prev <= GAP:
                    n_ev += 1
                    if e_pur[k]:
                        rev += e_price[k]
                        purch = True
                        catrev[e_cat[k]] += e_price[k]
                    prev = e_ts[k]
                    k += 1

                while hi < ce and c_ts[hi] <= start:
                    c = c_camp[hi]
                    if cnt[c] == 0:
                        slot[c] = nact
                        active[nact] = c
                        nact += 1
                    cnt[c] += 1
                    hi += 1
                while lo < hi and c_ts[lo] < start - WEEK:
                    c = c_camp[lo]
                    cnt[c] -= 1
                    if cnt[c] == 0:
                        last = active[nact - 1]
                        active[slot[c]] = last
                        slot[last] = slot[c]
                        nact -= 1
                    lo += 1

                if hi > lo:
                    b = hi - 1
                    x = hi - 2
                    while x >= lo and c_ts[x] == c_ts[b]:
                        rx, rb = c_row[x], c_row[b]
                        if _cmp(buf, cid_s[rx], cid_n[rx], cid_s[rb], cid_n[rb]) < 0:
                            b = x
                        x -= 1
                    c = c_camp[b]
                else:
                    c = ncamp
                d = sday - day0
                cr_sess[q, c, d] += 1
                if purch:
                    cr_purch[q, c, d] += 1
                cr_rev[q, c, d] += rev
                for a in range(nact):
                    assist[q, active[a], d] += 1
                d_sess += 1
                d_events += n_ev
                d_rev += rev
            o = _emit(o, p, day, d_sess, d_events, d_rev, catrev,
                      u_person, u_day, u_sess, u_events, u_rev, u_top)
            for a in range(nact):
                cnt[active[a]] = 0
            i = j
        u_count[q] = o - elo


# --- Output strings -------------------------------------------------------------

@nb.njit(cache=True, parallel=True)
def gather(buf, s, n):
    """Arrow string buffers (offsets, data) for the strings buf[s[j]:s[j]+n[j]]."""
    m = len(s)
    offs = np.zeros(m + 1, np.int64)
    for j in range(m):
        offs[j + 1] = offs[j] + n[j]
    data = np.empty(offs[m], np.uint8)
    for j in nb.prange(m):
        o = offs[j]
        st = s[j]
        for t in range(n[j]):
            data[o + t] = buf[st + t]
    return offs, data
