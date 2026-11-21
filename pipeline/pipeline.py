#!/usr/bin/env python3
"""
Shopping Sessions pipeline (DuckDB). Implements rules.md.

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)

Stages, each a DuckDB table:
  parse    Polars splits every CSV line into fields (our own dialect check)
  links -> idmap         union-find over linked IDs: user_id -> person
  events -> ev -> sess   dedup, validate, sessionize
  clicks -> clk          dedup, validate
  credit / assists       latest click per session; campaigns clicked in the 7 days before
"""
import glob
import json
import mmap
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", os.cpu_count() or 1))
MEM_GB = int(os.environ.get("MEM_GB", "8"))

# Polars sizes its pool from the host's CPU count, which in a container can be
# far more than the 8 we get. It reads this variable once, at import.
os.environ["POLARS_MAX_THREADS"] = str(NCPU)

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

DAY = 86_400_000
GAP = 1_800_000
WEEK = 7 * DAY
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.2f}s] {msg}", file=sys.stderr, flush=True)


try:
    os.makedirs(os.path.join(SCRATCH, "duckdb_tmp"), exist_ok=True)
except OSError:
    pass  # local runs without /scratch; DuckDB only needs it when it spills
con = duckdb.connect()
con.execute(f"SET threads = {NCPU}")
# The parsed Polars frames live outside DuckDB's budget, so leave room for them.
con.execute(f"SET memory_limit = '{max(1, int(MEM_GB * 0.6))}GB'")
con.execute(f"SET temp_directory = '{os.path.join(SCRATCH, 'duckdb_tmp')}'")
con.execute("SET preserve_insertion_order = false")
con.execute("SET TimeZone = 'UTC'")


def sql(q, label=None):
    con.execute(q)
    if label:
        log(label)


# --- Manifest -------------------------------------------------------------

with open(os.path.join(DATA, "manifest.json")) as f:
    manifest = json.load(f)


def parse_f1(s):
    from datetime import datetime, timezone
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


WS = parse_f1(manifest["window_start"])
WE = parse_f1(manifest["window_end"])
log(f"window {manifest['window_start']} .. {manifest['window_end']}")

# --- Macros: timestamps, digits -------------------------------------------
# Timestamps become BIGINT epoch ms, or NULL when bad. [0-9] (never \d) keeps
# digits ASCII. The calendar check is a DATE cast, which rejects Feb 30 etc.

sql(r"""
CREATE MACRO ts_iso(s) AS (
    (try_cast(substr(s, 1, 10) AS DATE) - DATE '1970-01-01')::BIGINT * 86400000
    + try_cast(substr(s, 12, 2) AS BIGINT) * 3600000
    + try_cast(substr(s, 15, 2) AS BIGINT) * 60000
    + try_cast(substr(s, 18, 2) AS BIGINT) * 1000
);
CREATE MACRO parse_ts(s) AS (
    CASE
    WHEN regexp_full_match(s, '[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])') THEN
        CASE WHEN length(s) = 20 THEN ts_iso(s)
             ELSE ts_iso(s) - (CASE WHEN substr(s, 20, 1) = '+' THEN 1 ELSE -1 END)
                  * (try_cast(substr(s, 21, 2) AS BIGINT) * 3600000 + try_cast(substr(s, 24, 2) AS BIGINT) * 60000)
        END
    WHEN regexp_full_match(s, '[0-9]+') THEN
        CASE WHEN length(ltrim(s, '0')) <= 18 THEN try_cast(s AS BIGINT) END
    END
);
CREATE MACRO ms_to_day(ms) AS (DATE '1970-01-01' + (ms // 86400000)::INTEGER);
""")

# --- Parsing --------------------------------------------------------------
# Polars reads each file as whole lines (a separator and no quote char that
# never occur), and its Rust string kernels validate and split them about
# twice as fast as DuckDB's. It keeps a \r inside a line but strips one at the
# end, so files containing any \r (never the generator's) are split on \n by
# hand instead. A threaded mmap search finds them without copying the files.

FIELD = r'(?:"[^",]*"|[^",]*)'


def has_cr(path):
    if os.path.getsize(path) == 0:
        return False
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        return m.find(b"\r") != -1


def parse_fields(kind, n):
    """Well-formed lines of `kind` as a Polars frame of n string columns f1..fn, quotes stripped.

    A line is well-formed when it has exactly n fields and every quote wraps a
    whole field; one regex checks both. Empty fields stay '', never null.
    """
    names = [f"f{i}" for i in range(1, n + 1)]
    files = sorted(glob.glob(os.path.join(DATA, kind, "*.csv")))
    with ThreadPoolExecutor(NCPU) as ex:
        cr = list(ex.map(has_cr, files))
    frames = []
    fast = [p for p, c in zip(files, cr) if not c]
    if fast:
        frames.append(pl.scan_csv(fast, separator="\x01", quote_char=None, has_header=True, new_columns=["line"],
                                  schema={"line": pl.String}, infer_schema=False))
    for p, c in zip(files, cr):
        if c:
            with open(p, encoding="utf-8", newline="") as f:
                frames.append(pl.LazyFrame({"line": f.read().split("\n")[1:]}, schema={"line": pl.String}))
    if not frames:
        return pl.DataFrame(schema={c: pl.String for c in names})
    pattern = "^" + FIELD + ("," + FIELD) * (n - 1) + "$"
    return (pl.concat(frames)
            .filter(pl.col("line").str.contains(pattern))
            .select(pl.col("line").str.replace_all('"', "", literal=True)
                    .str.split_exact(",", n - 1).struct.rename_fields(names).alias("s"))
            .unnest("s")
            .collect())


# --- Small dimension tables -----------------------------------------------

sql(f"""CREATE TABLE products AS SELECT * FROM read_csv('{os.path.join(DATA, "products.csv")}',
        columns={{'product_id': 'VARCHAR', 'category': 'VARCHAR'}}, header=true, auto_detect=false)""")
sql(f"""CREATE TABLE campaigns AS SELECT * FROM read_csv('{os.path.join(DATA, "campaigns.csv")}',
        columns={{'campaign_id': 'VARCHAR', 'channel': 'VARCHAR'}}, header=true, auto_detect=false)""")

# --- Links: who is one person ---------------------------------------------

con.register("links_f", parse_fields("links", 2))
sql(f"""
CREATE TABLE links AS
SELECT f1 AS a, f2 AS b FROM links_f
WHERE f1 <> '' AND f2 <> ''
""", "links parsed")
con.unregister("links_f")
sql("""
CREATE TABLE link_ids AS
SELECT id, (row_number() OVER () - 1)::INTEGER AS k
FROM (SELECT a AS id FROM links UNION SELECT b FROM links)
""")
n_ids = con.execute("SELECT count(*) FROM link_ids").fetchone()[0]
edges = con.execute("""
    SELECT ia.k AS ka, ib.k AS kb FROM links
    JOIN link_ids ia ON links.a = ia.id
    JOIN link_ids ib ON links.b = ib.id
""").fetchnumpy()
ka, kb = edges["ka"].astype(np.int32), edges["kb"].astype(np.int32)
graph = coo_matrix((np.ones(len(ka), dtype=np.int8), (ka, kb)), shape=(n_ids, n_ids)).tocsr()
_, labels = connected_components(graph, directed=False)
con.register("comp_arrow", pa.table({"k": np.arange(n_ids, dtype=np.int32), "comp": labels.astype(np.int32)}))
# A person is named by the byte-wise smallest of their IDs (DuckDB compares VARCHAR byte-wise).
sql("""
CREATE TABLE idmap AS
WITH c AS (SELECT link_ids.id, comp_arrow.comp FROM link_ids JOIN comp_arrow USING (k))
SELECT c.id, n.person FROM c JOIN (SELECT comp, min(id) AS person FROM c GROUP BY comp) n USING (comp)
""", f"union-find over {n_ids} linked ids, {len(ka)} links")
con.unregister("comp_arrow")

# --- Events ---------------------------------------------------------------
# One streaming pass: parse each copy into typed values plus an `ok` flag for
# the field rules, then keep the copy with the smallest ingest_ts per event_id.
# Judging every copy before dedup is the same as judging only the kept copy,
# because the flag travels with the copy it belongs to.

con.register("events_f", parse_fields("events", 7))
log("events parsed")
sql(f"""
CREATE TABLE ev AS
WITH parsed AS (
    SELECT f1 AS event_id, f2 AS user_id, f6 AS product_id,
           parse_ts(f3) AS ts, ts_iso(f4) AS ingest_ms,
           f5 = 'purchase' AS is_purchase,
           CASE WHEN f5 = 'purchase' AND regexp_full_match(f7, '[0-9]+') AND length(ltrim(f7, '0')) <= 8
                THEN try_cast(f7 AS INTEGER) END AS price,
           f5 IN ('view', 'cart', 'purchase') AND f2 <> '' AND f6 <> '' AS ok
    FROM events_f
    WHERE f1 <> ''
), kept AS (
    SELECT arg_min(struct_pack(user_id, product_id, ts, ingest_ms, is_purchase, price, ok), ingest_ms) AS e
    FROM parsed GROUP BY event_id
)
SELECT coalesce(m.person, k.e.user_id) AS person, k.e.ts AS ts, k.e.is_purchase AS is_purchase,
       coalesce(p.category, 'unknown') AS category,
       CASE WHEN k.e.is_purchase THEN k.e.price ELSE 0 END AS price
FROM kept k
LEFT JOIN idmap m ON k.e.user_id = m.id
LEFT JOIN products p ON k.e.product_id = p.product_id
WHERE k.e.ok
  AND k.e.ts >= {WS} AND k.e.ts < {WE}
  AND (NOT k.e.is_purchase OR k.e.price BETWEEN 1 AND 10000000)
  AND k.e.ingest_ms - k.e.ts <= {DAY}
""", "events deduped, validated")
con.unregister("events_f")

# --- Sessions -------------------------------------------------------------
# A row starts a session when the gap to the previous row is over 30 min.
# Rows with equal ts have gap 0 to each other, so only one of them can start a
# session, and the RANGE frame of the running max gives all peers the same start.

sql(f"""
CREATE TABLE ev_s AS
SELECT person, is_purchase, category, price,
       max(CASE WHEN prev IS NULL OR ts - prev > {GAP} THEN ts END)
           OVER (PARTITION BY person ORDER BY ts) AS start
FROM (SELECT *, lag(ts) OVER (PARTITION BY person ORDER BY ts) AS prev FROM ev)
""", "sessionized")
sql("DROP TABLE ev")

sql("""
CREATE TABLE sess AS
SELECT person, start, ms_to_day(start) AS day,
       count(*) AS n_events, sum(price) AS revenue, bool_or(is_purchase) AS has_purchase
FROM ev_s GROUP BY person, start
""", "sessions built")

# --- Clicks ---------------------------------------------------------------

con.register("clicks_f", parse_fields("clicks", 4))
sql(f"""
CREATE TABLE clk AS
SELECT DISTINCT coalesce(m.person, c.user_id) AS person, c.click_id, c.campaign_id, c.ts
FROM (
    SELECT f1 AS click_id, f2 AS user_id, f3 AS campaign_id, parse_ts(f4) AS ts
    FROM clicks_f
    WHERE f1 <> '' AND f2 <> ''
) c
LEFT JOIN idmap m ON c.user_id = m.id
WHERE c.ts >= {WS - WEEK} AND c.ts < {WE}
  AND c.campaign_id IN (SELECT campaign_id FROM campaigns)
""", "clicks parsed + validated")
con.unregister("clicks_f")

# --- Credit: the latest click at or before the session start, within 7 days.
# Ties in ts go to the smallest click_id, so collapse each (person, ts) first.

sql(f"""
CREATE TABLE credit AS
WITH latest AS (
    SELECT person, ts, arg_min(campaign_id, click_id) AS campaign_id FROM clk GROUP BY person, ts
)
SELECT s.*, CASE WHEN l.ts IS NOT NULL AND s.start - l.ts <= {WEEK}
                 THEN l.campaign_id ELSE 'organic' END AS campaign_id
FROM sess s ASOF LEFT JOIN latest l ON s.person = l.person AND l.ts <= s.start
""", "credit")

# --- Assists --------------------------------------------------------------
# A click at t assists every session of its person starting in [t, t + 7d].
# Per (person, campaign), merge overlapping intervals so none overlap; then a
# session is assisted by that campaign iff it falls in exactly one interval,
# and counting sessions per interval counts each (session, campaign) once.
# Counting without materializing session x campaign pairs: cut each interval
# into one piece per UTC day, number each person's sessions by start, and
# take (sessions starting <= hi) - (sessions starting < lo) with two ASOF joins.

sql(f"""
CREATE TABLE runs AS
SELECT person, campaign_id, min(ts) AS lo, max(ts) + {WEEK} AS hi
FROM (
    SELECT *, sum(new_run) OVER (PARTITION BY person, campaign_id ORDER BY ts) AS run
    FROM (
        SELECT person, campaign_id, ts,
               CASE WHEN ts - lag(ts) OVER (PARTITION BY person, campaign_id ORDER BY ts) <= {WEEK}
                    THEN 0 ELSE 1 END AS new_run
        FROM clk
    )
)
GROUP BY person, campaign_id, run
""", "click runs merged")

sql(f"""
CREATE TABLE sess_idx AS
SELECT person, start, row_number() OVER (PARTITION BY person ORDER BY start) AS idx FROM sess
""")
sql(f"""
CREATE TABLE pieces AS
WITH days AS (SELECT DISTINCT person, start // {DAY} AS d FROM sess),
cut AS (
    SELECT person, campaign_id, d, greatest(lo, d * {DAY}) AS lo, least(hi, (d + 1) * {DAY} - 1) AS hi
    FROM (SELECT *, unnest(range(greatest(lo, {WS}) // {DAY}, least(hi, {WE} - 1) // {DAY} + 1)) AS d FROM runs)
)
SELECT cut.* FROM cut SEMI JOIN days ON cut.person = days.person AND cut.d = days.d
""", "interval pieces")
sql("""
CREATE TABLE assists AS
WITH a AS (
    SELECT p.*, s.idx AS idx_hi FROM pieces p
    ASOF LEFT JOIN sess_idx s ON p.person = s.person AND s.start <= p.hi
), b AS (
    SELECT a.*, s.idx AS idx_lo FROM a
    ASOF LEFT JOIN sess_idx s ON a.person = s.person AND s.start < a.lo
)
SELECT campaign_id, ms_to_day(d * 86400000) AS day,
       sum(coalesce(idx_hi, 0) - coalesce(idx_lo, 0)) AS assisted_sessions
FROM b GROUP BY ALL HAVING sum(coalesce(idx_hi, 0) - coalesce(idx_lo, 0)) > 0
""", "assists counted")

# --- Outputs --------------------------------------------------------------

os.makedirs(os.path.join(OUT, "user_daily"), exist_ok=True)
os.makedirs(os.path.join(OUT, "campaign_daily"), exist_ok=True)

sql(f"""
COPY (
    WITH cat AS (
        SELECT s.person, s.day, e.category, sum(e.price) AS rev
        FROM ev_s e JOIN sess s ON e.person = s.person AND e.start = s.start
        WHERE e.is_purchase GROUP BY ALL
    ), top AS (
        SELECT person, day, arg_max(category, rev) AS top_category FROM (
            SELECT * FROM cat QUALIFY row_number() OVER (PARTITION BY person, day ORDER BY rev DESC, category) = 1
        ) GROUP BY ALL
    )
    SELECT s.person AS user_id, s.day, count(*)::BIGINT AS sessions, sum(s.n_events)::BIGINT AS events,
           sum(s.revenue)::BIGINT AS revenue_cents, any_value(t.top_category) AS top_category
    FROM sess s LEFT JOIN top t ON s.person = t.person AND s.day = t.day
    GROUP BY s.person, s.day
) TO '{os.path.join(OUT, "user_daily", "part-0.parquet")}' (FORMAT parquet)
""", "wrote user_daily")

sql(f"""
COPY (
    WITH credited AS (
        SELECT campaign_id, day, count(*) AS sessions, count_if(has_purchase) AS purchasing_sessions,
               sum(revenue) AS revenue_cents
        FROM credit GROUP BY ALL
    )
    SELECT campaign_id, coalesce(c.channel, 'organic') AS channel, day,
           coalesce(cr.sessions, 0)::BIGINT AS sessions,
           coalesce(cr.purchasing_sessions, 0)::BIGINT AS purchasing_sessions,
           coalesce(cr.revenue_cents, 0)::BIGINT AS revenue_cents,
           coalesce(a.assisted_sessions, 0)::BIGINT AS assisted_sessions
    FROM credited cr FULL OUTER JOIN assists a USING (campaign_id, day)
    LEFT JOIN campaigns c USING (campaign_id)
) TO '{os.path.join(OUT, "campaign_daily", "part-0.parquet")}' (FORMAT parquet)
""", "wrote campaign_daily")
