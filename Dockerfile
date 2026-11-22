# Shopping Sessions starter: DuckDB skeleton.
# The scorers are linux/amd64 (Intel Xeon). They build this file natively, so
# no FROM line pins a platform: `make build` builds for linux/amd64 anywhere.
FROM python:3.12-slim-bookworm

# The scorer runs the container as uid 1000; give that uid a name and a home.
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY pipeline/ /app/pipeline/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Compile the Numba parser now, so the timed run loads machine code instead of
# spending ~15 s compiling. "generic" makes the cache valid on any x86-64 CPU,
# whichever machine builds the image. pipeline.py copies it to /scratch at run
# time, because Numba ignores a cache directory it can't write to.
RUN NUMBA_CPU_NAME=generic NUMBA_CACHE_DIR=/app/numba_cache python /app/pipeline/fastparse.py \
    && chmod -R a+rX /app/numba_cache

# At run time the root filesystem is read-only, the user is 1000:1000, there is
# no network, /data is read-only, and /output and /scratch are the only writable
# places. run.sh points every temp and cache directory at /scratch.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
CMD ["/app/run.sh"]
