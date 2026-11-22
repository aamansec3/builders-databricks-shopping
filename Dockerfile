# Shopping Sessions: the Rust engine (engine/).
# The scorers are linux/amd64 (Intel Xeon). They build this file natively, so
# no FROM line pins a platform: `make build` builds for linux/amd64 anywhere.
# Stage 1: compile the Rust engine (engine/). Same Debian release as the final
# image, so the binary links against the same glibc.
FROM rust:1.100-slim-bookworm AS engine
WORKDIR /src
COPY engine/Cargo.toml engine/Cargo.lock ./
COPY engine/src ./src
RUN cargo build --release --locked

# Stage 2: the image the scorer runs: plain Debian with just the engine. (The
# Python/Numba pipeline in pipeline/ is no longer in the image; to go back to
# it, restore the Python stage from git history and run.sh's python line.)
FROM debian:bookworm-slim
COPY --from=engine /src/target/release/engine /app/engine
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# At run time the root filesystem is read-only, the user is 1000:1000, there is
# no network, /data is read-only, and /output and /scratch are the only writable
# places. run.sh points every temp and cache directory at /scratch.
WORKDIR /app
CMD ["/app/run.sh"]
