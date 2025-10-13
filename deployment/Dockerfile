# Multi-stage Dockerfile for PRISM-AI API Server
# Optimized for production deployment with GPU support

# Stage 1: Builder
FROM nvidia/cuda:13.0-devel-ubuntu22.04 AS builder

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# Copy Cargo files first for layer caching
COPY 03-Source-Code/Cargo.toml 03-Source-Code/Cargo.lock ./
COPY 03-Source-Code/src/neuromorphic/Cargo.toml ./src/neuromorphic/
COPY 03-Source-Code/src/quantum/Cargo.toml ./src/quantum/
COPY 03-Source-Code/src/foundation/Cargo.toml ./src/foundation/
COPY 03-Source-Code/src/shared-types/Cargo.toml ./src/shared-types/
COPY 03-Source-Code/src/prct-core/Cargo.toml ./src/prct-core/
COPY 03-Source-Code/src/mathematics/Cargo.toml ./src/mathematics/
COPY 03-Source-Code/validation/Cargo.toml ./validation/

# Create dummy source files for dependency caching
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/bin/api_server.rs && \
    echo "fn main() {}" > src/lib.rs && \
    mkdir -p src/neuromorphic/src && echo "fn main() {}" > src/neuromorphic/src/lib.rs && \
    mkdir -p src/quantum/src && echo "fn main() {}" > src/quantum/src/lib.rs && \
    mkdir -p src/foundation/src && echo "fn main() {}" > src/foundation/src/lib.rs && \
    mkdir -p src/shared-types/src && echo "fn main() {}" > src/shared-types/src/lib.rs && \
    mkdir -p src/prct-core/src && echo "fn main() {}" > src/prct-core/src/lib.rs && \
    mkdir -p src/mathematics/src && echo "fn main() {}" > src/mathematics/src/lib.rs && \
    mkdir -p validation/src && echo "fn main() {}" > validation/src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release --bin api_server --features api_server

# Copy actual source code
COPY 03-Source-Code/ .

# Build the actual application
RUN cargo build --release --bin api_server --features api_server

# Stage 2: Runtime
FROM nvidia/cuda:13.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 prism && \
    mkdir -p /app && \
    chown -R prism:prism /app

WORKDIR /app

# Copy binary from builder
COPY --from=builder --chown=prism:prism /build/target/release/api_server /app/api_server

# Switch to non-root user
USER prism

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment defaults
ENV RUST_LOG=info
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Run the server
CMD ["/app/api_server"]
