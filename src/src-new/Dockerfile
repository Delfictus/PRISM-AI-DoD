# PRISM-AI API Server Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM rust:1.75-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY benches ./benches
COPY examples ./examples

# Build for release with optimizations
RUN cargo build --release --bin api_server

# Stage 2: Runtime
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 prism && \
    mkdir -p /app/data /app/logs && \
    chown -R prism:prism /app

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/api_server /app/api_server

# Copy configuration files (if any)
# COPY config/ ./config/

# Set ownership
RUN chown -R prism:prism /app

# Switch to non-root user
USER prism

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Run the API server
CMD ["/app/api_server"]
