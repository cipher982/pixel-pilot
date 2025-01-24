FROM kasmweb/ubuntu-jammy-desktop:1.16.0

USER root

# Install additional Python tools and build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    python3-xlib \
    net-tools \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV with specific Python version
RUN python3.10 -V && \
    python3.10 -m pip install uv

# Switch to kasm-user
USER kasm-user
WORKDIR /home/kasm-user/pixelpilot

# Copy dependency files first
COPY --chown=kasm-user:kasm-user pyproject.toml uv.lock ./

# Install dependencies with dev extras
RUN UV_SYSTEM_PYTHON=/usr/bin/python3.10 uv sync --extra dev

# Copy project files
COPY --chown=kasm-user:kasm-user pixelpilot/ pixelpilot/
COPY --chown=kasm-user:kasm-user eval/ eval/
COPY --chown=kasm-user:kasm-user tests/ tests/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:1
ENV PATH="/home/kasm-user/pixelpilot/.venv/bin:${PATH}"
ENV PYTHONPATH="/home/kasm-user/pixelpilot"

# Entry point (uses default from base image) 