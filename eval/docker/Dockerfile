FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV and create venv
RUN pip install uv
RUN uv venv

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY pixelpilot ./pixelpilot
COPY eval ./eval

# Install dependencies
RUN uv sync

# Create test directories
RUN mkdir -p /app/eval/artifacts

# Set entrypoint to use venv
ENTRYPOINT ["/app/.venv/bin/python"] 