FROM python:3.12-slim

RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync

# Then copy source code
COPY eval/ eval/
COPY pixelpilot/ pixelpilot/

CMD ["uv", "run", "python", "eval/runner.py"] 
