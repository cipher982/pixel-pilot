FROM python:3.12-slim

RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv

WORKDIR /app
COPY eval/ .
COPY pixelpilot/ .
COPY pyproject.toml .
COPY uv.lock .

RUN uv sync

CMD ["python", "eval/runner.py"] 
