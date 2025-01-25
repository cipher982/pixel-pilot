FROM ubuntu:22.04

# Install desktop and required packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ubuntu-desktop \
    ubuntu-gnome-desktop \
    x11vnc \
    wget \
    sudo \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create user and add to groups
RUN useradd -m -s /bin/bash ai \
    && adduser ai sudo \
    && adduser ai ssl-cert

# Set password for ai user (required for sudo)
RUN echo 'ai:ai' | chpasswd

# Basic X11 setup
RUN mkdir -p /home/ai/.vnc \
    && chown -R ai:ai /home/ai/.vnc \
    && touch /home/ai/.Xauthority \
    && chown ai:ai /home/ai/.Xauthority

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies with dev extras
RUN pip3 install uv
RUN uv sync --extra dev

USER ai
WORKDIR /home/ai

EXPOSE 5900

# Use entrypoint script to handle modes
COPY eval/docker/scripts/container_entrypoint.sh /app/eval/docker/scripts/
ENTRYPOINT ["/app/eval/docker/scripts/container_entrypoint.sh"]
CMD ["eval"] 