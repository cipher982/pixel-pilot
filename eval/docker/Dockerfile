FROM ubuntu:22.04

# Install desktop and required packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ubuntu-desktop \
    ubuntu-gnome-desktop \
    wget \
    sudo \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install KasmVNC
RUN wget https://github.com/kasmtech/KasmVNC/releases/download/v1.3.0/kasmvncserver_jammy_1.3.0_arm64.deb \
    && apt-get update \
    && apt-get install -y ./kasmvncserver_*.deb \
    && rm kasmvncserver_*.deb \
    && rm -rf /var/lib/apt/lists/*

# Create user and add to groups
RUN useradd -m -s /bin/bash ai \
    && adduser ai sudo \
    && adduser ai ssl-cert

# Set password for ai user (required for sudo)
RUN echo 'ai:ai' | chpasswd

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies with dev extras
RUN pip3 install uv
RUN uv sync --extra dev

# Switch to user
USER ai
WORKDIR /home/ai

# Create VNC directory and setup KasmVNC user
RUN mkdir -p ~/.vnc \
    && vncpasswd -u ai -w -r

EXPOSE 6901

CMD ["vncserver", "-select-de", "gnome"] 