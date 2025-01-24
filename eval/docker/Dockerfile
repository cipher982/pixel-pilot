FROM ubuntu:22.04

# Install desktop and required packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ubuntu-desktop \
    ubuntu-gnome-desktop \
    wget \
    sudo \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install KasmVNC
RUN wget https://github.com/kasmtech/KasmVNC/releases/download/v1.3.0/kasmvncserver_jammy_1.3.0_amd64.deb \
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

USER ai
WORKDIR /home/ai

# Set VNC password
RUN mkdir -p ~/.vnc \
    && echo "ai" | vncpasswd -f > ~/.vnc/passwd \
    && chmod 600 ~/.vnc/passwd

EXPOSE 6901

CMD ["vncserver", "-select-de", "gnome", "-fg"] 