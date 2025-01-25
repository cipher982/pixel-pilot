FROM ubuntu:22.04

# Install desktop and required packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ubuntu-desktop \
    ubuntu-gnome-desktop \
    x11vnc \
    xserver-xorg-legacy \
    wget \
    sudo \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add xvfb in separate layer
RUN apt-get update && apt-get install -y xvfb && rm -rf /var/lib/apt/lists/*

# Add dbus in separate layer
RUN apt-get update && apt-get install -y dbus-x11 && rm -rf /var/lib/apt/lists/*

# Configure xserver-xorg-legacy
RUN sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config

# Create user and add to groups
RUN useradd -m -s /bin/bash ai \
    && adduser ai sudo \
    && adduser ai ssl-cert \
    && adduser ai video \
    && adduser ai tty

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

# Fix permissions for app directory
RUN chown -R ai:ai /app

USER ai
WORKDIR /home/ai

EXPOSE 5900

# Use entrypoint script to handle modes
COPY eval/docker/scripts/container_entrypoint.sh /app/eval/docker/scripts/
ENTRYPOINT ["/app/eval/docker/scripts/container_entrypoint.sh"]
CMD ["eval"] 