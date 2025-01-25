FROM ubuntu:22.04

# Install desktop and required packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    xfce4 \
    xfce4-terminal \
    x11vnc \
    xserver-xorg-legacy \
    wget \
    sudo \
    python3 \
    python3-pip \
    git \
    ssl-cert \
    && rm -rf /var/lib/apt/lists/*

# Add xvfb in separate layer
RUN apt-get update && apt-get install -y xvfb && rm -rf /var/lib/apt/lists/*

# Add dbus in separate layer
RUN apt-get update && apt-get install -y dbus-x11 && rm -rf /var/lib/apt/lists/*

# Configure xserver-xorg-legacy
RUN sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config

# Set up proper dbus directories and permissions
RUN mkdir -p /run/dbus \
    && chown messagebus:messagebus /run/dbus \
    && mkdir -p /var/run/dbus \
    && chown messagebus:messagebus /var/run/dbus \
    && mkdir -p /run/user/1000 \
    && chmod 700 /run/user/1000

# Create user and add to groups
RUN useradd -m -s /bin/bash ai \
    && adduser ai sudo \
    && adduser ai ssl-cert \
    && adduser ai video \
    && adduser ai tty

# Set ownership of runtime directory
RUN chown ai:ai /run/user/1000

# Set password for ai user (required for sudo)
RUN echo 'ai:ai' | chpasswd

# Configure sudo for dbus commands
RUN echo "ai ALL=(ALL) NOPASSWD: /usr/bin/mkdir -p /var/run/dbus" >> /etc/sudoers.d/ai && \
    echo "ai ALL=(ALL) NOPASSWD: /usr/bin/dbus-daemon" >> /etc/sudoers.d/ai && \
    chmod 0440 /etc/sudoers.d/ai

# Set up X11 socket directory
RUN mkdir -p /tmp/.X11-unix && \
    chmod 1777 /tmp/.X11-unix && \
    chown root:root /tmp/.X11-unix

# Basic X11 setup
RUN mkdir -p /home/ai/.vnc \
    && chown -R ai:ai /home/ai/.vnc \
    && touch /home/ai/.Xauthority \
    && chown ai:ai /home/ai/.Xauthority

WORKDIR /app

# Create and own venv directory
RUN mkdir -p /app/.venv && chown -R ai:ai /app/.venv

# Copy dependency files first
COPY pyproject.toml uv.lock ./
RUN chown ai:ai pyproject.toml uv.lock

# Install UV globally as root
RUN pip3 install uv

# Switch to ai user for project setup
USER ai

# Install project dependencies
RUN uv sync --extra dev

WORKDIR /home/ai

EXPOSE 5900

# Use entrypoint script to handle modes
COPY eval/docker/scripts/container_entrypoint.sh /app/eval/docker/scripts/
ENTRYPOINT ["/app/eval/docker/scripts/container_entrypoint.sh"]
CMD ["eval"] 