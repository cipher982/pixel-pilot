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

# Declare build arguments
ARG USER_NAME
ARG USER_HOME
ARG PROJECT_NAME

# Create user and add to groups
RUN useradd -m -s /bin/bash $USER_NAME \
    && adduser $USER_NAME sudo \
    && adduser $USER_NAME ssl-cert \
    && adduser $USER_NAME video \
    && adduser $USER_NAME tty

# Set ownership of runtime directory
RUN chown $USER_NAME:$USER_NAME /run/user/1000

# Set password for user (required for sudo)
RUN echo "$USER_NAME:$USER_NAME" | chpasswd

# Configure sudo for dbus commands
RUN echo "$USER_NAME ALL=(ALL) NOPASSWD: /usr/bin/mkdir -p /var/run/dbus" >> /etc/sudoers.d/$USER_NAME && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: /usr/bin/dbus-daemon" >> /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

# Set up X11 socket directory
RUN mkdir -p /tmp/.X11-unix && \
    chmod 1777 /tmp/.X11-unix && \
    chown root:root /tmp/.X11-unix

# Basic X11 setup
RUN mkdir -p $USER_HOME/.vnc \
    && chown -R $USER_NAME:$USER_NAME $USER_HOME/.vnc \
    && touch $USER_HOME/.Xauthority \
    && chown $USER_NAME:$USER_NAME $USER_HOME/.Xauthority

# Set up log directory
RUN mkdir -p /var/log/xfce && chown -R $USER_NAME:$USER_NAME /var/log/xfce

# Install UV globally as root
RUN pip3 install uv

# Switch to user for remaining operations
USER $USER_NAME
WORKDIR $USER_HOME

# Create project directory
RUN mkdir -p $USER_HOME/$PROJECT_NAME

# Copy project files (will be mounted in docker-compose)
COPY --chown=$USER_NAME:$USER_NAME pyproject.toml uv.lock $USER_HOME/$PROJECT_NAME/

# Install project dependencies
WORKDIR $USER_HOME/$PROJECT_NAME
RUN uv sync --extra dev

# Set PYTHONPATH to include project
ENV PYTHONPATH=$USER_HOME/$PROJECT_NAME

# Return to home directory as final workspace
WORKDIR $USER_HOME

EXPOSE 5900

ENTRYPOINT ["./pixel-pilot/eval/docker/scripts/container_entrypoint.sh"]
CMD ["eval"] 