#!/bin/bash

# Help message
show_help() {
    echo "Usage: $0 [test|eval]"
    echo
    echo "Modes:"
    echo "  test    Run unit tests"
    echo "  eval    Run evaluations"
    exit 1
}

# Parse mode from first argument or environment variable
MODE=${1:-${MODE:-eval}}  # Use first arg, fallback to env MODE, default to eval
echo "Starting container in $MODE mode..."

if [[ "$MODE" != "test" && "$MODE" != "eval" ]]; then
    show_help
fi

# Create necessary directories
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix

# Set log level for agent
export LOGLEVEL=INFO

# Start virtual framebuffer X server
set -x  # Enable debug for important steps
Xvfb :1 -screen 0 1280x720x24 &
export DISPLAY=:1
set +x  # Disable debug for polling

# Wait for desktop to be ready
echo "🖥️  Waiting for desktop environment..."
for i in $(seq 1 30); do
    if DISPLAY=:1 xdpyinfo >/dev/null 2>&1; then
        echo "Desktop is ready!"
        break
    fi
    sleep 1
done

set -x  # Re-enable debug for important steps

# Start system dbus (requires sudo)
sudo /usr/bin/mkdir -p /var/run/dbus
sudo /usr/bin/dbus-daemon --system --fork

# Start session dbus
dbus-daemon --session --address=unix:path=/tmp/dbus-session --nofork &
export $(dbus-launch)

# Start GNOME session
gnome-session &

# Start VNC server for optional access
x11vnc -display :1 -nopw -forever -shared &

# Change to project directory
cd /app
echo "Working directory: $(pwd)"

# Run tests or evaluations based on mode
if [[ "$MODE" == "test" ]]; then
    echo -e "\n📋 Running Tests...\n"
    uv run python -m pytest tests/
else
    echo -e "\n📋 Running Evaluation...\n"
    uv run python -u eval/runner.py
fi

# Keep container running if in eval mode
if [[ "$MODE" == "eval" ]]; then
    # Container will keep running as long as X server is running
    wait
fi 
