#!/bin/bash

# Create log directory
mkdir -p /var/log/pixelpilot

# Setup initial X11 auth file
touch ~/.Xauthority

# Start virtual framebuffer and redirect logs
Xvfb :0 -screen 0 1024x768x24 -ac > /var/log/pixelpilot/xvfb.log 2>&1 &
export DISPLAY=:0

echo "🖥️  Setting up X11 environment..."
for i in $(seq 1 10); do
    if xdpyinfo >/dev/null 2>&1; then
        echo "X server is ready"
        xauth generate :0 . trusted
        break
    fi
    sleep 1
done

# Start VNC server with minimal logging
x11vnc -display :0 -forever -nopw -quiet > /var/log/pixelpilot/x11vnc.log 2>&1 &

# Set log level for agent
export LOGLEVEL=INFO

echo -e "\n📋 Running PixelPilot Tests...\n"

# Run the test runner
uv run python eval/runner.py

# For debugging, save logs
echo -e "\n📝 X11 Setup Logs (if needed):"
echo "  - /var/log/pixelpilot/xvfb.log"
echo "  - /var/log/pixelpilot/x11vnc.log"
echo "  - /app/pixelpilot_debug.log"

# Keep container running for debugging if needed
tail -f /dev/null 