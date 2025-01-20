#!/bin/bash

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Start virtual framebuffer
Xvfb :0 -screen 0 1024x768x24 -ac &
export DISPLAY=:0

echo "Waiting for X server to start..."
for i in $(seq 1 10); do
    if xdpyinfo >/dev/null 2>&1; then
        echo "X server is ready"
        break
    fi
    sleep 1
done

# Start VNC server
echo "Starting VNC server..."
x11vnc -display :0 -forever -nopw &

# Run xeyes as a simple test
echo "Starting xeyes..."
xeyes &

# Wait for any signal
wait