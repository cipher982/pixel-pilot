#!/bin/bash

# Exit on any error
set -e

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Start virtual framebuffer
Xvfb :99 -screen 0 1024x768x24 -ac &
export DISPLAY=:99

echo "Waiting for X server to start..."
for i in $(seq 1 10); do
    if xdpyinfo >/dev/null 2>&1; then
        echo "X server is ready"
        break
    fi
    sleep 1
    if [ $i -eq 10 ]; then
        echo "X server failed to start"
        exit 1
    fi
done

# Run all tests
echo "Running all tests..."
uv run pytest tests -v

# Clean up
cleanup