#!/bin/bash

# Enable command logging
set -x

# Help message
show_help() {
    echo "Usage: $0 [test|eval]"
    echo
    echo "Modes:"
    echo "  test    Run unit tests"
    echo "  eval    Run evaluations"
    exit 1
}

# Parse mode
MODE=${1:-eval}  # Default to eval mode
if [[ "$MODE" != "test" && "$MODE" != "eval" ]]; then
    show_help
fi

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Set log level for agent
export LOGLEVEL=INFO

# Start VNC server
echo "🚀 Starting VNC server..."
date
echo "Starting VNC server with Gnome..."
vncserver -select-de gnome &
VNC_PID=$!
echo "VNC server started with PID: $VNC_PID"

# Wait for desktop to be ready
echo "🖥️  Waiting for desktop environment..."
date
for i in $(seq 1 30); do
    echo "Attempt $i: Checking if X server is ready..."
    if xdpyinfo >/dev/null 2>&1; then
        echo "Desktop is ready at attempt $i"
        date
        break
    fi
    sleep 1
done

# List running processes
echo "Running processes:"
ps aux | grep -E 'vnc|gnome'

# Change to project directory
cd /app
echo "Working directory: $(pwd)"

# Run appropriate command based on mode
if [[ "$MODE" == "test" ]]; then
    echo -e "\n🧪 Running Tests...\n"
    uv run python -m pytest tests/
else
    echo -e "\n📋 Running Evaluations...\n"
    uv run python -u eval/runner.py
fi

# Clean up and exit
cleanup 
