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

# Create log directory
mkdir -p /home/ai/pixel-pilot/logs

# Set log level for agent
export LOGLEVEL=INFO

# Wait for desktop to be ready (Kasmweb handles X11/VNC setup)
echo "🖥️  Waiting for desktop environment..."
for i in $(seq 1 30); do
    if xdpyinfo >/dev/null 2>&1; then
        echo "Desktop is ready"
        break
    fi
    sleep 1
done

# Change to project directory and activate venv
cd /home/ai/pixel-pilot
. .venv/bin/activate

# Run appropriate command based on mode
if [[ "$MODE" == "test" ]]; then
    echo -e "\n🧪 Running Tests...\n"
    python -m pytest tests/
else
    echo -e "\n📋 Running Evaluations...\n"
    python -u eval/runner.py
fi

# Clean up and exit
cleanup 
