#!/bin/bash

# Exit on any error
set -e

# Default to test mode if no argument provided
MODE=${1:-test}
USE_VNC=${VNC_ENABLED:-false}

# Create log directory
mkdir -p /var/log/pixelpilot

# Start Xvfb
Xvfb :99 -screen 0 1024x768x24 -ac > /var/log/pixelpilot/xvfb.log 2>&1 &
export DISPLAY=:99

# Wait for X server
for i in $(seq 1 10); do
    if xdpyinfo >/dev/null 2>&1; then
        echo "X server is ready"
        xauth generate :99 . trusted
        break
    fi
    sleep 1
done

# Choose script based on mode
case $MODE in
    test)
        CMD="/test-x11.sh"
        ;;
    eval)
        CMD="/run-eval.sh"
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 [test|eval]"
        exit 1
        ;;
esac

# If VNC is enabled, use the wait script
if [ "$USE_VNC" = "true" ]; then
    exec /wait-vnc.sh $CMD
else
    exec $CMD
fi 