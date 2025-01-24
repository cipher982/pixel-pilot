#!/bin/bash

# Enable command logging
set -x

echo "🔍 Debug Info:"
echo "DISPLAY=$DISPLAY"
echo "XAUTHORITY=$XAUTHORITY"
echo "Current directory: $(pwd)"
echo "X server check:"
xdpyinfo | head -n 3 || echo "X server not responding"

# Create log directory if it doesn't exist
mkdir -p /var/log/pixelpilot

# Start VNC server with more logging
echo "Starting VNC server..."
x11vnc -display :99 -forever -nopw -shared -verbose -logfile /var/log/pixelpilot/x11vnc-verbose.log &
VNC_PID=$!

# Give VNC server a moment to start
sleep 2

# Check if VNC server is running (using /proc instead of ps)
if [ ! -d "/proc/$VNC_PID" ]; then
    echo "❌ VNC server failed to start! Check logs:"
    tail -n 20 /var/log/pixelpilot/x11vnc-verbose.log
    exit 1
fi

# Verify VNC is listening
if ! netstat -tln 2>/dev/null | grep -q ':5900'; then
    echo "❌ VNC server not listening on port 5900! Check logs:"
    tail -n 20 /var/log/pixelpilot/x11vnc-verbose.log
    exit 1
fi

# Start a test window to show display is working
echo "Starting test window..."
xterm -geometry 80x24+10+10 -fg white -bg darkblue -T "PixelPilot Test Window" \
    -e "echo 'Display :99 is working! Waiting for tests to start...'; sleep infinity" &
XTERM_PID=$!

echo "🖥️  VNC server started on port 5900 (PID: $VNC_PID)"
echo "Waiting for VNC connection..."

# Function to check for VNC connection with logging
check_vnc_connection() {
    echo "Checking VNC connection..."
    # Check for established connections to VNC port
    netstat -tn | grep -q ":5900.*ESTABLISHED" && return 0
    
    # Backup check: look for client_set_net in recent logs
    tail -n 5 /var/log/pixelpilot/x11vnc-verbose.log | grep -q "client_set_net" && return 0
    
    return 1
}

# Show recent VNC logs
show_vnc_logs() {
    echo "Recent VNC server logs:"
    tail -n 10 /var/log/pixelpilot/x11vnc-verbose.log
}

# Wait for connection
TIMEOUT=30  # 30 seconds timeout
COUNT=0
while ! check_vnc_connection; do
    sleep 1
    COUNT=$((COUNT + 1))
    if [ $COUNT -eq $TIMEOUT ]; then
        echo "⚠️  No VNC connection after ${TIMEOUT} seconds."
        echo "Final connection attempt failed. Debug info:"
        show_vnc_logs
        echo "Continuing anyway..."
        break
    fi
    if [ $((COUNT % 5)) -eq 0 ]; then
        echo "Still waiting for VNC connection... (${COUNT}s/${TIMEOUT}s)"
        show_vnc_logs
    fi
done

if check_vnc_connection; then
    echo "✅ VNC client connected!"
    show_vnc_logs
else
    echo "⚠️ Proceeding without confirmed VNC connection"
fi

# Disable command logging before running main command
set +x

# Run the actual command
echo "🚀 Executing: $@"
exec "$@" 
