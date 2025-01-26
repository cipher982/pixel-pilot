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
VNC_WAIT=${VNC_WAIT:-false}  # Default to not waiting for VNC
echo "Starting container in $MODE mode..."

if [[ "$MODE" != "test" && "$MODE" != "eval" ]]; then
    show_help
fi

# Set up runtime directory and environment
echo "Setting up runtime directory..."
export XDG_RUNTIME_DIR="/run/user/1000"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus"

# Set log level for agent
export LOGLEVEL=INFO

# Start virtual framebuffer X server
echo "Starting X server..."
Xvfb :1 -screen 0 1280x720x24 &
XVFB_PID=$!
export DISPLAY=:1

# Wait for desktop to be ready
echo "🖥️  Waiting for desktop environment..."
for i in $(seq 1 30); do
    if DISPLAY=:1 xdpyinfo >/dev/null 2>&1; then
        echo "Desktop is ready!"
        break
    fi
    sleep 1
done

# Start system dbus (requires sudo)
echo "Starting system services..."
echo "  Creating dbus directory..."
sudo /usr/bin/mkdir -p /var/run/dbus >/dev/null 2>&1
echo "  Starting system dbus..."
sudo /usr/bin/dbus-daemon --system --fork >/dev/null 2>&1
echo "  System dbus started"

# Start session dbus
echo "  Starting session dbus daemon..."
dbus-daemon --session --fork >/dev/null 2>&1
DBUS_PID=$!
echo "  Launching dbus session..."
eval $(dbus-launch --sh-syntax 2>/dev/null)
echo "  Session dbus started"

# Start XFCE session
echo "Starting XFCE session..."
mkdir -p /var/log/xfce
startxfce4 > /var/log/xfce/xfce4.log 2>&1 &
XFCE_PID=$!

# Wait for XFCE to be fully ready
echo "Waiting for XFCE..."
while ! pgrep -f "xfce4-session" > /dev/null; do
    sleep 1
done
echo "XFCE session ready"

# Wait for XFCE panel to be ready
echo "Waiting for XFCE panel..."
while ! xwininfo -root -tree 2>/dev/null | grep -q "xfce4-panel"; do
    sleep 1
done
echo "XFCE panel detected"

# Start VNC server
echo "Starting VNC server..."
x11vnc -display :1 -nopw -forever -shared -quiet > /var/log/xfce/vnc.log 2>&1 &
VNC_PID=$!

# Change to project directory
cd "${USER_HOME}/${PROJECT_NAME}"
echo "Working directory: $(pwd)"

# If VNC_WAIT is true, wait for user to connect before proceeding
if [[ "$VNC_WAIT" == "true" ]]; then
    echo -e "\n⏸️  Waiting for user confirmation..."
    echo "----------------------------------------"
    echo "VNC server is ready on port 5900"
    echo "Connect with VNC viewer and click 'Start Tests' icon on the desktop"
    echo "----------------------------------------"
    
    # Create trigger file
    TRIGGER_FILE="/tmp/start_tests"
    rm -f "$TRIGGER_FILE"
    
    # Create desktop launcher
    mkdir -p "${USER_HOME}/Desktop"
    cat > "${USER_HOME}/Desktop/start_tests.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Start Tests
Comment=Click to start PixelPilot tests
Exec=touch /tmp/start_tests
Icon=system-run
Terminal=false
Categories=Utility;
EOF
    chown ${USER_NAME}:${USER_NAME} "${USER_HOME}/Desktop/start_tests.desktop"
    chmod 755 "${USER_HOME}/Desktop/start_tests.desktop"
    
    # Wait for trigger file
    while [ ! -f "$TRIGGER_FILE" ]; do
        sleep 1
    done
        
    echo "Starting tests..."
fi

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
    echo "Evaluation complete. VNC server running on port 5900..."
    # Wait for remaining background processes
    wait $XVFB_PID $XFCE_PID $VNC_PID
fi 
