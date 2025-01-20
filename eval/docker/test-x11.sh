#!/bin/bash

# Start virtual framebuffer
Xvfb :0 -screen 0 1024x768x24 -ac &
export DISPLAY=:0
sleep 1

# Run xeyes as a simple test
xeyes &

# Print what's running on our display
echo "Testing display $DISPLAY:"
xwininfo -root -children

# Keep container running
tail -f /dev/null 