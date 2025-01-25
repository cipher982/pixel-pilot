#!/usr/bin/expect -f

# No timeout
set timeout -1

# Create VNC directories
spawn bash -c "mkdir -p /home/ai/.vnc && chown -R ai:ai /home/ai/.vnc"
expect eof

# Set up VNC password
spawn kasmvncpasswd -u ai -w -r
expect "Password:"
send "aiaiai\r"
expect "Verify:"
send "aiaiai\r"
expect eof

# Start VNC server
spawn su - ai
expect "$"
send "kasmvncserver -depth 24 -geometry 1280x720 :1\r"
expect "Provide selection number:"
send "1\r"
expect "Enter username (default: ai):"
send "\r"
expect "Password:"
send "aiaiai\r"
expect "Verify:"
send "aiaiai\r"
expect "Please choose Desktop Environment to run:"
send "1\r"
expect "WARNING: /home/ai/.vnc/xstartup will be overwritten y/N?"
send "y\r"
expect "desktop-size"
expect "http-port"

# Keep container running
send "tail -f ~/.vnc/*.log\r"
interact 
