#!/bin/bash

# Exit on error
set -e

# Build the image
echo "Building desktop container..."
docker build -t pixel-pilot-desktop -f eval/docker/Dockerfile .

# Run the container
echo "Starting desktop container..."
docker run --rm -it \
    --shm-size=512m \
    -p 6901:6901 \
    -e VNC_PW=password \
    -v "$(pwd):/home/kasm-user/pixel-pilot" \
    pixel-pilot-desktop

echo "Desktop ready at https://localhost:6901"
echo "Username: kasm_user"
echo "Password: password" 