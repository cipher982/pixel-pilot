#!/bin/bash

# Build and run tests in Docker

# Exit on any error
set -e

# Build the test image
echo "Building test image..."
docker build -f eval/docker/Dockerfile.x11 -t pixel-pilot-test .

# Run the tests
echo "Running tests..."
docker run --rm pixel-pilot-test

# If you need to debug with VNC:
# docker run --rm -p 5900:5900 pixel-pilot-test /test-x11.sh 