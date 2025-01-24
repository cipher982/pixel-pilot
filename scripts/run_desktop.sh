#!/bin/bash

# Exit on any error
set -e

echo "Building desktop container..."
docker build -f eval/docker/Dockerfile -t pixel-pilot-desktop .

echo "Starting desktop container..."
docker run --rm -it \
    -p 5900:5900 \
    --name pixel-pilot-desktop \
    pixel-pilot-desktop 