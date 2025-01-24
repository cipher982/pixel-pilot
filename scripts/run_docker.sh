#!/bin/bash

# Exit on any error
set -e

# Help message
show_help() {
    echo "Usage: $0 [test|eval] [--vnc]"
    echo
    echo "Modes:"
    echo "  test    Run unit tests in Docker"
    echo "  eval    Run evaluations in Docker"
    echo
    echo "Options:"
    echo "  --vnc   Enable VNC server (port 5900)"
    exit 1
}

# Parse arguments
MODE=$1
shift || true

VNC=""
VNC_ENV=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --vnc)
            VNC="--publish 5900:5900"
            VNC_ENV="-e VNC_ENABLED=true"
            shift
            ;;
        *)
            show_help
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "test" && "$MODE" != "eval" ]]; then
    show_help
fi

# Build the image
echo "Building Docker image..."
docker build -f eval/docker/Dockerfile.x11 -t pixel-pilot-test .

# Run container with appropriate mode
echo "Running in $MODE mode..."
docker run --rm $VNC $VNC_ENV \
    --env-file .env \
    -v "$(pwd)/eval/test_cases:/app/eval/test_cases:ro" \
    -v "$(pwd)/eval/artifacts:/app/eval/artifacts" \
    -v "$(pwd)/pixelpilot:/app/pixelpilot:ro" \
    -v "$(pwd)/eval:/app/eval:ro" \
    pixel-pilot-test "$MODE"
