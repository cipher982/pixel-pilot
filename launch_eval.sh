#!/bin/bash

# Exit on any error
set -e

# Source environment variables from .env file
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

MODE="eval"  # Default to eval mode
VNC_WAIT=false   # Default to not waiting for VNC

# Check for required environment variables
if [ -z "$USER_NAME" ] || [ -z "$USER_HOME" ] || [ -z "$PROJECT_NAME" ]; then
    echo "Error: Required environment variables not set"
    echo "Please ensure these are set:"
    echo "  USER_NAME   (current: ${USER_NAME:-not set})"
    echo "  USER_HOME   (current: ${USER_HOME:-not set})"
    echo "  PROJECT_NAME (current: ${PROJECT_NAME:-not set})"
    exit 1
fi

# Help message
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [test|eval] [--vnc]"
    echo ""
    echo "Modes:"
    echo "  (default) - Run evaluation suite"
    echo "  test      - Run pytest suite"
    echo ""
    echo "Options:"
    echo "  --vnc     - Wait for VNC viewer connection before running tests"
    echo ""
    echo "The desktop environment is accessible via VNC at:"
    echo "localhost:5900"
    exit 0
fi

# Parse all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vnc)
            VNC_WAIT=true
            shift
            ;;
        test|eval)
            MODE=$1
            shift
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

echo "Building container..."
docker compose --env-file .env -f eval/docker/docker-compose.yml build

echo "Starting container in $MODE mode (VNC wait: $VNC_WAIT)..."
export MODE=$MODE
export VNC_WAIT=$VNC_WAIT
docker compose --env-file .env -f eval/docker/docker-compose.yml up \
    eval

echo "Done!" 