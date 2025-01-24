#!/bin/bash

# Exit on any error
set -e

MODE=${1:-eval}  # Default to eval mode if no argument provided

# Help message
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [test]"
    echo ""
    echo "Modes:"
    echo "  (default) - Run evaluation suite"
    echo "  test      - Run pytest suite"
    echo ""
    echo "The desktop environment is always accessible via web browser at:"
    echo "http://localhost:6901 (password: password)"
    exit 0
fi

# Validate mode
if [ "$MODE" != "eval" ] && [ "$MODE" != "test" ]; then
    echo "Error: Invalid mode '$MODE'"
    echo "Run '$0 --help' for usage information"
    exit 1
fi

echo "Building container..."
docker compose -f eval/docker/docker-compose.yml build

echo "Starting container in $MODE mode..."
docker compose -f eval/docker/docker-compose.yml run \
    --rm \
    -e MODE=$MODE \
    eval

echo "Done!" 