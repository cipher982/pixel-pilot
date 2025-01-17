#!/bin/bash

# Build the Docker image
docker build -t pixel-pilot-eval -f eval/docker/Dockerfile .

# Run the evaluation
docker run --rm \
    -v $(pwd)/eval/test_cases:/app/eval/test_cases \
    -v $(pwd)/eval/artifacts:/app/eval/artifacts \
    pixel-pilot-eval \
    eval/runner.py 