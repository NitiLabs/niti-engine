#!/bin/bash
# Niti Engine Task Runner (Standalone Docker version)

# Get the directory of this script to support execution from any working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
BUILD=false
COMMAND=""
ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --build)
      BUILD=true
      shift
      ;;
    run|shell)
      if [ -z "$COMMAND" ]; then
        COMMAND=$1
      else
        ARGS+=("$1")
      fi
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$COMMAND" ]; then
  echo "Usage:
  ./run.sh run [script.py] [args] [--build]  - Run a Python script inside the Docker container
  ./run.sh shell [--build]                   - Open an interactive shell inside the Docker container

Examples:
  ./run.sh run benchmark_direct.py           # Run the direct benchmark script
  ./run.sh run benchmark_direct.py --build   # Rebuild image and run benchmark
  ./run.sh shell                             # Drop into the container's bash shell"
  exit 1
fi

# Build image if requested or if it doesn't exist yet
if [ "$BUILD" = true ] || ! docker image inspect niti-engine &>/dev/null; then
  echo "=== Building niti-engine Docker Image ==="
  docker build -t niti-engine "$SCRIPT_DIR"
fi

case "$COMMAND" in
  run)
    if [ ${#ARGS[@]} -eq 0 ]; then
      echo "Error: Please specify a script to run (e.g., ./run.sh run benchmark_direct.py)"
      exit 1
    fi
    
    echo "=== Running ${ARGS[0]} in Docker Container ==="
    # Mount the local directory to support live changes
    docker run --rm -v "$SCRIPT_DIR":/app niti-engine python "${ARGS[@]}"
    ;;

  shell)
    echo "=== Opening Interactive Shell in Docker Container ==="
    docker run -it --rm -v "$SCRIPT_DIR":/app niti-engine /bin/bash
    ;;
esac
