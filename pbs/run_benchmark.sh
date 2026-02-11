#!/bin/bash
# Benchmark: PBS vs pyenv for Python installation

set -e
cd "$(dirname "$0")"

echo "=== PBS Benchmark ==="
echo "Building with pre-built Python binary..."
START=$(date +%s)
docker build --no-cache -f Dockerfile.pbs -t benchmark-pbs .
END=$(date +%s)
PBS_TIME=$((END-START))
PBS_SIZE=$(docker image inspect benchmark-pbs --format='{{.Size}}' | awk '{printf "%.1f", $1/1024/1024}')
echo "PBS: ${PBS_TIME}s, ${PBS_SIZE} MB"
docker run --rm benchmark-pbs

echo ""
echo "=== pyenv Benchmark ==="
echo "Building with Python compiled from source..."
START=$(date +%s)
docker build --no-cache -f Dockerfile.pyenv -t benchmark-pyenv .
END=$(date +%s)
PYENV_TIME=$((END-START))
PYENV_SIZE=$(docker image inspect benchmark-pyenv --format='{{.Size}}' | awk '{printf "%.1f", $1/1024/1024}')
echo "pyenv: ${PYENV_TIME}s, ${PYENV_SIZE} MB"
docker run --rm benchmark-pyenv

echo ""
echo "=== Results ==="
echo "PBS:   ${PBS_TIME}s, ${PBS_SIZE} MB"
echo "pyenv: ${PYENV_TIME}s, ${PYENV_SIZE} MB"
SPEEDUP=$(awk "BEGIN {printf \"%.1f\", $PYENV_TIME / $PBS_TIME}")
echo "Speedup: ${SPEEDUP}x"
