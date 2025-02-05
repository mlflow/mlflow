#!/bin/bash
set -ex

# Is a wheel present in dist directory?
if [ ! -f dist/*.whl ]; then
  echo "No wheel found in dist directory. Run 'python dev/build.py' to build a wheel."
  exit 1
fi

docker compose --project-directory tests/db down --volumes --remove-orphans > /dev/null 2>&1
docker compose --project-directory tests/db "$@"
