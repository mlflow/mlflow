#!/bin/bash
set -ex

docker compose --project-directory tests/db down --volumes --remove-orphans > /dev/null 2>&1
docker compose --project-directory tests/db "$@"
