#!/bin/bash
set -ex

docker compose --project-directory tests/db "$@"
