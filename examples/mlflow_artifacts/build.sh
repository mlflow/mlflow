#!/usr/bin/env bash

rm -rf dist
pip wheel --no-deps --wheel-dir dist ../..
DOCKERFILE=Dockerfile.dev docker compose build
