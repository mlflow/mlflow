#!/usr/bin/env bash

rm dist/*
pip wheel --no-deps --wheel-dir dist ../..
docker-compose build
