#!/usr/bin/env bash

if grep -InP '(?<!import\s)\bM(lf|LF|lF)low\b(?!\()' "$@"; then
    exit 1
else
    exit 0
fi
