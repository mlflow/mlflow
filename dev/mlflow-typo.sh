#!/usr/bin/env bash

if grep -nP "(^|\s+)M(lf|LF)low([.\?]?$|\s+)" "$@"; then
    exit 1
else
    exit 0
fi
