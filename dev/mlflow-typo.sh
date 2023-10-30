#!/usr/bin/env bash

if grep -P "(^|\s+)M(lf|LF)low([.\?]$|\s+)" "$@"; then
    exit 1
else
    exit 0
fi
