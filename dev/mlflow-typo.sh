#!/usr/bin/env bash

ALLOWED_PATTERNS='Mlflow\(|"Mlflow"|import Mlflow$'
if grep -InP ' \bM(lf|LF|lF)low\b' "$@" | grep -Pv "$ALLOWED_PATTERNS"; then
    exit 1
else
    exit 0
fi
