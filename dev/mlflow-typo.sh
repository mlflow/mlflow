#!/usr/bin/env bash

ALLOWED_PATTERNS='Mlflow\(|"Mlflow"|import Mlflow$'
# comma separated string of excluded directories
EXCLUDED_DIRS="mlflow/server/js/src/lang"

if grep -InE ' \bM(lf|LF|lF)low\b' --exclude-dir={$EXCLUDED_DIRS} "$@" | grep -vE "$ALLOWED_PATTERNS"; then
    echo -e "\nFound typo for MLflow spelling in above file(s). Please use 'MLflow' instead of 'Mlflow'."
    exit 1
else
    exit 0
fi
