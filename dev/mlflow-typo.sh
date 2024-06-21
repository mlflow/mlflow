#!/usr/bin/env bash

ALLOWED_PATTERNS='Mlflow\(|"Mlflow"|import Mlflow$'
if grep -InE ' \bM(lf|LF|lF)low\b' "$@" | grep -vE "$ALLOWED_PATTERNS"; then
    echo -e "\nFound typo for MLflow spelling in above file(s). Please use 'MLflow' instead of 'Mlflow'."
    exit 1
else
    exit 0
fi
