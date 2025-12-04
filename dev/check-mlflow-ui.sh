#!/usr/bin/env bash

# Pre-commit hook to flag usage of 'mlflow ui' and suggest 'mlflow server' instead
# 'mlflow ui' is deprecated in favor of 'mlflow server'
#
# Note: Files that are allowed to contain 'mlflow ui' should be excluded via the
# 'exclude' parameter in .pre-commit-config.yaml rather than in this script.

# Use bin/rg to search for 'mlflow ui' in the provided files
# The search is case-sensitive and looks for the exact phrase
if bin/rg "mlflow ui" --line-number "$@"; then
    echo ""
    echo "Error: Found usage of 'mlflow ui' in the above file(s)."
    echo "Please use 'mlflow server' instead of 'mlflow ui'."
    echo "'mlflow ui' is deprecated and 'mlflow server' is the preferred command."
    exit 1
else
    exit 0
fi
