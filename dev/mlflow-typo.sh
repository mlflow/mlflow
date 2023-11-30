#!/usr/bin/env bash

# Patterns or paths to exclude (can include regex)
EXCLUDE_PATTERNS=("^examples/llms/rag/.*(?<!\\.ipynb)$")

for file in "$@"; do
    # Check if the file matches any exclude patterns
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ $file =~ $pattern ]]; then
            # Skip this file
            continue 2
        fi
    done

    # Run the grep check
    if grep -nP '\bM(lf|LF|lF)low\b' "$file"; then
        exit 1
    fi
done

exit 0