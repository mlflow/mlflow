#!/usr/bin/env bash

TRACE_UI_SRC="static-files/lib/notebook-trace-renderer/index.html"

# If offending files are found, list them and throw an exception
if grep -In $TRACE_UI_SRC --include="*.ipynb" "$@"; then
    echo -e "\nError: Found the MLflow Trace UI iframe in the above notebooks."
    echo "Please remove the iframe from the notebook source, as it will not render properly on previews or the website"
    exit 1
else
  exit 0
fi
