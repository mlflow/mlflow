#!/usr/bin/env bash

TRACE_UI_SRC="static-files/lib/notebook-trace-renderer/index.html"

# If offending files are found, list them and throw an exception
if grep -In $TRACE_UI_SRC --include="*.ipynb" "$@"; then
    echo -e "\nError: Found the MLflow Trace UI iframe in the above notebooks."
    echo "The trace UI in cell outputs will not render correctly in previews or the website"
    echo "Please run \`mlflow.tracing.disable_notebook_display()\` and rerun the cell to remove the iframe."
    exit 1
else
  exit 0
fi
