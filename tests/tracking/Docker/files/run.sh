#!/bin/sh

set -e

if [ -z $FILE_DIR ]; then
  echo >&2 "FILE_DIR must be set"
  exit 1
fi

mkdir $FILE_DIR && mlflow server \
    --file-store $FILE_DIR \
    --host 0.0.0.0 \
    --port $PORT
