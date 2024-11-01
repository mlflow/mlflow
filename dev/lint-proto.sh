#!/usr/bin/env bash

if grep -n 'com.databricks.mlflow.api.MlflowTrackingMessage' "$@"; then
  echo 'Remove com.databricks.mlflow.api.MlflowTrackingMessage'
  exit 1
fi
