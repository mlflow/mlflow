# OpenTelemetry Protocol Buffer Definitions

This directory contains the OpenTelemetry Protocol Buffer (protobuf) specifications.

## What's Included?

MLflow only vendors the minimal set of OpenTelemetry proto files needed:

- `proto/trace/v1/trace.proto` - Trace data model (used by MLflow tracing)
- `proto/common/v1/common.proto` - Common data types
- `proto/resource/v1/resource.proto` - Resource attributes

**Note:** This is intentionally a minimal subset. If MLflow needs additional OpenTelemetry
proto files (e.g., metrics, logs), update `update.sh` to include the additional files in
the extraction list.

## Updating

To update the OpenTelemetry proto files to a new version:

1. Edit `update.sh` and update the `COMMIT_SHA` variable
2. Run the script:

   ```sh
   ./mlflow/protos/opentelemetry/update.sh
   ```
