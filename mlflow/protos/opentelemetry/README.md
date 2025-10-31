# OpenTelemetry Protocol Buffer Definitions

This directory contains the OpenTelemetry Protocol Buffer (protobuf) specifications.

## Version

**OpenTelemetry Proto Version**: v1.7.0

Source: https://github.com/open-telemetry/opentelemetry-proto

## Why Vendored?

These proto files are committed directly to the MLflow repository instead of being
downloaded during build time to:

- Simplify the build process
- Ensure reproducible builds
- Avoid network dependencies during compilation
- Provide better version control

## What's Included?

MLflow only vendors the minimal set of OpenTelemetry proto files needed:

- `proto/trace/v1/trace.proto` - Trace data model (used by MLflow tracing)
- `proto/common/v1/common.proto` - Common data types
- `proto/resource/v1/resource.proto` - Resource attributes

**Note:** This is intentionally a minimal subset. If MLflow needs additional OpenTelemetry
proto files (e.g., metrics, logs), update `dev/update-opentelemetry-protos.sh` to include
the additional files in the extraction list.

## Updating

To update the OpenTelemetry proto files to a new version:

1. Edit `dev/update-opentelemetry-protos.sh` and update the `VERSION` variable
2. Run the script:
   ```bash
   bash dev/update-opentelemetry-protos.sh
   ```
