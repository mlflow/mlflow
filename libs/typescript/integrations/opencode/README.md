# mlflow-opencode

MLflow tracing plugin for [Opencode](https://opencode.ai).

For full documentation, see: https://mlflow.org/docs/latest/genai/tracing/integrations/listing/opencode

## Installation

```bash
bun add mlflow-opencode
# or
npm install mlflow-opencode
```

## Usage

1. Add to your `opencode.json`:

```json
{
  "plugin": ["mlflow-opencode"]
}
```

2. Set environment variables:

```bash
export MLFLOW_OPENCODE_TRACING_ENABLED=true
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

3. Run Opencode normally - traces are created automatically.

## License

Apache-2.0
