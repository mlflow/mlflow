# Historical Pyfunc Models

These serialized model files are used in backwards compatibility tests, so we can ensure that models logged with old versions of MLflow are still able to be loaded in newer versions.

The model class lives in `backward_compatibility_model.py` and is saved with
`code_path`, so cloudpickle records an importable class instead of Python bytecode.
This keeps the fixtures usable across Python minor versions while retaining the
historical MLflow model format.

To regenerate a fixture, remove its version directory and run the following from
the MLflow root, replacing `2.7.1` with the desired MLflow version:

```bash
uv run --no-project --isolated --python 3.11 --exclude-newer P7D \
  --with mlflow==2.7.1 --with cloudpickle==2.2.1 \
  --with pip==23.3 --with setuptools==68.2.2 --with wheel==0.41.2 \
  python tests/resources/pyfunc_models/generate.py
```
