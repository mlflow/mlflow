# Pyfunc model example

This example demonstrates the use of a pyfunc model with custom inference logic.
More specifically:

- train a simple classification model
- create a _pyfunc_ model that encapsulates the classification model with an attached module for custom inference logic

## Structure of this example

This examples contains a `train.py` file that trains a scikit-learn model with iris dataset and uses MLflow Tracking APIs to log the model. The nested **mlflow run** delivers the packaging of `pyfunc` model and `custom_code` module is attached
to act as a custom inference logic layer in inference time.

```
├── train.py
├── infer_model_code_path.py
└── custom_code.py
```

## Running this example

1. Train and log the model

```
$ python train.py
```

or train and log the model using inferred code paths

```
$ python infer_model_code_paths.py
```

2. Serve the pyfunc model

```bash
# Replace <pyfunc_run_id> with the run ID obtained in the previous step
$ mlflow models serve -m "runs:/<pyfunc_run_id>/model" -p 5001
```

3. Send a request

```
$ curl http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_records": [[1, 1, 1, 1]]
}'
```

The response should look like this:

```
[0]
```
