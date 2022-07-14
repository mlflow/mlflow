# Pyfunc model examples

This examples demonstrates the use of a pyfunc model with custom inference logic.
More specifically:
 - train a simple regression model
 - create a *pyfunc* model that encapsulates the regression model with an attached module for custom inference logic

# Structure of this example

This examples contains a ``train.py`` file that trains a scikit-learn model with iris dataset and uses MLflow Tracking APIs to log the model.

Then a nested **mlflow run** delivers the packaging of ``pyfunc`` model and ``custom_code`` module is attached
to act as a custom inference logic layer in inference time.
```
├── custom_code
│   ├── __init__.py
│   └── custom_code.py
├── train.py
├── __init__.py
```

# Steps to reproduce

1. Start an mlflow server
```
(terminal1) $ mlflow server  --default-artifact-root mlruns
```

2. Train and log the model
```
(terminal2) $ python train.py
```

3. Serve pyfunc model
```
(terminal3) $ mlflow models serve -m "runs:/<pyfunc_runid>/model" -p 5001 --no-conda
```

4. Send a request 
```
(terminal2) $ curl http://127.0.0.1:5001/invocations -H 'Content-Type: application/json; format=pandas-records' -d '[[1,1,1,1]]'
```