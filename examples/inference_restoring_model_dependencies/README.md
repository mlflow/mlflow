### MLflow evaluation Examples

The examples in this directory illustrate how you can use the `mlflow.pyfunc.spark_udf` API with argument
`env_manager="conda"` to create a spark UDF for model inference with restoring the python environment used
in model training.

- Example `sklearn_model_inference_restoring_dependencies.py` runs a sklearn model inference via spark UDF 
with restoring the python environment used in model training.


#### Prerequisites

```
pip install scikit-learn
```

#### How to run the examples

Run in this directory with Python.

```
python sklearn_model_inference_restoring_dependencies.py
```
