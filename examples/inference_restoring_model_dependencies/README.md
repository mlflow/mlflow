### MLflow evaluation Examples

The examples in this directory illustrate how you can use the `mlflow.pyfunc.spark_udf` API with argument
`env_manager="conda"` to create a spark UDF for model inference that executes in an environment containing the exact dependency versions used during training.

- Example `sklearn_model_inference_restoring_dependencies.py` runs a sklearn model inference via spark UDF 
using a python environment containing the precise versions of dependencies used during model training.


#### Prerequisites

```
pip install scikit-learn
```

#### How to run the examples

```
python sklearn_model_inference_restoring_dependencies.py
```
