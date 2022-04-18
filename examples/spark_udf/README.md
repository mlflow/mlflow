### MLflow Spark UDF Examples

The examples in this directory illustrate how you can use the `mlflow.pyfunc.spark_udf` API for batch inference,
including environment reproducibility capabilities with argument `env_manager="conda"`,
which creates a spark UDF for model inference that executes in an environment containing the exact dependency
versions used during training.

- Example `spark_udf.py` runs a sklearn model inference via spark UDF 
using a python environment containing the precise versions of dependencies used during model training.


#### Prerequisites

```
pip install scikit-learn
```

#### How to run the examples

```
python sklearn_model_inference_restoring_dependencies.py
```
