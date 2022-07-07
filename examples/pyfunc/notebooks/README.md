### MLflow restore model dependencies examples

The example "restore_model_dependencies_example.ipynb" in this directory illustrates
how you can use the `mlflow.pyfunc.get_model_dependencies` API to get the dependencies from a model URI
and install them, restoring the exact python environment that was used to build the model.

#### Prerequisites

```
pip install scikit-learn
```

#### How to run the example

Use jupyter to load the notebook "restore_model_dependencies_example.ipynb" and run the notebook.
