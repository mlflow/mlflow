### MLflow Fastai Example

This simple example illustrates how you can use the `mlflow.fastai.autolog()` API
to track parameters, metrics, and artifacts while training a simple MNIST model. Derived from fastai's GitHub Repository 
of [vision examples](https://github.com/fastai/fastai/blob/master/examples/train_mnist.py), this code is modified to run 
as an MLflow project, with either default or supplied arguments. The default arguments are learning rate(`lr=0.01`) 
and number of epochs (`epochs=5`).

You can use this example as a template and attempt advanced examples in
[Fastai tutorials](https://docs.fast.ai/applications.html), using the `mlflow.fastai` model flavor and MLflow tracking API to
track your experiments.

#### How to run this code

You can run the `fastai` example with default or supplied arguments in three ways:

1. Run from the current git directory with Python. 

**Note**:  This example assumes that you have all the dependencies for `fastai` library installed in your development environment. 

 `python train.py`
 
 `python train.py --lr=0.02 --epochs=3`

2. Run from the current git directory as an MLflow Project

 `mlflow run . -e main`
 
 `mlflow run . -e main -P lr=0.02 -P epochs=3`
 
3. Run from outside git repository as an MLflow Project

 `mlflow run https://github.com/mlflow/mlflow/\#examples/fastai`
 
 `mlflow run https://github.com/mlflow/mlflow/\#examples/fastai -P lr=0.02 -P epochs=3`
 
#### How to inspect your runs
All these runs will create a `mlruns` directory at the same directory level where you execute
these commands. To inspect the parameters, metrics, and artifacts automatically
logged by the `mlflow.fastai.autolog()` API, launch the MLflow UI using: `mlflow ui`.

In your browser, connect to `localhost:5000 or 127.0.0.1:5000`.
