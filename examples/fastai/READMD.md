### MLflow Fastai Example

This simple example illustrates how you can use `mlflow.fastai.autolog()` API
to track parameters, metrics, and artifacts while training a simple MNIST model. Derived from fastai GitHub Repository 
of [vision examples](https://github.com/fastai/fastai/blob/master/examples/train_mnist.py), the code is modified to run 
as a MLflow project, with either default or supplied arguments. The default arguments are learning rate(`lr=0.01`) 
and number of epochs (`epochs=5`).

You can use this example as a template and attempt advanced examples in
[Fastai tutorials](https://docs.fast.ai/vision.html), using `mlfow.fastai` model flavor and MLflow tracking API to
track your experiments.

#### How to run the code

You can run the `fastai` example with default or supplied arguments in three ways:

1. Run from the current git directory with Python. 

**Note**:  This assumes that you have all the dependencies for `fastai` library installed in your enviroment. 

 `python train.py`
 
 `python train.py --lr=0.02 --epochs=3`

2. Run from the current git directory as a MLflow Project

 `mlflow run . -e main`
 
 `mlflow run . -e main -P lr=0.02 -P epochs=3`
 
3. Run from outside git repository as a MLflow Project

 `mlflow run https://github.com/mlflow/\#examples/fastai`
 
 `mlflow run https://github.com/mlflow/\#examples/fastai -P lr=0.02 -P epochs=3`

### MLflow UI
All runs will create a `mlruns` directory at the same directory level where you execute
these commands. To launch an MLflow UI to inspect the parameters, metrics, and artifacts automatically
logged by the `mflow.fastai.autolog()`, launch the MLflow UI: `mlflow ui`.

In your brower, connect to `localhost:5000 or 127.0.0.1:5000`.
