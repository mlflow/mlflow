.. _tutorial:

Tutorial
========

This tutorial showcases how you can use MLflow end-to-end to:

- Train a linear regression model 
- Package the code that trains the model in a reusable and reproducible model format 
- Deploy the model into a simple HTTP server that will enable you to score predictions

This tutorial uses a dataset to predict the quality of wine based on quantitative features 
like the wine's "fixed acidity", "pH", "residual sugar", and so on. The dataset
is from UCI's `machine learning repository <http://archive.ics.uci.edu/ml/datasets/Wine+Quality>`_.
[1]_

.. contents:: Table of Contents
  :local:
  :depth: 1

What You'll Need
----------------
This tutorial uses MLflow, `conda <https://conda.io/docs/user-guide/install/index.html#>`_, and the tutorial code located at
``example/tutorial`` in the MLflow repository. To download the tutorial code, run:

.. code::

    git clone https://github.com/databricks/mlflow

Training the Model
------------------
First, train a linear regression model that takes two hyperparameters: ``alpha`` and ``l1_ratio``. The code is located at ``example/tutorial/train.py`` and is reproduced below.

.. code:: python

    import os
    import sys

    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNet

    import mlflow
    import mlflow.sklearn
    # Run from the root of MLflow
    # Read the wine-quality csv file 
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")

This example uses the familiar pandas, numpy, and sklearn APIs to create a simple machine learning
model. The :doc:`MLflow tracking APIs<tracking/>` log information about each
training run, like the hyperparameters ``alpha`` and ``l1_ratio``, used to train the model and metrics, like
the root mean square error, used to evaluate the model. The example also serializes the
model in a format that MLflow knows how to deploy.

You can run the example with default hyperparameters as follows:

.. code:: bash

    python example/tutorial/train.py

Try out some other values for ``alpha`` and ``l1_ratio`` by passing them as arguments to ``train.py``:

.. code:: bash

    python example/tutorial/train.py <alpha> <l1_ratio>

Each time you run the example, MLflow logs information about your experiment runs in the directory ``mlruns``.

.. note::
    If you would like to use the Jupyter notebook version of ``train.py``, try out the tutorial notebook at ``example/tutorial/train.py/train.ipynb``.

Comparing the Models
--------------------

Next, use the MLflow UI to compare the models that you have produced. Run ``mlflow ui``
in the same current working directory as the one that contains the ``mlruns`` directory and
open http://localhost:5000 in your browser.

On this page, you can see a list of experiment runs with metrics you can use to compare the models.

.. image:: _static/images/tutorial-compare.png

You can see that the lower ``alpha`` is, the better the model. You can also
use the search feature to quickly filter out many models. For example, the query ``metrics.rmse < 0.8``
returns all the models with root mean squared error less than 0.8. For more complex manipulations,
you can download this table as a CSV and use your favorite data munging software to analyze it.

Packaging the Training Code
---------------------------
Now that you have your training code, you can package it so that other data scientists can easily reuse the model, or so that you can run the training remotely, for example on Databricks. You do this by using :doc:`projects` conventions to specify the
dependencies and entry points to your code. The ``example/tutorial/MLproject`` file specifies that the project has the dependencies located in a
`Conda environment file <https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-file-manually>`_
called ``conda.yaml`` and has one entry point that takes two parameters: ``alpha`` and ``l1_ratio``.

.. code:: yaml

    # example/tutorial/MLproject

    name: tutorial

    conda_env: conda.yaml

    entry_points:
      main:
        parameters:
          alpha: float
          l1_ratio: {type: float, default: 0.1}
        command: "python train.py {alpha} {l1_ratio}"
        
        
The Conda file lists the dependencies:

.. code:: yaml

    # example/tutorial/conda.yaml

    name: tutorial
    channels:
      - defaults
    dependencies:
      - numpy=1.14.3
      - pandas=0.22.0
      - scikit-learn=0.19.1
      - pip:
        - mlflow

To run this project, invoke ``mlflow run example/tutorial -P alpha=0.42``. After running
this command, MLflow will run your training code in a new Conda environment with the dependencies
specified in ``conda.yaml``.

If the repository has an ``MLproject`` file in the root you can also run a project directly from GitHub. This tutorial is duplicated in the https://github.com/databricks/mlflow-example repository
which you can run with ``mlflow run git@github.com:databricks/mlflow-example.git -P alpha=0.42``.

Serving the Model
-----------------
Now that you have packaged your model using the MLproject convention and have identified the best model,
it is time to deploy the model using :doc:`models`. An MLflow Model is a standard format for
packaging machine learning models that can be used in a variety of downstream tools — for example,
real-time serving through a REST API or batch inference on Apache Spark.

In the example training code, after training the linear regression model, a function
in MLflow saved the model as an artifact within the run.

.. code::

    mlflow.sklearn.log_model(lr, "model")

To view this artifact, you can use the UI again. When you click a date in the list of experiment
runs you'll see this page.

.. image:: _static/images/tutorial-artifact.png

At the bottom, you can see that the call to ``mlflow.sklearn.log_model`` produced two files in
``/Users/mlflow/mlflow-prototype/mlruns/0/7c1a0d5c42844dcdb8f5191146925174/artifacts/model``.
The first file, ``MLmodel``, is a metadata file that tells MLflow how to load the model. The
second file, ``model.pkl``, is a serialized version of the linear regression model that you trained.

In this example, you can use this MLmodel format with MLflow to deploy a local REST server that can serve predictions.

To deploy the server, run:

.. code::

    mlflow sklearn serve /Users/mlflow/mlflow-prototype/mlruns/0/7c1a0d5c42844dcdb8f5191146925174/artifacts/model -p 1234

.. note::

    The version of Python used to create the model must be the same as the one running ``mlflow sklearn``.
    If this is not the case, you may see the error
    ``UnicodeDecodeError: 'ascii' codec can't decode byte 0x9f in position 1: ordinal not in range(128)``
    or ``raise ValueError, "unsupported pickle protocol: %d"``.

To serve a prediction, run:

.. code::

    curl -X POST -H "Content-Type:application/json" --data '[{"fixed acidity": 6.2, "volatile acidity": 0.66, "citric acid": 0.48, "residual sugar": 1.2, "chlorides": 0.029, "free sulfur dioxide": 29, "total sulfur dioxide": 75, "density": 0.98, "pH": 3.33, "sulphates": 0.39, "alcohol": 12.8}]' http://127.0.0.1:1234/invocations

which should return something like::

    {"predictions": [6.379428821398614]}


More Resources
--------------
Congratulations on finishing the tutorial! For more reading, see :doc:`tracking`, :doc:`projects`, :doc:`models`,
and more.


.. [1] P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
