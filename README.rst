====================
MLflow Alpha Release
====================

**Note:** The current version of MLflow is an alpha. This means that APIs and storage formats
are subject to change!

Installing
----------
Install MLflow from PyPi via ``pip install mlflow``

MLflow requires ``conda`` to be on the ``PATH`` for the projects feature.

Documentation
-------------
Official documentation for MLflow can be found at https://mlflow.org/docs/latest/index.html.

Running a Sample App With the Tracking API
------------------------------------------
The programs in ``example`` use the MLflow Tracking API. For instance, run::

    python example/quickstart/test.py

This program will use MLflow log API, which stores tracking data in ``./mlruns``, which can then
be viewed with the Tracking UI.


Launching the Tracking UI
-------------------------
The MLflow Tracking UI will show runs logged in ``./mlruns`` at `<http://localhost:5000>`_.
Start it with::

    mlflow ui


Running a Project from a URI
----------------------------
The ``mlflow run`` command lets you run a project packaged with a MLproject file from a local path
or a Git URI::

    mlflow run example/tutorial -P alpha=0.4

    mlflow run git@github.com:databricks/mlflow-example.git -P alpha=0.4

See ``example/tutorial`` for a sample project with an MLproject file.


Saving and Serving Models
-------------------------
To illustrate managing models, the ``mlflow.sklearn`` package can log Scikit-learn models as
MLflow artifacts and then load them again for serving. There is an example training application in
``example/quickstart/test_sklearn.py`` that you can run as follows::

    $ python example/quickstart/test_sklearn.py
    Score: 0.666
    Model saved in run <run-id>

    $ mlflow sklearn serve -r <run-id> model

    $ curl -d '[{"x": 1}, {"x": -1}]' -H 'Content-Type: application/json' -X POST localhost:5000/invocations





Contributing
------------
We happily welcome contributions, please see our `contribution guide <CONTRIBUTING.rst>`_
for details.
