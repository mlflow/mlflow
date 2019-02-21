===================
MLflow Beta Release
===================

**Note:** The current version of MLflow is a beta release. This means that APIs and data formats
are subject to change!

**Note 2:** We do not currently support running MLflow on Windows. Despite this, we would appreciate any contributions
to make MLflow work better on Windows.

Installing
----------
Install MLflow from PyPi via ``pip install mlflow``

MLflow requires ``conda`` to be on the ``PATH`` for the projects feature.

Nightly snapshots of MLflow master are also available `here <https://mlflow-snapshots.s3-us-west-2.amazonaws.com/>`_.

Documentation
-------------
Official documentation for MLflow can be found at https://mlflow.org/docs/latest/index.html.

Community
---------
To discuss MLflow or get help, please subscribe to our mailing list (mlflow-users@googlegroups.com) or
join us on Slack at https://tinyurl.com/mlflow-slack.

To report bugs, please use GitHub issues.

Running a Sample App With the Tracking API
------------------------------------------
The programs in ``examples`` use the MLflow Tracking API. For instance, run::

    python examples/quickstart/mlflow_tracking.py

This program will use `MLflow Tracking API <https://mlflow.org/docs/latest/tracking.html>`_,
which logs tracking data in ``./mlruns``. This can then be viewed with the Tracking UI.


Launching the Tracking UI
-------------------------
The MLflow Tracking UI will show runs logged in ``./mlruns`` at `<http://localhost:5000>`_.
Start it with::

    mlflow ui

**Note:** Running ``mlflow ui`` from within a clone of MLflow is not recommended - doing so will
run the dev UI from source. We recommend running the UI from a different working directory, using the
``--file-store`` option to specify which log directory to run against. Alternatively, see instructions
for running the dev UI in the `contributor guide <CONTRIBUTING.rst>`_.


Running a Project from a URI
----------------------------
The ``mlflow run`` command lets you run a project packaged with a MLproject file from a local path
or a Git URI::

    mlflow run examples/sklearn_elasticnet_wine -P alpha=0.4

    mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=0.4

See ``examples/sklearn_elasticnet_wine`` for a sample project with an MLproject file.


Saving and Serving Models
-------------------------
To illustrate managing models, the ``mlflow.sklearn`` package can log scikit-learn models as
MLflow artifacts and then load them again for serving. There is an example training application in
``examples/sklearn_logisitic_regression/train.py`` that you can run as follows::

    $ python examples/sklearn_logisitic_regression/train.py
    Score: 0.666
    Model saved in run <run-id>

    $ mlflow sklearn serve -r <run-id> -m model

    $ curl -d '[{"x": 1}, {"x": -1}]' -H 'Content-Type: application/json' -X POST localhost:5000/invocations





Contributing
------------
We happily welcome contributions to MLflow. Please see our `contribution guide <CONTRIBUTING.rst>`_
for details.
