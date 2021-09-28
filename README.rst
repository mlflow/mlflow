=============================================
MLflow: A Machine Learning Lifecycle Platform
=============================================

MLflow is a platform to streamline machine learning development, including tracking experiments, packaging code
into reproducible runs, and sharing and deploying models. MLflow offers a set of lightweight APIs that can be
used with any existing machine learning application or library (TensorFlow, PyTorch, XGBoost, etc), wherever you
currently run ML code (e.g. in notebooks, standalone applications or the cloud). MLflow's current components are:

* `MLflow Tracking <https://mlflow.org/docs/latest/tracking.html>`_: An API to log parameters, code, and
  results in machine learning experiments and compare them using an interactive UI.
* `MLflow Projects <https://mlflow.org/docs/latest/projects.html>`_: A code packaging format for reproducible
  runs using Conda and Docker, so you can share your ML code with others.
* `MLflow Models <https://mlflow.org/docs/latest/models.html>`_: A model packaging format and tools that let
  you easily deploy the same model (from any ML library) to batch and real-time scoring on platforms such as
  Docker, Apache Spark, Azure ML and AWS SageMaker.
* `MLflow Model Registry <https://mlflow.org/docs/latest/model-registry.html>`_: A centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models.

|docs| |labeling| |examples| |cross-version-tests| |pypi| |conda-forge| |cran| |maven| |license| |downloads| |slack|

.. |docs| image:: https://img.shields.io/badge/docs-latest-success.svg
    :target: https://mlflow.org/docs/latest/index.html
    :alt: Latest Docs
.. |labeling| image:: https://github.com/mlflow/mlflow/workflows/Labeling/badge.svg
    :target: https://github.com/mlflow/mlflow/actions?query=workflow%3ALabeling
    :alt: Labeling Action Status
.. |examples| image:: https://github.com/mlflow/mlflow/workflows/Examples/badge.svg?event=schedule
    :target: https://github.com/mlflow/mlflow/actions?query=workflow%3AExamples+event%3Aschedule
    :alt: Examples Action Status
.. |cross-version-tests| image:: https://github.com/mlflow/mlflow/workflows/Cross%20version%20tests/badge.svg?event=schedule
    :target: https://github.com/mlflow/mlflow/actions?query=workflow%3ACross%2Bversion%2Btests+event%3Aschedule
    :alt: Examples Action Status
.. |pypi| image:: https://img.shields.io/pypi/v/mlflow.svg
    :target: https://pypi.org/project/mlflow/
    :alt: Latest Python Release
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/mlflow.svg
    :target: https://anaconda.org/conda-forge/mlflow
    :alt: Latest Conda Release
.. |cran| image:: https://img.shields.io/cran/v/mlflow.svg
    :target: https://cran.r-project.org/package=mlflow
    :alt: Latest CRAN Release
.. |maven| image:: https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg
    :target: https://mvnrepository.com/artifact/org.mlflow
    :alt: Maven Central
.. |license| image:: https://img.shields.io/badge/license-Apache%202-brightgreen.svg
    :target: https://github.com/mlflow/mlflow/blob/master/LICENSE.txt
    :alt: Apache 2 License
.. |downloads| image:: https://pepy.tech/badge/mlflow
    :target: https://pepy.tech/project/mlflow
    :alt: Total Downloads
.. |slack| image:: https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40
    :target: `Slack`_
    :alt: Slack

.. _Slack: https://join.slack.com/t/mlflow-users/shared_invite/zt-g6qwro5u-odM7pRnZxNX_w56mcsHp8g

Installing
----------
Install MLflow from PyPI via ``pip install mlflow``

MLflow requires ``conda`` to be on the ``PATH`` for the projects feature.

Nightly snapshots of MLflow master are also available `here <https://mlflow-snapshots.s3-us-west-2.amazonaws.com/>`_.

Install a lower dependency subset of MLflow from PyPI via ``pip install mlflow-skinny``
Extra dependencies can be added per desired scenario.
For example, ``pip install mlflow-skinny pandas numpy`` allows for mlflow.pyfunc.log_model support.

Documentation
-------------
Official documentation for MLflow can be found at https://mlflow.org/docs/latest/index.html.

Roadmap
-------
The current MLflow Roadmap is available at https://github.com/mlflow/mlflow/milestone/3. We are
seeking contributions to all of our roadmap items with the ``help wanted`` label. Please see the
`Contributing`_ section for more information.

Community
---------
For help or questions about MLflow usage (e.g. "how do I do X?") see the `docs <https://mlflow.org/docs/latest/index.html>`_
or `Stack Overflow <https://stackoverflow.com/questions/tagged/mlflow>`_.

To report a bug, file a documentation issue, or submit a feature request, please open a GitHub issue.

For release announcements and other discussions, please subscribe to our mailing list (mlflow-users@googlegroups.com)
or join us on `Slack`_.

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
run the dev UI from source. We recommend running the UI from a different working directory,
specifying a backend store via the ``--backend-store-uri`` option. Alternatively, see
instructions for running the dev UI in the `contributor guide <CONTRIBUTING.rst>`_.


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
``examples/sklearn_logistic_regression/train.py`` that you can run as follows::

    $ python examples/sklearn_logistic_regression/train.py
    Score: 0.666
    Model saved in run <run-id>

    $ mlflow models serve --model-uri runs:/<run-id>/model

    $ curl -d '{"columns":[0],"index":[0,1],"data":[[1],[-1]]}' -H 'Content-Type: application/json'  localhost:5000/invocations


Contributing
------------
We happily welcome contributions to MLflow. We are also seeking contributions to items on the
`MLflow Roadmap <https://github.com/mlflow/mlflow/milestone/3>`_. Please see our
`contribution guide <CONTRIBUTING.rst>`_ to learn more about contributing to MLflow.
