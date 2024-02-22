.. _backend-stores:

==============
Backend Stores
==============

The backend store is a core component in `MLflow Tracking <../index.html>`_ where MLflow stores metadata for 
:ref:`Runs <runs>` and experiments such as:

* Run ID
* Start & end time
* Parameters
* Metrics
* Code version (only if you launch runs from an :ref:`MLflow Project <projects>`).
* Source file name (only if you launch runs from an :ref:`MLflow Project <projects>`).

Note that large model artifacts such as model weight files are stored in `artifact store <artifacts-stores.html>`_.

.. _where_runs_are_recorded:

Configure Backend Store
=======================
By default, MLflow stores metadata in local files in the ``./mlruns`` directory, but MLflow can store metadata to databases as well.
You can configure the location by passing the desired **tracking URI** to MLflow, via either of the following methods:

* Set the ``MLFLOW_TRACKING_URI`` environment variable.
* Call :py:func:`mlflow.set_tracking_uri` in your code.
* If you are running a :ref:`Tracking Server <tracking_server>`, you can set the ``tracking_uri`` option when starting the server, like ``mlflow server --backend-store-uri sqlite:///mydb.sqlite``

Continue to the next section for the supported format of tracking URLs.
Also visit :ref:`this guidance <tracking_setup>` for how to set up the backend store properly for your workflow.

Supported Store Types
=====================
MLflow supports the following types of tracking URI for backend stores:

- Local file path (specified as ``file:/my/local/dir``), where data is just directly stored locally to a system disk where your code is executing.
- A Database, encoded as ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>``. MLflow supports the dialects ``mysql``, ``mssql``, ``sqlite``, and ``postgresql``. For more details, see `SQLAlchemy database uri <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_.
- HTTP server (specified as ``https://my-server:5000``), which is a server hosting an :ref:`MLflow tracking server <tracking_server>`.
- Databricks workspace (specified as ``databricks`` or as ``databricks://<profileName>``, a `Databricks CLI profile <https://github.com/databricks/databricks-cli#installation>`_).
  Refer to Access the MLflow tracking server from outside Databricks `[AWS] <http://docs.databricks.com/applications/mlflow/access-hosted-tracking-server.html>`_
  `[Azure] <http://docs.microsoft.com/azure/databricks/applications/mlflow/access-hosted-tracking-server>`_, or `the quickstart <../getting-started/intro-quickstart/index.html>`_ to
  easily get started with hosted MLflow on Databricks Community Edition.

.. important::
    In order to use :ref:`Model Registry <registry>` functionality, you must run your server using a database-backed store. See :ref:`this FAQ <tracking-with-model-registry>` for more information.

.. important::
    ``mlflow server`` will fail against a database-backed store with an out-of-date database schema.
    To prevent this, upgrade your database schema to the latest supported version using
    ``mlflow db upgrade [db_uri]``. Schema migrations can result in database downtime, may
    take longer on larger databases, and are not guaranteed to be transactional. You should always
    take a backup of your database prior to running ``mlflow db upgrade`` - consult your database's
    documentation for instructions on taking a backup.

.. note::
    In Sep 2023, we increased the max length for params recorded in a Run from 500 to 8k (but we limit param value max length to 6000 internally).
    `mlflow/2d6e25af4d3e_increase_max_param_val_length <https://github.com/mlflow/mlflow/blob/master/mlflow/store/db_migrations/versions/2d6e25af4d3e_increase_max_param_val_length.py>`_
    is a non-invertible migration script that increases the cap in existing database to 8k . Please be careful if you want to upgrade and backup your database before upgrading.


Deletion Behavior
=================
In order to allow MLflow Runs to be restored, Run metadata and artifacts are not automatically removed
from the backend store or artifact store when a Run is deleted. The :ref:`mlflow gc <cli>` CLI is provided
for permanently removing Run metadata and artifacts for deleted runs.


SQLAlchemy Options
==================
You can inject some `SQLAlchemy connection pooling options <https://docs.sqlalchemy.org/en/latest/core/pooling.html>`_ using environment variables.

+-----------------------------------------+-----------------------------+
| MLflow Environment Variable             | SQLAlchemy QueuePool Option |
+-----------------------------------------+-----------------------------+
| ``MLFLOW_SQLALCHEMYSTORE_POOL_SIZE``    | ``pool_size``               |
+-----------------------------------------+-----------------------------+
| ``MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE`` | ``pool_recycle``            |
+-----------------------------------------+-----------------------------+
| ``MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW`` | ``max_overflow``            |
+-----------------------------------------+-----------------------------+


File Store Performance
======================

MLflow will automatically try to use `LibYAML <https://pyyaml.org/wiki/LibYAML>`_ bindings if they are already installed.
However, if you notice any performance issues when using *file store* backend, it could mean LibYAML is not installed on your system.
On Linux or Mac you can easily install it using your system package manager:

.. code-block:: sh

    # On Ubuntu/Debian
    apt-get install libyaml-cpp-dev libyaml-dev

    # On macOS using Homebrew
    brew install yaml-cpp libyaml

After installing LibYAML, you need to reinstall PyYAML:

.. code-block:: sh

    # Reinstall PyYAML
    pip --no-cache-dir install --force-reinstall -I pyyaml
