==============
Backend Stores
==============

The backend store is where MLflow Tracking Server stores experiment and run metadata as well as
params, metrics, and tags for runs. MLflow supports two types of backend stores: *file store* and
*database-backed store*.

.. note::
    In order to use model registry functionality, you must run your server using a database-backed store.


Use ``--backend-store-uri`` to configure the type of backend store. You specify:

- A file store backend as ``./path_to_store`` or ``file:/path_to_store``
- A database-backed store as `SQLAlchemy database URI <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_.
  The database URI typically takes the format ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>``.
  MLflow supports the database dialects ``mysql``, ``mssql``, ``sqlite``, and ``postgresql``.
  Drivers are optional. If you do not specify a driver, SQLAlchemy uses a dialect's default driver.
  For example, ``--backend-store-uri sqlite:///mlflow.db`` would use a local SQLite database.


.. important::

    ``mlflow server`` will fail against a database-backed store with an out-of-date database schema.
    To prevent this, upgrade your database schema to the latest supported version using
    ``mlflow db upgrade [db_uri]``. Schema migrations can result in database downtime, may
    take longer on larger databases, and are not guaranteed to be transactional. You should always
    take a backup of your database prior to running ``mlflow db upgrade`` - consult your database's
    documentation for instructions on taking a backup.

.. note::
    ``2d6e25af4d3e_increase_max_param_val_length`` is a non-invertible migration script that increases 
    the param value length to 8k (but we limit param value max length to 6000 internally). Please be careful
    if you want to upgrade and backup your database before upgrading.


By default ``--backend-store-uri`` is set to the local ``./mlruns`` directory (the same as when
running ``mlflow run`` locally), but when running a server, make sure that this points to a
persistent (that is, non-ephemeral) file system location.


Deletion Behavior
~~~~~~~~~~~~~~~~~
In order to allow MLflow Runs to be restored, Run metadata and artifacts are not automatically removed
from the backend store or artifact store when a Run is deleted. The :ref:`mlflow gc <cli>` CLI is provided
for permanently removing Run metadata and artifacts for deleted runs.


SQLAlchemy Options
~~~~~~~~~~~~~~~~~~

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


File store performance
~~~~~~~~~~~~~~~~~~~~~~

MLflow will automatically try to use `LibYAML <https://pyyaml.org/wiki/LibYAML>`_ bindings if they are already installed.
However if you notice any performance issues when using *file store* backend, it could mean LibYAML is not installed on your system.
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
