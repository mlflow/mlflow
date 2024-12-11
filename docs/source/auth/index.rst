.. _auth:

=====================
MLflow Authentication
=====================

.. note::
    This feature is still experimental and may change in a future release without warning.

MLflow supports basic HTTP authentication to enable access control over experiments and registered models.
Once enabled, any visitor will be required to login before they can view any resource from the Tracking Server.

.. contents:: Table of Contents
  :local:
  :depth: 2

MLflow Authentication provides Python and REST API for managing users and permissions. 

.. toctree::
  :glob:
  :maxdepth: 1

  *

Overview
========

To enable MLflow authentication, launch the MLflow UI with the following command:

.. code-block:: bash

    mlflow server --app-name basic-auth


Server admin can choose to disable this feature anytime by restarting the server without the ``app-name`` flag. 
Any users and permissions created will be persisted on a SQL database and will be back in service once the feature is re-enabled.

Due to the nature of HTTP authentication, it is only supported on a remote Tracking Server, where users send
requests to the server via REST APIs.

How It Works
============

Permissions
-----------

The available permissions are:

.. list-table::
   :widths: 10 10 10 10 10
   :header-rows: 1

   * - Permission
     - Can read
     - Can update
     - Can delete
     - Can manage
   * - ``READ``
     - Yes
     - No
     - No
     - No
   * - ``EDIT``
     - Yes
     - Yes
     - No
     - No
   * - ``MANAGE``
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``NO_PERMISSIONS``
     - No
     - No
     - No
     - No

The default permission for all users is ``READ``. It can be changed in the :ref:`configuration <configuration>` file.

Permissions can be granted on individual resources for each user. 
Supported resources include ``Experiment`` and ``Registered Model``.
To access an API endpoint, an user must have the required permission.
Otherwise, a ``403 Forbidden`` response will be returned.

Required Permissions for accessing experiments:

.. list-table::
   :widths: 10 10 10 10
   :header-rows: 1

   * - API
     - Endpoint
     - Method
     - Required permission
   * - :ref:`Create Experiment <mlflowMlflowServicecreateExperiment>`
     - ``2.0/mlflow/experiments/create``
     - ``POST``
     - None
   * - :ref:`Get Experiment <mlflowMlflowServicegetExperiment>`
     - ``2.0/mlflow/experiments/get``
     - ``GET``
     - can_read
   * - :ref:`Get Experiment By Name <mlflowMlflowServicegetExperimentByName>`
     - ``2.0/mlflow/experiments/get-by-name``
     - ``GET``
     - can_read
   * - :ref:`Delete Experiment <mlflowMlflowServicedeleteExperiment>`
     - ``2.0/mlflow/experiments/delete``
     - ``POST``
     - can_delete
   * - :ref:`Restore Experiment <mlflowMlflowServicerestoreExperiment>`
     - ``2.0/mlflow/experiments/restore``
     - ``POST``
     - can_delete
   * - :ref:`Update Experiment <mlflowMlflowServiceupdateExperiment>`
     - ``2.0/mlflow/experiments/update``
     - ``POST``
     - can_update
   * - :ref:`Search Experiments <mlflowMlflowServicesearchExperiments>`
     - ``2.0/mlflow/experiments/search``
     - ``POST``
     - None
   * - :ref:`Search Experiments <mlflowMlflowServicesearchExperiments>`
     - ``2.0/mlflow/experiments/search``
     - ``GET``
     - None
   * - :ref:`Set Experiment Tag <mlflowMlflowServicesetExperimentTag>`
     - ``2.0/mlflow/experiments/set-experiment-tag``
     - ``POST``
     - can_update
   * - :ref:`Create Run <mlflowMlflowServicecreateRun>`
     - ``2.0/mlflow/runs/create``
     - ``POST``
     - can_update
   * - :ref:`Get Run <mlflowMlflowServicegetRun>`
     - ``2.0/mlflow/runs/get``
     - ``GET``
     - can_read
   * - :ref:`Update Run <mlflowMlflowServiceupdateRun>`
     - ``2.0/mlflow/runs/update``
     - ``POST``
     - can_update
   * - :ref:`Delete Run <mlflowMlflowServicedeleteRun>`
     - ``2.0/mlflow/runs/delete``
     - ``POST``
     - can_delete
   * - :ref:`Restore Run <mlflowMlflowServicerestoreRun>`
     - ``2.0/mlflow/runs/restore``
     - ``POST``
     - can_delete
   * - :ref:`Search Runs <mlflowMlflowServicesearchRuns>`
     - ``2.0/mlflow/runs/search``
     - ``POST``
     - None
   * - :ref:`Set Tag <mlflowMlflowServicesetTag>`
     - ``2.0/mlflow/runs/set-tag``
     - ``POST``
     - can_update
   * - :ref:`Delete Tag <mlflowMlflowServicedeleteTag>`
     - ``2.0/mlflow/runs/delete-tag``
     - ``POST``
     - can_update
   * - :ref:`Log Metric <mlflowMlflowServicelogMetric>`
     - ``2.0/mlflow/runs/log-metric``
     - ``POST``
     - can_update
   * - :ref:`Log Param <mlflowMlflowServicelogParam>`
     - ``2.0/mlflow/runs/log-parameter``
     - ``POST``
     - can_update
   * - :ref:`Log Batch <mlflowMlflowServicelogBatch>`
     - ``2.0/mlflow/runs/log-batch``
     - ``POST``
     - can_update
   * - :ref:`Log Model <mlflowMlflowServicelogModel>`
     - ``2.0/mlflow/runs/log-model``
     - ``POST``
     - can_update
   * - :ref:`List Artifacts <mlflowMlflowServicelistArtifacts>`
     - ``2.0/mlflow/artifacts/list``
     - ``GET``
     - can_read
   * - :ref:`Get Metric History <mlflowMlflowServicegetMetricHistory>`
     - ``2.0/mlflow/metrics/get-history``
     - ``GET``
     - can_read

Required Permissions for accessing registered models:

.. list-table::
   :widths: 10 10 10 10
   :header-rows: 1

   * - API
     - Endpoint
     - Method
     - Required permission
   * - :ref:`Create Registered Model <mlflowModelRegistryServicecreateRegisteredModel>`
     - ``2.0/mlflow/registered-models/create``
     - ``POST``
     - None
   * - :ref:`Rename Registered Model <mlflowModelRegistryServicerenameRegisteredModel>`
     - ``2.0/mlflow/registered-models/rename``
     - ``POST``
     - can_update
   * - :ref:`Update Registered Model <mlflowModelRegistryServiceupdateRegisteredModel>`
     - ``2.0/mlflow/registered-models/update``
     - ``PATCH``
     - can_update
   * - :ref:`Delete Registered Model <mlflowModelRegistryServicedeleteRegisteredModel>`
     - ``2.0/mlflow/registered-models/delete``
     - ``DELETE``
     - can_delete
   * - :ref:`Get Registered Model <mlflowModelRegistryServicegetRegisteredModel>`
     - ``2.0/mlflow/registered-models/get``
     - ``GET``
     - can_read
   * - :ref:`Search Registered Models <mlflowModelRegistryServicesearchRegisteredModels>`
     - ``2.0/mlflow/registered-models/search``
     - ``GET``
     - None
   * - :ref:`Get Latest Versions <mlflowModelRegistryServicegetLatestVersions>`
     - ``2.0/mlflow/registered-models/get-latest-versions``
     - ``POST``
     - can_read
   * - :ref:`Get Latest Versions <mlflowModelRegistryServicegetLatestVersions>`
     - ``2.0/mlflow/registered-models/get-latest-versions``
     - ``GET``
     - can_read
   * - :ref:`Set Registered Model Tag <mlflowModelRegistryServicesetRegisteredModelTag>`
     - ``2.0/mlflow/registered-models/set-tag``
     - ``POST``
     - can_update
   * - :ref:`Delete Registered Model Tag <mlflowModelRegistryServicedeleteRegisteredModelTag>`
     - ``2.0/mlflow/registered-models/delete-tag``
     - ``DELETE``
     - can_update
   * - :ref:`Set Registered Model Alias <mlflowModelRegistryServicesetRegisteredModelAlias>`
     - ``2.0/mlflow/registered-models/alias``
     - ``POST``
     - can_update
   * - :ref:`Delete Registered Model Alias <mlflowModelRegistryServicedeleteRegisteredModelAlias>`
     - ``2.0/mlflow/registered-models/alias``
     - ``DELETE``
     - can_delete
   * - :ref:`Get Model Version By Alias <mlflowModelRegistryServicegetModelVersionByAlias>`
     - ``2.0/mlflow/registered-models/alias``
     - ``GET``
     - can_read
   * - :ref:`Create Model Version <mlflowModelRegistryServicecreateModelVersion>`
     - ``2.0/mlflow/model-versions/create``
     - ``POST``
     - can_update
   * - :ref:`Update Model Version <mlflowModelRegistryServiceupdateModelVersion>`
     - ``2.0/mlflow/model-versions/update``
     - ``PATCH``
     - can_update
   * - :ref:`Transition Model Version Stage <mlflowModelRegistryServicetransitionModelVersionStage>`
     - ``2.0/mlflow/model-versions/transition-stage``
     - ``POST``
     - can_update
   * - :ref:`Delete Model Version <mlflowModelRegistryServicedeleteModelVersion>`
     - ``2.0/mlflow/model-versions/delete``
     - ``DELETE``
     - can_delete
   * - :ref:`Get Model Version <mlflowModelRegistryServicegetModelVersion>`
     - ``2.0/mlflow/model-versions/get``
     - ``GET``
     - can_read
   * - :ref:`Search Model Versions <mlflowModelRegistryServicesearchModelVersions>`
     - ``2.0/mlflow/model-versions/search``
     - ``GET``
     - None
   * - :ref:`Get Model Version Download Uri <mlflowModelRegistryServicegetModelVersionDownloadUri>`
     - ``2.0/mlflow/model-versions/get-download-uri``
     - ``GET``
     - can_read
   * - :ref:`Set Model Version Tag <mlflowModelRegistryServicesetModelVersionTag>`
     - ``2.0/mlflow/model-versions/set-tag``
     - ``POST``
     - can_update
   * - :ref:`Delete Model Version Tag <mlflowModelRegistryServicedeleteModelVersionTag>`
     - ``2.0/mlflow/model-versions/delete-tag``
     - ``DELETE``
     - can_delete

MLflow Authentication introduces several new API endpoints to manage users and permissions.

.. list-table::
   :widths: 10 10 10 10
   :header-rows: 1

   * - API
     - Endpoint
     - Method
     - Required permission
   * - :ref:`Create User <mlflowAuthServicecreateUser>`
     - ``2.0/mlflow/users/create``
     - ``POST``
     - None
   * - :ref:`Get User <mlflowAuthServicegetUser>`
     - ``2.0/mlflow/users/get``
     - ``GET``
     - Only readable by that user
   * - :ref:`Update User Password <mlflowAuthServiceupdateUserPassword>`
     - ``2.0/mlflow/users/update-password``
     - ``PATCH``
     - Only updatable by that user
   * - :ref:`Update User Admin <mlflowAuthServiceupdateUserAdmin>`
     - ``2.0/mlflow/users/update-admin``
     - ``PATCH``
     - Only admin
   * - :ref:`Delete User <mlflowAuthServicedeleteUser>`
     - ``2.0/mlflow/users/delete``
     - ``DELETE``
     - Only admin
   * - :ref:`Create Experiment Permission <mlflowAuthServicecreateExperimentPermission>`
     - ``2.0/mlflow/experiments/permissions/create``
     - ``POST``
     - can_manage
   * - :ref:`Get Experiment Permission <mlflowAuthServicegetExperimentPermission>`
     - ``2.0/mlflow/experiments/permissions/get``
     - ``GET``
     - can_manage
   * - :ref:`Update Experiment Permission <mlflowAuthServiceupdateExperimentPermission>`
     - ``2.0/mlflow/experiments/permissions/update``
     - ``PATCH``
     - can_manage
   * - :ref:`Delete Experiment Permission <mlflowAuthServicedeleteExperimentPermission>`
     - ``2.0/mlflow/experiments/permissions/delete``
     - ``DELETE``
     - can_manage
   * - :ref:`Create Registered Model Permission <mlflowAuthServicecreateRegisteredModelPermission>`
     - ``2.0/mlflow/registered-models/permissions/create``
     - ``POST``
     - can_manage
   * - :ref:`Get Registered Model Permission <mlflowAuthServicegetRegisteredModelPermission>`
     - ``2.0/mlflow/registered-models/permissions/get``
     - ``GET``
     - can_manage
   * - :ref:`Update Registered Model Permission <mlflowAuthServiceupdateRegisteredModelPermission>`
     - ``2.0/mlflow/registered-models/permissions/update``
     - ``PATCH``
     - can_manage
   * - :ref:`Delete Registered Model Permission <mlflowAuthServicedeleteRegisteredModelPermission>`
     - ``2.0/mlflow/registered-models/permissions/delete``
     - ``DELETE``
     - can_manage

Some APIs will also have their behaviour modified.
For example, the creator of an experiment will automatically be granted ``MANAGE`` permission
on that experiment, so that the creator can grant or revoke other users' access to that experiment.

.. list-table::
   :widths: 10 10 10 10
   :header-rows: 1

   * - API
     - Endpoint
     - Method
     - Effect
   * - :ref:`Create Experiment <mlflowMlflowServicecreateExperiment>`
     - ``2.0/mlflow/experiments/create``
     - ``POST``
     - Automatically grants ``MANAGE`` permission to the creator.
   * - :ref:`Create Registered Model <mlflowModelRegistryServicecreateRegisteredModel>`
     - ``2.0/mlflow/registered-models/create``
     - ``POST``
     - Automatically grants ``MANAGE`` permission to the creator.
   * - :ref:`Search Experiments <mlflowMlflowServicesearchExperiments>`
     - ``2.0/mlflow/experiments/search``
     - ``POST``
     - Only returns experiments which the user has ``READ`` permission on.
   * - :ref:`Search Experiments <mlflowMlflowServicesearchExperiments>`
     - ``2.0/mlflow/experiments/search``
     - ``GET``
     - Only returns experiments which the user has ``READ`` permission on.
   * - :ref:`Search Runs <mlflowMlflowServicesearchRuns>`
     - ``2.0/mlflow/runs/search``
     - ``POST``
     - Only returns experiments which the user has ``READ`` permission on.
   * - :ref:`Search Registered Models <mlflowModelRegistryServicesearchRegisteredModels>`
     - ``2.0/mlflow/registered-models/search``
     - ``GET``
     - Only returns registered models which the user has ``READ`` permission on.
   * - :ref:`Search Model Versions <mlflowModelRegistryServicesearchModelVersions>`
     - ``2.0/mlflow/model-versions/search``
     - ``GET``
     - Only returns registered models which the user has ``READ`` permission on.


Permissions Database
--------------------

All users and permissions are stored in a database in ``basic_auth.db``, relative to the directory where MLflow server is launched.
The location can be changed in the :ref:`configuration <configuration>` file. To run migrations, use the following command:

.. code-block::

    python -m mlflow.server.auth db upgrade --url <database_url>

Admin Users
-----------

Admin users have unrestricted access to all MLflow resources,
**including creating or deleting users, updating password and admin status of other users,
granting or revoking permissions from other users, and managing permissions for all 
MLflow resources,** even if ``NO_PERMISSIONS`` is explicitly set to that admin account.

MLflow has a built-in admin user that will be created the first time that the MLflow authentication feature is enabled.

.. note::
    It is recommended that you update the default admin password as soon as possible after creation.

The default admin user credentials are as follows:


.. list-table::
   :widths: 10 10
   :header-rows: 1

   * - Username
     - Password
   * - ``admin``
     - ``password``


Multiple admin users can exist by promoting other users to admin, using the ``2.0/mlflow/users/update-admin`` endpoint.

.. code-block:: bash
    :caption: Example

    # authenticate as built-in admin user
    export MLFLOW_TRACKING_USERNAME=admin
    export MLFLOW_TRACKING_PASSWORD=password
 
.. code-block:: python

    from mlflow.server import get_app_client

    tracking_uri = "http://localhost:5000/"

    auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
    auth_client.create_user(username="user1", password="pw1")
    auth_client.update_user_admin(username="user1", is_admin=True)


Managing Permissions
--------------------

MLflow provides :ref:`REST APIs <mlflowAuthServiceCreateUser>` and a client class 
:py:func:`AuthServiceClient<mlflow.server.auth.client.AuthServiceClient>` to manage users and permissions.
To instantiate ``AuthServiceClient``, it is recommended that you use :py:func:`mlflow.server.get_app_client`.

.. code-block:: bash
    :caption: Example

    export MLFLOW_TRACKING_USERNAME=admin
    export MLFLOW_TRACKING_PASSWORD=password

.. code-block:: python

    from mlflow import MlflowClient
    from mlflow.server import get_app_client

    tracking_uri = "http://localhost:5000/"

    auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
    auth_client.create_user(username="user1", password="pw1")
    auth_client.create_user(username="user2", password="pw2")

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(name="experiment")

    auth_client.create_experiment_permission(
        experiment_id=experiment_id, username="user2", permission="MANAGE"
    )


Authenticating to MLflow
========================

Using MLflow UI
---------------

When a user first visits the MLflow UI on a browser, they will be prompted to login. 
There is no limit to how many login attempts can be made.

Currently, MLflow UI does not display any information about the current user.
Once a user is logged in, the only way to log out is to close the browser.

    .. image:: ../_static/images/auth_prompt.png

Using Environment Variables
---------------------------

MLflow provides two environment variables for authentication: ``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD``.
To use basic authentication, you must set both environment variables.

.. code-block:: bash

    export MLFLOW_TRACKING_USERNAME=username
    export MLFLOW_TRACKING_PASSWORD=password


.. code-block:: python

    import mlflow

    mlflow.set_tracking_uri("https://<mlflow_tracking_uri>/")
    with mlflow.start_run():
        ...

Using Credentials File
----------------------

You can save your credentials in a file to remove the need for setting environment variables every time.
The credentials should be saved in ``~/.mlflow/credentials`` using ``INI`` format. Note that the password 
will be stored unencrypted on disk, and is protected only by filesystem permissions.

If the environment variables ``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD`` are configured,
they override any credentials provided in the credentials file.

.. code-block:: ini
    :caption: Credentials file format

    [mlflow]
    mlflow_tracking_username = username
    mlflow_tracking_password = password

Using REST API
--------------

A user can authenticate using the HTTP ``Authorization`` request header.
See https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication for more information.

In Python, you can use the ``requests`` library:

.. code-block:: python

    import requests

    response = requests.get(
        "https://<mlflow_tracking_uri>/",
        auth=("username", "password"),
    )


Creating a New User
===================

.. important::
    To create a new user, you are required to authenticate with admin privileges.

Using MLflow UI
---------------

MLflow UI provides a simple page for creating new users at ``<tracking_uri>/signup``.

    .. image:: ../_static/images/auth_signup_form.png

Using REST API
--------------

Alternatively, you can send ``POST`` requests to the Tracking Server endpoint ``2.0/users/create``.

In Python, you can use the ``requests`` library:

.. code-block:: python

    import requests

    response = requests.post(
        "https://<mlflow_tracking_uri>/api/2.0/mlflow/users/create",
        json={
            "username": "username",
            "password": "password",
        },
    )

Using MLflow AuthServiceClient
------------------------------

MLflow :py:func:`AuthServiceClient<mlflow.server.auth.client.AuthServiceClient>`
provides a function to create new users easily.

.. code-block:: python

    import mlflow

    auth_client = mlflow.server.get_app_client(
        "basic-auth", tracking_uri="https://<mlflow_tracking_uri>/"
    )
    auth_client.create_user(username="username", password="password")

.. _configuration:

Configuration
=============

Authentication configuration is located at ``mlflow/server/auth/basic_auth.ini``:

.. list-table::
   :widths: 10 10
   :header-rows: 1

   * - Variable
     - Description
   * - ``default_permission``
     - Default permission on all resources
   * - ``database_uri``
     - Database location to store permission and user data
   * - ``admin_username``
     - Default admin username if the admin is not already created
   * - ``admin_password``
     - Default admin password if the admin is not already created
   * - ``authorization_function``
     - Function to authenticate requests

Alternatively, assign the environment variable ``MLFLOW_AUTH_CONFIG_PATH`` to point
to your custom configuration file.

The ``authorization_function`` setting supports pluggable authentication methods
if you want to use another authentication method than HTTP basic auth. The value
specifies ``module_name:function_name``. The function has the following signature:

    .. code-block:: python

        def authenticate_request() -> Union[Authorization, Response]:
            ...

The function should return a ``werkzeug.datastructures.Authorization`` object if
the request is authenticated, or a ``Response`` object (typically
``401: Unauthorized``) if the request is not authenticated. For an example of how
to implement a custom authentication method, see ``tests/server/auth/jwt_auth.py``.
**NOTE:** This example is not intended for production use.

Connecting to a Centralized Database
====================================

By default, MLflow Authentication uses a local SQLite database to store user and permission data.
In the case of a multi-node deployment, it is recommended to use a centralized database to store this data.

To connect to a centralized database, you can set the ``database_uri`` configuration variable to the database URL.

.. code-block:: ini
    :caption: Example: ``/path/to/my_auth_config.ini``

    [mlflow]
    database_uri = postgresql://username:password@hostname:port/database

Then, start the MLflow server with the ``MLFLOW_AUTH_CONFIG_PATH`` environment variable
set to the path of your configuration file.

.. code-block:: bash

    MLFLOW_AUTH_CONFIG_PATH=/path/to/my_auth_config.ini mlflow server --app-name basic-auth

The database must be created before starting the MLflow server. The database schema will be created automatically
when the server starts.

Custom Authentication
=====================

MLflow authentication is designed to be extensible. If your organization desires more advanced authentication logic 
(e.g., token-based authentication), it is possible to install a third party plugin or to create your own plugin.

Your plugin should be an installable Python package.
It should include an app factory that extends the MLflow app and, optionally, implement a client to manage permissions.
The app factory function name will be passed to the ``--app`` argument in Flask CLI.
See https://flask.palletsprojects.com/en/latest/cli/#application-discovery for more information.

.. code-block:: python
    :caption: Example: ``my_auth/__init__.py``

    from flask import Flask
    from mlflow.server import app


    def create_app(app: Flask = app):
        app.add_url_rule(...)
        return app


    class MyAuthClient:
        ...

Then, the plugin should be installed in your Python environment:

.. code-block:: bash

    pip install my_auth

Then, register your plugin in ``mlflow/setup.py``:

.. code-block:: python

    setup(
        ...,
        entry_points="""
            ...

            [mlflow.app]
            my-auth=my_auth:create_app

            [mlflow.app.client]
            my-auth=my_auth:MyAuthClient
        """,
    )

Then, you can start the MLflow server:

.. code-block:: bash

    mlflow server --app-name my-auth
