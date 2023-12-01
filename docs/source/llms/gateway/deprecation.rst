.. _gateway-deprecation:

=============================
MLflow AI Gateway Deprecation
=============================

The MLflow AI Gateway is deprecated and has been replaced by the `MLflow Deployments for LLMs <../deployments/index.html>`_.
This page describes how to migrate from the MLflow AI Gateway to the `MLflow Deployments for LLMs <../deployments/index.html>`_.

Configuration YAML file
~~~~~~~~~~~~~~~~~~~~~~~

Deprecated:

.. code-block:: yaml

    routes:
      - name: chat
        route_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-3.5-turbo
          config:
            openai_api_key: $OPENAI_API_KEY

New:

.. code-block:: yaml

    endpoints:  # Renamed to 'endpoints'
      - name: chat
        endpoint_type: llm/v1/chat  # Renamed to 'endpoint_type'
        model:
          provider: openai
          name: gpt-3.5-turbo
          config:
            openai_api_key: $OPENAI_API_KEY


Launching the server
~~~~~~~~~~~~~~~~~~~~

Deprecated:

.. code-block:: bash

    mlflow gateway start --config-path path/to/config.yaml

New:

.. code-block:: bash

    mlflow deployments start-server --config-path path/to/config.yaml
    #      ^^^^^^^^^^^^^^^^^^^^^^^^


Querying the server
~~~~~~~~~~~~~~~~~~~

The fluent APIs have been replaced by the methods of :py:class:`mlflow.deployments.DatabricksDeploymentClient`.
See the table below for the mapping between the deprecated and new APIs.

+----------------------------------+--------------------------------------+
| Deprecated                       | New                                  |
+==================================+======================================+
| mlflow.gateway.set_gateway_uri   | mlflow.deployments.get_deploy_client |
+----------------------------------+--------------------------------------+
| mlflow.gateway.get_route         | MlflowDeploymentClient.get_endpoint  |
+----------------------------------+--------------------------------------+
| mlflow.gateway.search_routes     | MlflowDeploymentClient.list_endpoints|
+----------------------------------+--------------------------------------+
| mlflow.gateway.query             | MlflowDeploymentClient.predict       |
+----------------------------------+--------------------------------------+

Deprecated:

.. code-block:: python

    import mlflow

    mlflow.gateway.set_gateway_uri("http://localhost:5000")

    route = mlflow.gateway.get_route("chat")
    routes = mlflow.gateway.search_routes()
    response = mlflow.gateway.query(
        route="chat",
        data={
            "message": [
                {"role": "user", "content": "Hello"},
            ]
        },
    )

New:

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("http://localhost:5000")
    endpoint = client.get_endpoint("chat")
    endpoints = client.list_endpoints()
    response = client.predict(
        endpoint="chat",
        inputs={
            "message": [
                {"role": "user", "content": "Hello"},
            ]
        },
    )


Databricks
~~~~~~~~~~

The fluent APIs have been replaced by the methods of :py:class:`mlflow.deployments.DatabricksDeploymentClient`.
See the table below for the mapping between the deprecated and new APIs.

+----------------------------------+-----------------------------------------------+
| Deprecated                       | New                                           |
+==================================+===============================================+
| mlflow.gateway.set_gateway_uri   | databricks.deployments.get_deploy_client      |
+----------------------------------+-----------------------------------------------+
| mlflow.gateway.get_route         | DatabricksDeploymentClient.get_endpoint       |
+----------------------------------+-----------------------------------------------+
| mlflow.gateway.search_routes     | DatabricksDeploymentClient.list_endpoints     |
+----------------------------------+-----------------------------------------------+
| mlflow.gateway.get_limits        | DatabricksDeploymentClient.get_endpoint       |
+----------------------------------+-----------------------------------------------+
| mlflow.gateway.set_limits        | DatabricksDeploymentClient.update_endpoint    |
+----------------------------------+-----------------------------------------------+
| mlflow.gateway.query             | DatabricksDeploymentClient.predict            |
+----------------------------------+-----------------------------------------------+

Deprecated:

.. code-block:: python

    import mlflow

    route = "chat"
    mlflow.gateway.set_gateway_uri("databricks")
    route = mlflow.gateway.get_route(route)
    routes = mlflow.gateway.search_routes()
    limits = mlflow.gateway.get_limits(route)
    mlflow.gateway.set_limits(
        route, [{"key": "user", "renewal_period": "minute", "calls": 50}]
    )
    response = mlflow.gateway.query(
        route="chat",
        data={
            "message": [
                {"role": "user", "content": "Hello"},
            ]
        },
    )

New:

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    endpoint = client.get_endpoint("chat")
    endpoints = client.list_endpoints()
    limits = client.gen_endpoint(endpoint)["rate_limits"]
    client.update_endpoint(
        endpoint,
        {"rate_limits": [{"key": "user", "renewal_period": "minute", "calls": 50}]},
    )
    response = client.predict(
        endpoint="chat",
        inputs={
            "message": [
                {"role": "user", "content": "Hello"},
            ]
        },
    )
