.. _gateway-migration:

=================================
MLflow AI Gateway Migration Guide
=================================

The MLflow AI Gateway is deprecated and has been replaced by the `MLflow Deployments for LLMs <../deployments/index.html>`_.
This page is a migration guide for users of the MLflow AI Gateway.

Configuration YAML file
~~~~~~~~~~~~~~~~~~~~~~~

Deprecated:

.. code-block:: yaml

    routes:
      - name: chat
        route_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY

New:

.. code-block:: yaml

    endpoints:  # Renamed to "endpoints"
      - name: chat
        endpoint_type: llm/v1/chat  # Renamed to "endpoint_type"
        model:
          provider: openai
          name: gpt-4o-mini
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
    #      Renamed to "deployments start-server"

Querying the server
~~~~~~~~~~~~~~~~~~~

The fluent APIs have been replaced by the :py:class:`mlflow.deployments.MlflowDeploymentClient` APIs.
See the table below for the mapping between the deprecated and new APIs.

+-----------------------------------------+----------------------------------------------------+
| Deprecated                              | New                                                |
+=========================================+====================================================+
| mlflow.gateway.get_route(name)          | client.get_endpoint(name)                          |
+-----------------------------------------+----------------------------------------------------+
| mlflow.gateway.search_routes()          | client.list_endpoints()                            |
+-----------------------------------------+----------------------------------------------------+
| mlflow.gateway.query(name, data)        | client.predict(endpoint=name, inputs=data)         |
+-----------------------------------------+----------------------------------------------------+

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

The fluent APIs have been replaced by the :py:class:`mlflow.deployments.DatabricksDeploymentClient` APIs.
See the table below for the mapping between the deprecated and new APIs.

+-----------------------------------------+----------------------------------------------------+
| Deprecated                              | New                                                |
+=========================================+====================================================+
| mlflow.gateway.create_route(name, ...)  | client.create_endpoint(name, ...)                  |
+-----------------------------------------+----------------------------------------------------+
| mlflow.gateway.get_route(name)          | client.get_endpoint(name)                          |
+-----------------------------------------+----------------------------------------------------+
| mlflow.gateway.search_routes()          | client.list_endpoints()                            |
+-----------------------------------------+----------------------------------------------------+
| mlflow.gateway.delete_route(name)       | client.delete_endpoint(name)                       |
+-----------------------------------------+----------------------------------------------------+
| mlflow.gateway.get_limits(name)         | client.get_endpoint(name)["rate_limits"]           |
+-----------------------------------------+----------------------------------------------------+
| mlflow.gateway.set_limits(name, limits) | client.update_endpoint(name, limits)               |
+-----------------------------------------+----------------------------------------------------+
| mlflow.gateway.query(name, data)        | client.predict(endpoint=name, inputs=data)         |
+-----------------------------------------+----------------------------------------------------+

Deprecated:

.. code-block:: python

    import mlflow

    mlflow.gateway.set_gateway_uri("databricks")

    name = "chat"
    mlflow.gateway.create_route(name, ...)
    route = mlflow.gateway.get_route(name)
    routes = mlflow.gateway.search_routes()
    limits = mlflow.gateway.get_limits(name)
    mlflow.gateway.set_limits(name, limits)
    response = mlflow.gateway.query(
        route=name,
        data={
            "message": [
                {"role": "user", "content": "Hello"},
            ]
        },
    )
    mlflow.gateway.delete_route(name)

New:

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")

    name = "chat"
    client.create_endpoint(name, ...)
    endpoint = client.get_endpoint(name)
    endpoints = client.list_endpoints()
    limits = client.gen_endpoint(name)["rate_limits"]
    client.update_endpoint(name, {"rate_limits": limits})
    response = client.predict(
        endpoint=name,
        inputs={
            "message": [
                {"role": "user", "content": "Hello"},
            ]
        },
    )
    client.delete_endpoint(name)
