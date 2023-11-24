.. _gateway-deprecation:

=============================
MLflow AI Gateway Deprecation
=============================

The MLflow AI Gateway is deprecated and has been replaced by the MLflow deployments API.
This page describes how to migrate from the MLflow AI Gateway to the MLflow deployments API.

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

See also
~~~~~~~~

- :py:class:`mlflow.deployments.MLflowDeploymentClient`
