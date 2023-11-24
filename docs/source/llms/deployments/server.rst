===============================================
MLflow Deployments for LLMs: Deployments Server
===============================================

This page demonstrates how to use the MLflow deployments API for LLMs.

Prerequisites
-------------

Create an OpenAI API key and set it as an environment variable:

.. code-block:: bash

    export OPENAI_API_KEY=<your-api-key>


Create a config file
--------------------

.. code-block:: yaml

    # /path/to/config.yaml

    routes:
    - name: chat
        route_type: llm/v1/chat
        model:
        provider: openai
        name: gpt-3.5-turbo
        config:
            openai_api_key: $OPENAI_API_KEY


Start the Deployments server
----------------------------

.. code-block:: bash

    mlflow deployments start-server --config-path /path/to/config.yaml


Create a client
---------------

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("http://localhost:5000")


See :py:class:`mlflow.deployments.MlflowDeploymentClient` for what operations the client supports.


Reading endpoints
-----------------

.. code-block:: python

    name = "chat"
    print(client.list_endpoints())
    print(client.get_endpoint(name))


Querying the endpoint
---------------------

.. code-block:: python

    print(
        client.predict(
            endpoint=name,
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 128,
            },
        ),
    )
