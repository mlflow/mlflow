==================
Deployments Server
==================

This page demonstrates how to use the Deployments server, and how to interact with it using
:py:class:`MlflowDeploymentClient <mlflow.deployments.MlflowDeploymentClient>`.

Prerequisites
-------------

Create an OpenAI API key and set it as an environment variable:

.. code-block:: bash

    export OPENAI_API_KEY=<your-api-key>


Create a config file
--------------------

.. code-block:: yaml

    # /path/to/config.yaml

    endpoints:
      - name: chat
        endpoint_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-3.5-turbo
          config:
            openai_api_key: $OPENAI_API_KEY


Start the Deployments server
----------------------------

.. code-block:: bash

    mlflow deployments start-server --config-path /path/to/config.yaml


Create `MlflowDeploymentClient`
-------------------------------

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("http://localhost:5000")


Read endpoints
--------------

.. code-block:: python

    name = "chat"
    print(client.list_endpoints())
    print(client.get_endpoint(name))


Query the endpoint
------------------

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
