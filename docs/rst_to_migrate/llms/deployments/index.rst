



.. note::

  

+--------------------------+----------+------------------------------+-------------------------------------------------------+
| Configuration Parameter  | Required | Default                      | Description                                           |
+==========================+==========+==============================+=======================================================+
| **aws_region**           | No       | AWS_REGION/AWS_DEFAULT_REGION| The AWS Region to use for bedrock access.             |
|                          |          |                              |                                                       |
+--------------------------+----------+------------------------------+-------------------------------------------------------+
| **aws_secret_access_key**| Yes      |                              | AWS secret access key for the IAM user/role           |
|                          |          |                              | authorized to use bedrock                             |
+--------------------------+----------+------------------------------+-------------------------------------------------------+
| **aws_access_key_id**    | Yes      |                              | AWS access key ID for the IAM user/role               |
|                          |          |                              | authorized to use Bedrock                             |
+--------------------------+----------+------------------------------+-------------------------------------------------------+
| **aws_session_token**    | No       | None                         | Optional session token, if required                   |
+--------------------------+----------+------------------------------+-------------------------------------------------------+

Alternatively, for role-based authentication, an Amazon Bedrock endpoint can be defined and initialized with an a IAM Role  ARN that is authorized to access Bedrock.  The MLflow AI Gateway will attempt to assume this role with using the standard credential provider chain and will renew the role credentials if they have expired.

+--------------------------+----------+------------------------------+-------------------------------------------------------+
| Configuration Parameter  | Required | Default                      | Description                                           |
+==========================+==========+==============================+=======================================================+
| **aws_region**           | No       | AWS_REGION/AWS_DEFAULT_REGION| The AWS Region to use for bedrock access.             |
|                          |          |                              |                                                       |
+--------------------------+----------+------------------------------+-------------------------------------------------------+
| **aws_role_arn**         | Yes      |                              | An AWS role authorized to use Bedrock.  The standard  |
|                          |          |                              | credential provider chain *must* be able to find      |
|                          |          |                              | credentials authorized to assume this role.           |
+--------------------------+----------+------------------------------+-------------------------------------------------------+
|**session_length_seconds**| No       | 900                          | The length of session to request.                     |
+--------------------------+----------+------------------------------+-------------------------------------------------------+

MLflow Model Serving
++++++++++++++++++++

+-------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter | Required | Default                  | Description                                           |
+=========================+==========+==========================+=======================================================+
| **model_server_url**    | Yes      | N/A                      | This is the url of the MLflow Model Server.           |
+-------------------------+----------+--------------------------+-------------------------------------------------------+

Note that with MLflow model serving, the ``name`` parameter for the ``model`` definition is not used for validation and is only present for reference purposes. This alias can be
useful for understanding a particular version or endpoint definition that was used that can be referenced back to a deployed model. You may choose any name that you wish, provided that
it is JSON serializable.

Azure OpenAI
++++++++++++

Azure provides two different mechanisms for integrating with OpenAI, each corresponding to a different type of security validation. One relies on an access token for validation, referred to as ``azure``, while the other uses Azure Active Directory (Azure AD) integration for authentication, termed as ``azuread``.

To match your user's interaction and security access requirements, adjust the ``openai_api_type`` parameter to represent the preferred security validation model. This will ensure seamless interaction and reliable security for your Azure-OpenAI integration.

+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| Configuration Parameter    | Required | Default | Description                                                                                   |
+============================+==========+=========+===============================================================================================+
| **openai_api_key**         | Yes      |         | This is the API key for the Azure OpenAI service.                                             |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_type**        | Yes      |         | This field must be either ``azure`` or ``azuread`` depending on the security access protocol. |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_base**        | Yes      |         | This is the base URL for the Azure OpenAI API service provided by Azure.                      |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_version**     | Yes      |         | The version of the Azure OpenAI service to utilize, specified by a date.                      |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_deployment_name** | Yes      |         | This is the name of the deployment resource for the Azure OpenAI service.                     |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_organization**    | No       |         | This is an optional field to specify the organization in OpenAI.                              |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+


Mistral
+++++++

+--------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter  | Required | Default                  | Description                                           |
+==========================+==========+==========================+=======================================================+
| **mistral_api_key**       | Yes      | N/A                      | This is the API key for the Mistral service.         |
+--------------------------+----------+--------------------------+-------------------------------------------------------+


TogetherAI
++++++++++

+---------------------------------------------------------------------------------------------------+
| Configuration Parameter  | Required | Default  | Description                                      |
+--------------------------+----------+----------+--------------------------------------------------+
| **togetherai_api_key**   | Yes      | N/A      | This is the API key for the TogetherAI service.  |
+--------------------------+----------+----------+--------------------------------------------------+

An example configuration for Azure OpenAI is:

.. code-block:: yaml

    endpoints:
      - name: completions
        endpoint_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-35-turbo
          config:
            openai_api_type: "azuread"
            openai_api_key: $AZURE_AAD_TOKEN
            openai_deployment_name: "{your_deployment_name}"
            openai_api_base: "https://{your_resource_name}-azureopenai.openai.azure.com/"
            openai_api_version: "2023-05-15"
        limit:
          renewal_period: minute
          calls: 10


.. note::

    Azure OpenAI has distinct features as compared with the direct OpenAI service. For an overview, please see `the comparison documentation <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/how-to/switching-endpoints>`_.

For specifying an API key, there are three options:

1. (Preferred) Use an environment variable to store the API key and reference it in the YAML configuration file. This is denoted by a ``$`` symbol before the name of the environment variable.
2. (Preferred) Define the API key in a file and reference the location of that key-bearing file within the YAML configuration file.
3. Directly include it in the YAML configuration file.

.. important::

    The use of environment variables or file-based keys is recommended for better security practices. If the API key is directly included in the configuration file, it should be ensured that the file is securely stored and appropriately access controlled.
    Please ensure that the configuration file is stored in a secure location as it contains sensitive API keys.

.. _deployments_query:

Querying the AI Gateway server
==============================

Once the MLflow AI Gateway has been configured and started, it is ready to receive traffic from users.

.. _standard_deployments_parameters:

Standard Query Parameters
-------------------------

The MLflow AI Gateway defines standard parameters for chat, completions, and embeddings that can be
used when querying any endpoint regardless of its provider. Each parameter has a standard range and
default value. When querying an endpoint with a particular provider, the MLflow AI Gateway automatically
scales parameter values according to the provider's value ranges for that parameter.

Completions
~~~~~~~~~~~

The standard parameters for completions endpoints with type ``llm/v1/completions`` are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Query Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **prompt**                    | string         | Yes      | N/A           | The prompt for which to generate completions.         |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **n**                         | integer        | No       | 1             | The number of completions to generate for the         |
|                               |                |          |               | specified prompt, between 1 and 5.                    |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **temperature**               | float          | No       | 0.0           | The sampling temperature to use, between 0 and 1.     |
|                               |                |          |               | Higher values will make the output more random, and   |
|                               |                |          |               | lower values will make the output more deterministic. |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **max_tokens**                | integer        | No       | None          | The maximum completion length, between 1 and infinity |
|                               |                |          |               | (unlimited).                                          |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **stop**                      | array[string]  | No       | None          | Sequences where the model should stop generating      |
|                               |                |          |               | tokens and return the completion.                     |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+

Chat
~~~~

The standard parameters for chat endpoints with type ``llm/v1/chat`` are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Query Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **messages**                  | array[message] | Yes      | N/A           | A list of messages in a conversation from which to    |
|                               |                |          |               | a new message (chat completion). For information      |
|                               |                |          |               | about the message structure, see                      |
|                               |                |          |               | :ref:`deployments_chat_message_structure`.            |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **n**                         | integer        | No       | 1             | The number of chat completions to generate for the    |
|                               |                |          |               | specified prompt, between 1 and 5.                    |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **temperature**               | float          | No       | 0.0           | The sampling temperature to use, between 0 and 1.     |
|                               |                |          |               | Higher values will make the output more random, and   |
|                               |                |          |               | lower values will make the output more deterministic. |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **max_tokens**                | integer        | No       | None          | The maximum completion length, between 1 and infinity |
|                               |                |          |               | (unlimited).                                          |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| **stop**                      | array[string]  | No       | None          | Sequences where the model should stop generating      |
|                               |                |          |               | tokens and return the chat completion.                |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+

.. _deployments_chat_message_structure:

Messages
^^^^^^^^

Each chat message is a string dictionary containing the following fields:

+-------------------------------+----------+--------------------------+-------------------------------------------------------+
| Field Name                    | Required | Default                  | Description                                           |
+===============================+==========+==========================+=======================================================+
| **role**                      | Yes      | N/A                      | The role of the conversation participant who sent the |
|                               |          |                          | message. Must be one of: ``"system"``, ``"user"``, or |
|                               |          |                          | ``"assistant"``.                                      |
+-------------------------------+----------+--------------------------+-------------------------------------------------------+
| **content**                   | Yes      | N/A                      | The message content.                                  |
+-------------------------------+----------+--------------------------+-------------------------------------------------------+

Embeddings
~~~~~~~~~~

The standard parameters for completions endpoints with type ``llm/v1/embeddings`` are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Query Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **input**                     | string         | Yes      | N/A           | A string or list of strings for which to generate     |
|                               | or             |          |               | embeddings.                                           |
|                               | array[string]  |          |               |                                                       |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+

Additional Query Parameters
---------------------------
In addition to the :ref:`standard_deployments_parameters`, you can pass any additional parameters supported by the endpoint's provider as part of your query. For example:

- ``logit_bias`` (supported by OpenAI, Cohere)
- ``top_k`` (supported by MosaicML, Anthropic, PaLM, Cohere)
- ``frequency_penalty`` (supported by OpenAI, Cohere, AI21 Labs)
- ``presence_penalty`` (supported by OpenAI, Cohere, AI21 Labs)
- ``stream`` (supported by OpenAI, Cohere)

Below is an example of submitting a query request to an MLflow AI Gateway endpoint using additional parameters:

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("http://my.deployments:8888")

    data = {
        "prompt": (
            "What would happen if an asteroid the size of "
            "a basketball encountered the Earth traveling at 0.5c? "
            "Please provide your answer in .rst format for the purposes of documentation."
        ),
        "temperature": 0.5,
        "max_tokens": 1000,
        "n": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
    }

    client.predict(endpoint="completions-gpt4", inputs=data)

The results of the query are:

.. code-block:: python

    {
        "id": "chatcmpl-8Pr33fsCAtD2L4oZHlyfOkiYHLapc",
        "object": "text_completion",
        "created": 1701172809,
        "model": "gpt-4-0613",
        "choices": [
            {
                "index": 0,
                "text": "If an asteroid the size of a basketball ...",
            }
        ],
        "usage": {
            "prompt_tokens": 43,
            "completion_tokens": 592,
            "total_tokens": 635,
        },
    }

Streaming
~~~~~~~~~

Some providers support streaming responses. Streaming responses are useful when you want to
receive responses as they are generated, rather than waiting for the entire response to be
generated before receiving it. Streaming responses are supported by the following providers:

+------------+---------------------+--------------+
|  Provider  | Endpoints                          |
+------------+---------------------+--------------+
|            | llm/v1/completions  | llm/v1/chat  |
+============+=====================+==============+
| OpenAI     | ✓                   | ✓            |
+------------+---------------------+--------------+
| Cohere     | ✓                   | ✓            |
+------------+---------------------+--------------+
| Anthropic  | ✘                   | ✓            |
+------------+---------------------+--------------+

To enable streaming responses, set the ``stream`` parameter to ``true`` in your request. For example:

.. code-block:: bash

     curl -X POST http://my.deployments:8888/endpoints/chat/invocations \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "hello"}], "stream": true}'


The results of the query follow the `OpenAI schema <https://platform.openai.com/docs/api-reference/chat/streaming>`_.


Chat
^^^^

.. code-block:: text

    data: {"choices": [{"delta": {"content": null, "role": "assistant"}, "finish_reason": null, "index": 0}], "created": 1701161926, "id": "chatcmpl-8PoDWSiVE8MHNsUZF2awkW5gNGYs3", "model": "gpt-35-turbo", "object": "chat.completion.chunk"}

    data: {"choices": [{"delta": {"content": "Hello", "role": null}, "finish_reason": null, "index": 0}], "created": 1701161926, "id": "chatcmpl-8PoDWSiVE8MHNsUZF2awkW5gNGYs3", "model": "gpt-35-turbo", "object": "chat.completion.chunk"}

    data: {"choices": [{"delta": {"content": " there", "role": null}, "finish_reason": null, "index": 0}], "created": 1701161926, "id": "chatcmpl-8PoDWSiVE8MHNsUZF2awkW5gNGYs3", "model": "gpt-35-turbo", "object": "chat.completion.chunk"}

    data: {"choices": [{"delta": {"content": null, "role": null}, "finish_reason": "stop", "index": 0}], "created": 1701161926, "id": "chatcmpl-8PoDWSiVE8MHNsUZF2awkW5gNGYs3", "model": "gpt-35-turbo", "object": "chat.completion.chunk"}


Completions
^^^^^^^^^^^

.. code-block:: text

    data: {"choices": [{"delta": {"role": null, "content": null}, "finish_reason": null, "index": 0}], "created": 1701161629, "id": "chatcmpl-8Po8jVXzljc245k1Ah4UsAcm2zxQ2", "model": "gpt-35-turbo", "object": "text_completion_chunk"}

    data: {"choices": [{"delta": {"role": null, "content": "If"}, "finish_reason": null, "index": 0}], "created": 1701161629, "id": "chatcmpl-8Po8jVXzljc245k1Ah4UsAcm2zxQ2", "model": "gpt-35-turbo", "object": "text_completion_chunk"}

    data: {"choices": [{"delta": {"role": null, "content": " an"}, "finish_reason": null, "index": 0}], "created": 1701161629, "id": "chatcmpl-8Po8jVXzljc245k1Ah4UsAcm2zxQ2", "model": "gpt-35-turbo", "object": "text_completion_chunk"}

    data: {"choices": [{"delta": {"role": null, "content": " asteroid"}, "finish_reason": null, "index": 0}], "created": 1701161629, "id": "chatcmpl-8Po8jVXzljc245k1Ah4UsAcm2zxQ2", "model": "gpt-35-turbo", "object": "text_completion_chunk"}

    data: {"choices": [{"delta": {"role": null, "content": null}, "finish_reason": "length", "index": 0}], "created": 1701161629, "id": "chatcmpl-8Po8jVXzljc245k1Ah4UsAcm2zxQ2", "model": "gpt-35-turbo", "object": "text_completion_chunk"}


FastAPI Documentation ("/docs")
-------------------------------

FastAPI, the framework used for building the MLflow AI Gateway, provides an automatic interactive API
documentation interface, which is accessible at the "/docs" endpoint (e.g., ``http://my.deployments:9000/docs``).
This interactive interface is very handy for exploring and testing the available API endpoints.

As a convenience, accessing the root URL (e.g., ``http://my.deployments:9000``) redirects to this "/docs" endpoint.

MLflow Python Client APIs
-------------------------
:class:`MlflowDeploymentClient <mlflow.deployments.MlflowDeploymentClient>` is the user-facing client API that is used to interact with the MLflow AI Gateway.
It abstracts the HTTP requests to the gateway server via a simple, easy-to-use Python API.

.. _deployments_client_api:

Client API
~~~~~~~~~~

To use the ``MlflowDeploymentClient`` API, see the below examples for the available API methods:

1. Create an ``MlflowDeploymentClient``

    .. code-block:: python

        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("http://my.deployments:8888")

2. List all endpoints:

    The :meth:`list_endpoints() <mlflow.deployments.MlflowDeploymentClient.list_endpoints>` method returns a list of all endpoints.

    .. code-block:: python

        endpoint = client.list_endpoints()
        for endpoint in endpoints:
            print(endpoint)

3. Query an endpoint:

    The :meth:`predict() <mlflow.deployments.MlflowDeploymentClient.predict>` method submits a query to a configured provider endpoint.
    The data structure you send in the query depends on the endpoint.

    .. code-block:: python

        response = client.predict(
            endpoint="chat",
            inputs={"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}]},
        )
        print(response)


LangChain Integration
~~~~~~~~~~~~~~~~~~~~~

`LangChain <https://github.com/langchain-ai/langchain>`_ supports `an integration for MLflow Deployments <https://python.langchain.com/docs/ecosystem/integrations/providers/mlflow>`_.
This integration enable users to use prompt engineering, retrieval augmented generation, and other techniques with LLMs in the gateway server.

.. code-block:: python
    :caption: Example

    import mlflow
    from langchain import LLMChain, PromptTemplate
    from langchain.llms import Mlflow

    llm = Mlflow(target_uri="http://127.0.0.1:5000", endpoint="completions")
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["adjective"],
            template="Tell me a {adjective} joke",
        ),
    )
    result = llm_chain.run(adjective="funny")
    print(result)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(llm_chain, "model")

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    print(model.predict([{"adjective": "funny"}]))


.. _deployments_mlflow_models:

MLflow Models
~~~~~~~~~~~~~
Interfacing with MLflow Models can be done in two ways. With the use of a custom PyFunc Model, a query can be issued directly to a gateway server endpoint and used in a broader context within a model.
Data may be augmented, manipulated, or used in a mixture of experts paradigm. The other means of utilizing the MLflow AI Gateway along with MLflow Models is to define a served MLflow model directly as
an endpoint within a gateway server.

Using the gateway server to Query a served MLflow Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a full walkthrough and example of using the MLflow serving integration to query a model directly through the MLflow AI Gateway, please see `the full example <https://github.com/mlflow/mlflow/tree/master/examples/deployments/mlflow_serving/README.md>`_.
Within the guide, you will see the entire end-to-end process of serving multiple models from different servers and configuring an MLflow AI Gateway instance to provide a single unified point to handle queries from.

Using an MLflow Model to Query the gateway server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also build and deploy MLflow Models that call the MLflow AI Gateway.
The example below demonstrates how to use a gateway server from within a custom ``pyfunc`` model.


.. note::
    The custom ``Model`` shown in the example below is utilizing environment variables for the gateway server's uri. These values can also be set manually within the
    definition or can be applied via :func:`mlflow.deployments.get_deployments_target` after the uri has been set. For the example below, the value for ``MLFLOW_DEPLOYMENTS_TARGET`` is
    ``http://127.0.0.1:5000/``. For an actual deployment use case, this value would be set to the configured and production deployment server.

.. code-block:: python

    import os
    import pandas as pd
    import mlflow


    def predict(data):
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client(os.environ["MLFLOW_DEPLOYMENTS_TARGET"])

        payload = data.to_dict(orient="records")
        return [
            client.predict(endpoint="completions", inputs=query)["choices"][0]["text"]
            for query in payload
        ]


    input_example = pd.DataFrame.from_dict(
        {"prompt": ["Where is the moon?", "What is a comet made of?"]}
    )
    signature = mlflow.models.infer_signature(
        input_example, ["Above our heads.", "It's mostly ice and rocks."]
    )

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=predict,
            registered_model_name="anthropic_completions",
            artifact_path="anthropic_completions",
            input_example=input_example,
            signature=signature,
        )

    df = pd.DataFrame.from_dict(
        {
            "prompt": ["Tell me about Jupiter", "Tell me about Saturn"],
            "temperature": 0.6,
            "max_records": 500,
        }
    )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    print(loaded_model.predict(df))

This custom MLflow model can be used in the same way as any other MLflow model. It can be used within a ``spark_udf``, used with :func:`mlflow.evaluate`, or `deploy <https://mlflow.org/docs/latest/models.html#built-in-deployment-tools>`_ like any other model.

.. _deployments_rest_api:

REST API
~~~~~~~~
The REST API allows you to send HTTP requests directly to the MLflow AI Gateway. This is useful if you're not using Python or if you prefer to interact with a gateway server using HTTP directly.

Here are some examples for how you might use curl to interact with the MLflow AI Gateway:

1. Get information about a particular endpoint: ``GET /api/2.0/endpoints/{name}``

   This route returns a serialized representation of the endpoint data structure.
   This provides information about the name and type, as well as the model details for the requested endpoint.

   .. code-block:: bash

       curl -X GET http://my.deployments:8888/api/2.0/endpoints/embeddings

2. List all endpoints: ``GET /api/2.0/endpoints/``

   This route returns a list of all endpoints.

   .. code-block:: bash

       curl -X GET http://my.deployments:8888/api/2.0/endpoints/

3. Query an endpoint: ``POST /endpoints/{name}/invocations``

   This route allows you to submit a query to a configured provider endpoint. The data structure you send in the query depends on the endpoint. Here are examples for the "completions", "chat", and "embeddings" endpoints:

   * ``Completions``

     .. code-block:: bash

         curl -X POST http://my.deployments:8888/endpoints/completions/invocations \
           -H "Content-Type: application/json" \
           -d '{"prompt": "Describe the probability distribution of the decay chain of U-235"}'


   * ``Chat``

     .. code-block:: bash

         curl -X POST http://my.deployments:8888/endpoints/chat/invocations \
           -H "Content-Type: application/json" \
           -d '{"messages": [{"role": "user", "content": "Can you write a limerick about orange flavored popsicles?"}]}'

   * ``Embeddings``

     .. code-block:: bash

         curl -X POST http://my.deployments:8888/endpoints/embeddings/invocations \
           -H "Content-Type: application/json" \
           -d '{"input": ["I would like to return my shipment of beanie babies, please", "Can I please speak to a human now?"]}'

**Note:** Remember to replace ``my.deployments:8888`` with the URL of your actual MLflow AI Gateway.

.. _deployments_plugin:

Plugin LLM Provider (Experimental)
==================================

.. attention::
    This feature is in active development and is marked as Experimental. It may change in a future release without warning.

The MLflow AI Gateway supports the use of custom language model providers through the use of plugins.
A plugin is a Python package that provides a custom implementation of a language model provider.
This allows users to integrate their own language model services with the MLflow AI Gateway.

To create a custom plugin, you need to implement a provider class that inherits from ``mlflow.gateway.providers.BaseProvider``,
and a config class that inherits from ``mlflow.gateway.base_models.ConfigModel``.

.. code-block:: python
    :caption: Example

    import os
    from typing import AsyncIterable

    from pydantic import validator
    from mlflow.gateway.base_models import ConfigModel
    from mlflow.gateway.config import RouteConfig
    from mlflow.gateway.providers import BaseProvider
    from mlflow.gateway.schemas import chat, completions, embeddings


    class MyLLMConfig(ConfigModel):
        # This model defines the configuration for the provider such as API keys
        my_llm_api_key: str

        @validator("my_llm_api_key", pre=True)
        def validate_my_llm_api_key(cls, value):
            return os.environ[value.lstrip("$")]


    class MyLLMProvider(BaseProvider):
        # Define the provider name. This will be displayed in log and error messages.
        NAME = "my_llm"
        # Define the config model for the provider.
        # This must be a subclass of ConfigModel.
        CONFIG_TYPE = MyLLMConfig

        def __init__(self, config: RouteConfig) -> None:
            super().__init__(config)
            if config.model.config is None or not isinstance(
                config.model.config, MyLLMConfig
            ):
                raise TypeError(f"Unexpected config type {config.model.config}")
            self.my_llm_config: MyLLMConfig = config.model.config

        # You can implement one or more of the following methods
        # depending on the capabilities of your provider.
        # Implementing `completions`, `chat` and `embeddings` will enable the respective endpoints.
        # Implementing `completions_stream` and `chat_stream` will enable the `stream=True`
        # option for the respective endpoints.
        # Unimplemented methods will return a 501 Not Implemented HTTP response upon invocation.
        async def completions_stream(
            self, payload: completions.RequestPayload
        ) -> AsyncIterable[completions.StreamResponsePayload]:
            ...

        async def completions(
            self, payload: completions.RequestPayload
        ) -> completions.ResponsePayload:
            ...

        async def chat_stream(
            self, payload: chat.RequestPayload
        ) -> AsyncIterable[chat.StreamResponsePayload]:
            ...

        async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
            ...

        async def embeddings(
            self, payload: embeddings.RequestPayload
        ) -> embeddings.ResponsePayload:
            ...

Then, you need to create a Python package that contains the plugin implementation.
You must specify an entry point under the ``mlflow.gateway.providers`` group, so that your plugin can be detected by MLflow.
The entry point should be in the format ``<name> = <module>:<class>``.

.. code-block:: toml
    :caption: pyproject.toml

    [project]
    name = "my_llm"
    version = "1.0"

    [project.entry-points."mlflow.gateway.providers"]
    my_llm = "my_llm.providers:MyLLMProvider"

    [tool.setuptools.packages.find]
    include = ["my_llm*"]
    namespaces = false

You can specify more than one entry point in the same package if you have multiple providers.
Note that entry point names must be globally unique. If two plugins specify the same entry point name,
MLflow will raise an error at startup time.

MLflow already provides a number of providers by default. Your plugin name cannot be the same as any one
of them. See :ref:`deployments_configuration_details` for a complete list of default providers.

Finally, you need to install the plugin package in the same environment as the MLflow AI Gateway.

.. important::

    Only install plugin packages from sources that you trust. Starting a server with a plugin provider will
    execute any arbitrary code that is defined within the plugin package.

Then, you can specify the plugin provider according to the entry point name
in the MLflow AI Gateway configuration file.

.. code-block:: yaml

    endpoints:
      - name: chat
        endpoint_type: llm/v1/chat
        model:
          provider: my_llm
          name: my-model-0.1.2
          config:
            my_llm_api_key: $MY_LLM_API_KEY

Example
-------

A working example can be found in the MLflow repository at
`examples/deployments/deployments_server/plugin <https://github.com/mlflow/mlflow/tree/master/examples/deployments/deployments_server/plugin>`__.

MLflow AI Gateway API Documentation
===========================================

`API documentation <./api.html>`_

OpenAI Compatibility
====================

MLflow AI Gateway is compatible with OpenAI API and supports the ``chat``, ``completions``, and ``embeddings`` APIs.
The OpenAI client can be used to query the server as shown in the example below:

1. Create a configuration file:

.. code-block:: yaml

    endpoints:
      - name: my-chat
        endpoint_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY

2. Start the server with the configuration file:

.. code-block:: shell

    mlflow gateway start --config-path /path/to/config.yaml --port 7000

3. Once the server is up and running, query the server using the OpenAI client:

.. code-block:: python

    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:7000/v1")
    completion = client.chat.completions.create(
        model="my-chat",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(completion.choices[0].message.content)



Unity Catalog Integration
=========================

.. toctree::
    :maxdepth: 1
    :hidden:

    uc_integration

See `Unity Catalog Integration <./uc_integration.html>`_ for how to integrate the MLflow AI Gateway with Unity Catalog.

.. _deployments_security:

gateway server Security Considerations
==========================================

Remember to ensure secure access to the system that the MLflow AI Gateway is running in to protect access to these keys.

An effective way to secure your gateway server is by placing it behind a reverse proxy. This will allow the reverse proxy to handle incoming requests and forward them to the MLflow AI Gateway. The reverse proxy effectively shields your application from direct exposure to Internet traffic.

A popular choice for a reverse proxy is `Nginx`. In addition to handling the traffic to your application, `Nginx` can also serve static files and load balance the traffic if you have multiple instances of your application running.

Furthermore, to ensure the integrity and confidentiality of data between the client and the server, it's highly recommended to enable HTTPS on your reverse proxy.

In addition to the reverse proxy, it's also recommended to add an authentication layer before the requests reach the MLflow AI Gateway. This could be HTTP Basic Authentication, OAuth, or any other method that suits your needs.

For example, here's a simple configuration for Nginx with Basic Authentication:

.. code-block:: nginx

    http {
        server {
            listen 80;

            location / {
                auth_basic "Restricted Content";
                auth_basic_user_file /etc/nginx/.htpasswd;

                proxy_pass http://localhost:5000;  # Replace with the MLflow AI Gateway port
            }
        }
    }

In this example, `/etc/nginx/.htpasswd` is a file that contains the username and password for authentication.

These measures, together with a proper network setup, can significantly improve the security of your system and ensure that only authorized users have access to submit requests to your LLM services.
