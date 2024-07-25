.. _gateway:

================================
MLflow AI Gateway (Experimental)
================================

.. warning::

    MLflow AI gateway is deprecated and has been replaced by `the deployments API <deployments>`
    for generative AI. See :ref:`gateway-migration` for migration.

The MLflow AI Gateway service is a powerful tool designed to streamline the usage and management of
various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization.
It offers a high-level interface that simplifies the interaction with these services by providing
a unified endpoint to handle specific LLM related requests.

A major advantage of using the MLflow AI Gateway service is its centralized management of API keys.
By storing these keys in one secure location, organizations can significantly enhance their
security posture by minimizing the exposure of sensitive API keys throughout the system. It also
helps to prevent exposing these keys within code or requiring end-users to manage keys safely.

The gateway is designed to be flexible and adaptable, capable of easily defining and managing routes by updating the
configuration file. This enables the easy incorporation
of new LLM providers or provider LLM types into the system without necessitating changes to
applications that interface with the gateway. This level of adaptability makes the MLflow AI Gateway
Service an invaluable tool in environments that require agility and quick response to changes.

This simplification and centralization of language model interactions, coupled with the added
layer of security for API key management, make the MLflow AI Gateway service an ideal choice for
organizations that use LLMs on a regular basis.

.. toctree::
    :maxdepth: 1
    :hidden:

    guides/index
    migration

Tutorials and Guides
====================

If you're interested in diving right in to a step by step guide that will get you up and running with the MLflow AI Gateway 
as fast as possible, the guides below will be your best first stop. 

.. raw:: html

    <a href="guides/index.html" class="download-btn">View the AI Gateway Getting Started Guide</a><br/>

.. _gateway-quickstart:

Quickstart
==========

The following guide will assist you in getting up and running, using a 3-route configuration to
OpenAI services for chat, completions, and embeddings.

Step 1: Install the MLflow AI Gateway service
---------------------------------------------
First, you need to install the MLflow AI Gateway service on your machine. You can do this using pip from PyPI or from the MLflow repository.

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    pip install 'mlflow[gateway]'

Step 2: Set the OpenAI API Key(s) for each provider
---------------------------------------------------
The Gateway service needs to communicate with the OpenAI API. To do this, it requires an API key.
You can create an API key from the OpenAI dashboard.

For this example, we're only connecting with OpenAI. If there are additional providers within the
configuration, these keys will need to be set as well.

Once you have the key, you can set it as an environment variable in your terminal:

.. code-block:: sh

    export OPENAI_API_KEY=your_api_key_here

This sets a temporary session-based environment variable. For production use cases, it is advisable
to store this key in the ``.bashrc`` or ``.zshrc`` files so that the key doesn't have to be re-entered upon
system restart.

Step 3: Create a Gateway Configuration File
-------------------------------------------
Next, you need to create a Gateway configuration file. This is a YAML file where you specify the
routes that the Gateway service should expose. Let's create a file with three routes using OpenAI as a provider: completions, chat, and embeddings.

For details about the configuration file's parameters (including parameters for other providers besides OpenAI), see the :ref:`gateway_configuration_details` section below.

.. code-block:: yaml

    routes:
      - name: completions
        route_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY

      - name: chat
        route_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY

      - name: embeddings
        route_type: llm/v1/embeddings
        model:
          provider: openai
          name: text-embedding-ada-002
          config:
            openai_api_key: $OPENAI_API_KEY

Save this file to a location on the system that is going to be running the MLflow AI Gateway server.

Step 4: Start the Gateway Service
---------------------------------
You're now ready to start the Gateway service!

Use the MLflow AI Gateway ``start`` command and specify the path to your configuration file:

.. code-block:: sh

    mlflow gateway start --config-path config.yaml --port {port} --host {host} --workers {worker count}

The configuration file can also be set using the ``MLFLOW_GATEWAY_CONFIG_PATH`` environment variable:

.. code-block:: bash

    export MLFLOW_GATEWAY_CONFIG_PATH=/path/to/config.yaml

If you do not specify the host, a localhost address will be used.

If you do not specify the port, port 5000 will be used.

The worker count for gunicorn defaults to 2 workers.

Step 5: Access the Interactive API Documentation
------------------------------------------------
The MLflow AI Gateway service provides an interactive API documentation endpoint that you can use to explore
and test the exposed routes. Navigate to ``http://{host}:{port}/`` (or ``http://{host}:{port}/docs``) in your browser to access it.

The docs endpoint allow for direct interaction with the routes and permits submitting actual requests to the
provider services by click on the "try it now" option within the endpoint definition entry.

Step 6: Send Requests Using the Fluent API
------------------------------------------
For information on formatting requirements and how to pass parameters, see :ref:`gateway_query`.

Here's an example of how to send a chat request using the :ref:`gateway_fluent_api` :

.. code-block:: python

    from mlflow.gateway import query, set_gateway_uri

    set_gateway_uri(gateway_uri="http://localhost:5000")

    response = query(
        "chat",
        {"messages": [{"role": "user", "content": "What is the best day of the week?"}]},
    )

    print(response)

**Note:** Remember to change the uri definition to the actual uri of your Gateway server.

The returned response will be in this data structure (the actual content and token values will likely be different):

.. code-block:: python

    {
        "candidates": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nIt's hard to say what the best day of the week is.",
                },
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 13,
            "output_tokens": 15,
            "total_tokens": 28,
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/chat",
        },
    }


Step 7: Send Requests Using the Client API
------------------------------------------
See the :ref:`gateway_client_api` section for further information.

Step 8: Send Requests to Routes via REST API
--------------------------------------------
You can now send requests to the exposed routes.
See the :ref:`REST examples <gateway_rest_api>` for guidance on request formatting.

Step 9: Compare Provider Models
-------------------------------
Here's an example of adding a new model from a provider to determine which model instance is better for a given use case.

Firstly, update the :ref:`MLflow AI Gateway config <gateway_configuration>` YAML file with the additional route definition to test:

.. code-block:: yaml

    routes:
      - name: completions
        route_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY
      - name: completions-gpt4
        route_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-4
          config:
            openai_api_key: $OPENAI_API_KEY

This updated configuration adds a new completions route ``completions-gpt4`` while still preserving the original ``completions``
route that was configured with the ``gpt-4o-mini``  model.

Once the configuration file is updated, simply save your changes. The Gateway will automatically create the new route with zero downtime.

At this point, you may use the :ref:`gateway_fluent_api` to query both routes with similar prompts to decide which model performs best for your use case.

If you no longer need a route, you can delete it from the configuration YAML and save your changes. The AI Gateway will automatically remove the route.

Step 10: Use AI Gateway routes for model development
----------------------------------------------------

Now that you have created several AI Gateway routes, you can create MLflow Models that query these
routes to build application-specific logic using techniques like prompt engineering. For more
information, see :ref:`AI Gateway and MLflow Models <gateway_mlflow_models>`.

.. _gateway-concepts:

Concepts
========

There are several concepts that are referred to within the MLflow AI Gateway APIs, the configuration definitions, examples, and documentation.
Becoming familiar with these terms will help in configuring new endpoints (routes) and ease the use of the interface APIs for the AI Gateway.

.. _providers:

Providers
---------
The MLflow AI Gateway is designed to support a variety of model providers.
A provider represents the source of the machine learning models, such as OpenAI, Anthropic, and so on.
Each provider has its specific characteristics and configurations that are encapsulated within the model part of a route in the MLflow AI Gateway.

Supported Provider Models
~~~~~~~~~~~~~~~~~~~~~~~~~
The table below presents a non-exhaustive list of models and a corresponding route type within the MLflow AI Gateway.
With the rapid development of LLMs, there is no guarantee that this list will be up to date at all times. However, the associations listed
below can be used as a helpful guide when configuring a given route for any newly released model types as they become available with a given provider.
``N/A`` means that the provider or the AI Gateway implementation currently doesn't support the route type.


+--------------------------+--------------------------+--------------------------+--------------------------+
|  Provider                | Routes                                                                         |
+--------------------------+--------------------------+--------------------------+--------------------------+
|                          | llm/v1/completions       | llm/v1/chat              | llm/v1/embeddings        |
+==========================+==========================+==========================+==========================+
| OpenAI                   | - gpt-3.5-turbo          | - gpt-3.5-turbo          | - text-embedding-ada-002 |
|                          | - gpt-4                  | - gpt-4                  |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| MosaicML                 | - mpt-7b-instruct        | - llama2-70b-chat†       | - instructor-large       |
|                          | - mpt-30b-instruct       |                          | - instructor-xl          |
|                          | - llama2-70b-chat†       |                          |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Anthropic                | - claude-1               | N/A                      | N/A                      |
|                          | - claude-1.3-100k        |                          |                          |
|                          | - claude-2               |                          |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Cohere                   | - command                | N/A                      | - embed-english-v2.0     |
|                          | - command-light-nightly  |                          | - embed-multilingual-v2.0|
+--------------------------+--------------------------+--------------------------+--------------------------+
| Azure OpenAI             | - text-davinci-003       | - gpt-35-turbo           | - text-embedding-ada-002 |
|                          | - gpt-35-turbo           | - gpt-4                  |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| PaLM                     | - text-bison-001         | - chat-bison-001         | - embedding-gecko-001    |
+--------------------------+--------------------------+--------------------------+--------------------------+
| MLflow                   | - MLflow served models*  | - MLflow served models*  | - MLflow served models** |
+--------------------------+--------------------------+--------------------------+--------------------------+
| HuggingFace TGI          | N/A                      | - HF TGI Models          | N/A                      |
+--------------------------+--------------------------+--------------------------+--------------------------+
| AI21 Labs                | - j2-ultra               | N/A                      | N/A                      |
|                          | - j2-mid                 |                          |                          |
|                          | - j2-light               |                          |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Amazon Bedrock           | - Amazon Titan           | N/A                      | N/A                      |
|                          | - Third-party providers  |                          |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Mistral                  | - mistral-tiny           | N/A                      |  - mistral-embed         |
|                          | - mistral-small          |                          |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| TogetherAI               | - google/gemma-2b        | - dbrx-instruct          |  - BAAI/bge-large-en-v1.5|
|                          | - microsoft/phi-2        |                          |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+

† Llama 2 is licensed under the `LLAMA 2 Community License <https://ai.meta.com/llama/license/>`_, Copyright © Meta Platforms, Inc. All Rights Reserved.

Within each model block in the configuration file, the provider field is used to specify the name
of the provider for that model. This is a string value that needs to correspond to a provider the MLflow AI Gateway supports.

.. note::
    `*` MLflow Model Serving will only work for chat or completions if the output return is in a route-compatible format. The
    response must conform to either an output of ``{"predictions": str}`` or ``{"predictions": {"candidates": str}}``. Any complex return type from a model that
    does not conform to these structures will raise an exception at query time.

    `**` Embeddings support is only available for models whose response signatures conform to the structured format of ``{"predictions": List[float]}``
    or ``{"predictions": List[List[float]]}``. Any other return type will raise an exception at query time. ``FeatureExtractionPipeline`` in ``transformers`` and
    models using the ``sentence_transformers`` flavor will return the correct data structures for the embeddings route.

Here's an example of a provider configuration within a route:

.. code-block:: yaml

    routes:
      - name: chat
        route_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4
          config:
            openai_api_key: $OPENAI_API_KEY

In the above configuration, ``openai`` is the `provider` for the model.

As of now, the MLflow AI Gateway supports the following providers:

* **mosaicml**: This is used for models offered by `MosaicML <https://docs.mosaicml.com/en/latest/>`_.
* **openai**: This is used for models offered by `OpenAI <https://platform.openai.com/>`_ and the `Azure <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/>`_ integrations for Azure OpenAI and Azure OpenAI with AAD.
* **anthropic**: This is used for models offered by `Anthropic <https://docs.anthropic.com/claude/docs>`_.
* **cohere**: This is used for models offered by `Cohere <https://docs.cohere.com/docs>`_.
* **palm**: This is used for models offered by `PaLM <https://developers.generativeai.google/api/rest/generativelanguage/models/>`_.
* **huggingface text generation inference**: This is used for models deployed using `Huggingface Text Generation Inference <https://huggingface.co/docs/text-generation-inference/index>`_.
* **ai21labs**: This is used for models offered by `AI21 Labs <https://studio.ai21.com/foundation-models>`_.
* **bedrock**: This is used for models offered by `Amazon Bedrock <https://aws.amazon.com/bedrock/>`_.
* **mistral**: This is used for models offered by `Mistral <https://docs.mistral.ai/>`_.
* **togetherai**: This is used for models offered by `TogetherAI <https://docs.together.ai/docs/>`_.

More providers are being added continually. Check the latest version of the MLflow AI Gateway Docs for the
most up-to-date list of supported providers.

Remember, the provider you specify must be one that the MLflow AI Gateway supports. If the provider
is not supported, the Gateway will return an error when trying to route requests to that provider.

.. _routes:

Routes
------

`Routes` are central to how the MLflow AI Gateway functions. Each route acts as a proxy endpoint for the
user, forwarding requests to the underlying :ref:`gateway_models` and :ref:`providers` specified in the configuration file.

A route in the MLflow AI Gateway consists of the following fields:

* **name**: This is the unique identifier for the route. This will be part of the URL when making API calls via the MLflow AI Gateway.

* **type**: The type of the route corresponds to the type of language model interaction you desire. For instance, ``llm/v1/completions`` for text completion operations, ``llm/v1/embeddings`` for text embeddings, and ``llm/v1/chat`` for chat operations.

* **model**: Defines the model to which this route will forward requests. The model contains the following details:

    * **provider**: Specifies the name of the :ref:`provider <providers>` for this model. For example, ``openai`` for OpenAI's ``GPT-4o`` models.
    * **name**: The name of the model to use. For example, ``gpt-4o-mini`` for OpenAI's ``GPT-4o-Mini`` model.
    * **config**: Contains any additional configuration details required for the model. This includes specifying the API base URL and the API key.

Here's an example of a route configuration:

.. code-block:: yaml

    routes:
      - name: completions
        type: chat/completions
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY

In the example above, a request sent to the completions route would be forwarded to the
``gpt-4o-mini`` model provided by ``openai``.

The routes in the configuration file can be updated at any time, and the MLflow AI Gateway will
automatically update its available routes without requiring a restart. This feature provides you
with the flexibility to add, remove, or modify routes as your needs change. It enables 'hot-swapping'
of routes, providing a seamless experience for any applications or services that interact with the MLflow AI Gateway.

When defining routes in the configuration file, ensure that each name is unique to prevent conflicts.
Duplicate route names will raise an ``MlflowException``.

.. _gateway_models:

Models
------

The ``model`` section within a ``route`` specifies which model to use for generating responses.
This configuration block needs to contain a ``name`` field which is used to specify the exact model instance to be used.
Additionally, a :ref:`provider <providers>` needs to be specified, one that you have an authenticated access api key for.

Different endpoint types are often associated with specific models.
For instance, the ``llm/v1/chat`` and ``llm/v1/completions`` endpoints are generally associated with
conversational models, while ``llm/v1/embeddings`` endpoints would typically be associated with
embedding or transformer models. The model you choose should be appropriate for the type of endpoint specified.

Here's an example of a model name configuration within a route:

.. code-block:: yaml

    routes:
      - name: embeddings
        route_type: llm/v1/embeddings
        model:
          provider: openai
          name: text-embedding-ada-002
          config:
            openai_api_key: $OPENAI_API_KEY


In the above configuration, ``text-embedding-ada-002`` is the model used for the embeddings endpoint.

When specifying a model, it is critical that the provider supports the model you are requesting.
For instance, ``openai`` as a provider supports models like ``text-embedding-ada-002``, but other providers
may not. If the model is not supported by the provider, the MLflow AI Gateway will return an HTTP 4xx error
when trying to route requests to that model.

.. important::

    Always check the latest documentation of the specified provider to ensure that the model you want
    to use is supported for the type of endpoint you're configuring.

Remember, the model you choose directly affects the results of the responses you'll get from the
API calls. Therefore, choose a model that fits your use-case requirements. For instance,
for generating conversational responses, you would typically choose a chat model.
Conversely, for generating embeddings of text, you would choose an embedding model.

.. _gateway_configuration:

Configuring the AI Gateway
==========================

The MLflow AI Gateway service relies on a user-provided configuration file, written in YAML,
that defines the routes and providers available to the service. The configuration file dictates
how the gateway interacts with various language model providers and determines the end-points that
users can access.

AI Gateway Configuration
------------------------

The configuration file includes a series of sections, each representing a unique route.
Each route section has a name, a type, and a model specification, which includes the model
provider, name, and configuration details. The configuration section typically contains the base
URL for the API and an environment variable for the API key.

Here is an example of a single-route configuration:

.. code-block:: yaml

    routes:
      - name: chat
        route_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY


In this example, we define a route named ``chat`` that corresponds to the ``llm/v1/chat`` type, which
will use the ``gpt-4o-mini`` model from OpenAI to return query responses from the OpenAI service.

The Gateway configuration is very easy to update.
Simply edit the configuration file and save your changes, and the MLflow AI Gateway service will automatically
update the routes with zero disruption or down time. This allows you to try out new providers or model types while keeping your applications steady and reliable.

In order to define an API key for a given provider, there are three primary options:

1. Directly include it in the YAML configuration file.
2. Use an environment variable to store the API key and reference it in the YAML configuration file.
3. Define your API key in a file and reference the location of that key-bearing file within the YAML configuration file.

If you choose to include the API key directly, replace ``$OPENAI_API_KEY`` in the YAML file with your
actual API key.

.. warning::

    The MLflow AI Gateway service provides direct access to billed external LLM services. It is strongly recommended to restrict access to this server. See the section on :ref:`security <gateway_security>` for guidance.

If you prefer to use an environment variable (recommended), you can define it in your shell
environment. For example:

.. code-block:: bash

     export OPENAI_API_KEY="your_openai_api_key"

**Note:** Replace "your_openai_api_key" with your actual OpenAI API key.

.. _gateway_configuration_details:

AI Gateway Configuration Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MLflow AI Gateway service relies on a user-provided configuration file. It defines how the gateway interacts with various language model providers and dictates the routes that users can access.

The configuration file is written in YAML and includes a series of sections, each representing a unique route. Each route section has a name, a type, and a model specification, which includes the provider, model name, and provider-specific configuration details.

Here are the details of each configuration parameter:

General Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **routes**: This is a list of route configurations. Each route represents a unique endpoint that maps to a particular language model service.

Each route has the following configuration parameters:

- **name**: This is the name of the route. It needs to be a unique name without spaces or any non-alphanumeric characters other than hyphen and underscore.

- **route_type**: This specifies the type of service offered by this route. This determines the interface for inputs to a route and the returned outputs. Current supported route types are:

  - "llm/v1/completions"
  - "llm/v1/chat"
  - "llm/v1/embeddings"

- **model**: This defines the provider-specific details of the language model. It contains the following fields:

  - **provider**: This indicates the provider of the AI model. It accepts the following values:

    - "openai"
    - "mosaicml"
    - "anthropic"
    - "cohere"
    - "palm"
    - "azure" / "azuread"
    - "mlflow-model-serving"
    - "huggingface-text-generation-inference"
    - "ai21labs"
    - "bedrock"
    - "mistral"
    - "togetherai"

  - **name**: This is an optional field to specify the name of the model.
  - **config**: This contains provider-specific configuration details.

Provider-Specific Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenAI
++++++

+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| Configuration Parameter | Required | Default                       | Description                                                 |
+=========================+==========+===============================+=============================================================+
| **openai_api_key**      | Yes      |                               | This is the API key for the OpenAI service.                 |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_type**     | No       |                               | This is an optional field to specify the type of OpenAI API |
|                         |          |                               | to use.                                                     |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_base**     | No       | `https://api.openai.com/v1`   | This is the base URL for the OpenAI API.                    |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_version**  | No       |                               | This is an optional field to specify the OpenAI API         |
|                         |          |                               | version.                                                    |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_organization** | No       |                               | This is an optional field to specify the organization in    |
|                         |          |                               | OpenAI.                                                     |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+


MosaicML
+++++++++

+-------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter | Required | Default                  | Description                                           |
+=========================+==========+==========================+=======================================================+
| **mosaicml_api_key**    | Yes      | N/A                      | This is the API key for the MosaicML service.         |
+-------------------------+----------+--------------------------+-------------------------------------------------------+


Cohere
++++++

+--------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter  | Required | Default                  | Description                                           |
+==========================+==========+==========================+=======================================================+
| **cohere_api_key**       | Yes      | N/A                      | This is the API key for the Cohere service.           |
+--------------------------+----------+--------------------------+-------------------------------------------------------+

HuggingFace Text Generation Inference
+++++++++++++++++++++++++++++++++++++

+-------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter | Required | Default                  | Description                                           |
+=========================+==========+==========================+=======================================================+
| **hf_server_url**       | Yes      | N/A                      | This is the url of the Huggingface TGI Server.        |
+-------------------------+----------+--------------------------+-------------------------------------------------------+


PaLM
++++

+--------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter  | Required | Default                  | Description                                           |
+==========================+==========+==========================+=======================================================+
| **palm_api_key**         | Yes      | N/A                      | This is the API key for the PaLM service.             |
+--------------------------+----------+--------------------------+-------------------------------------------------------+


AI21 Labs
+++++++++

+--------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter  | Required | Default                  | Description                                           |
+==========================+==========+==========================+=======================================================+
| **ai21labs_api_key**     | Yes      | N/A                      | This is the API key for the AI21 Labs service.        |
+--------------------------+----------+--------------------------+-------------------------------------------------------+


Anthropic
+++++++++

+-------------------------+----------+--------------------------+-------------------------------------------------------+
| Configuration Parameter | Required | Default                  | Description                                           |
+=========================+==========+==========================+=======================================================+
| **anthropic_api_key**   | Yes      | N/A                      | This is the API key for the Anthropic service.        |
+-------------------------+----------+--------------------------+-------------------------------------------------------+

Amazon Bedrock
++++++++++++++

Top-level model configuration for Amazon Bedrock routes must be one of the following two supported authentication modes: `key-based` or `role-based`.

+--------------------------+----------+------------------------------+-------------------------------------------------------+
| Configuration Parameter  | Required | Default                      | Description                                           |
+==========================+==========+==============================+=======================================================+
| **aws_config**           | No       |                              | An object with either the key-based or role-based     |
|                          |          |                              | schema below.                                         |
+--------------------------+----------+------------------------------+-------------------------------------------------------+


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


To use key-based authentication, define an Amazon Bedrock route with the required fields below.
.. note::

  If using a configured route purely for development or testing, utilizing an IAM User role or a temporary short-lived standard IAM role are recommended; while for production deployments, a standard long-expiry IAM role is recommended to ensure that the route is capable of handling authentication for a long period. If the authentication expires and a new set of keys need to be supplied, the route must be recreated in order to persist the new keys.

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

Alternatively, for role-based authentication, an Amazon Bedrock route can be defined and initialized with an a IAM Role  ARN that is authorized to access Bedrock.  The MLflow AI Gateway will attempt to assume this role with using the standard credential provider chain and will renew the role credentials if they have expired.

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
useful for understanding a particular version or route definition that was used that can be referenced back to a deployed model. You may choose any name that you wish, provided that
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


An example configuration for Azure OpenAI is:

.. code-block:: yaml

    routes:
      - name: completions
        route_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-35-turbo
          config:
            openai_api_type: "azuread"
            openai_api_key: $AZURE_AAD_TOKEN
            openai_deployment_name: "{your_deployment_name}"
            openai_api_base: "https://{your_resource_name}-azureopenai.openai.azure.com/"
            openai_api_version: "2023-05-15"


.. note::

    Azure OpenAI has distinct features as compared with the direct OpenAI service. For an overview, please see `the comparison documentation <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/how-to/switching-endpoints>`_.

For specifying an API key, there are three options:

1. (Preferred) Use an environment variable to store the API key and reference it in the YAML configuration file. This is denoted by a ``$`` symbol before the name of the environment variable.
2. (Preferred) Define the API key in a file and reference the location of that key-bearing file within the YAML configuration file.
3. Directly include it in the YAML configuration file.

.. important::

    The use of environment variables or file-based keys is recommended for better security practices. If the API key is directly included in the configuration file, it should be ensured that the file is securely stored and appropriately access controlled.
    Please ensure that the configuration file is stored in a secure location as it contains sensitive API keys.

.. _gateway_query:

Querying the AI Gateway
=======================

Once the MLflow AI Gateway server has been configured and started, it is ready to receive traffic from users.

.. _standard_query_parameters:

Standard Query Parameters
-------------------------

The MLflow AI Gateway defines standard parameters for chat, completions, and embeddings that can be
used when querying any route regardless of its provider. Each parameter has a standard range and
default value. When querying a route with a particular provider, the MLflow AI Gateway automatically
scales parameter values according to the provider's value ranges for that parameter.

Completions
~~~~~~~~~~~

The standard parameters for completions routes with type ``llm/v1/completions`` are:

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

The standard parameters for chat routes with type ``llm/v1/chat`` are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Query Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **messages**                  | array[message] | Yes      | N/A           | A list of messages in a conversation from which to    |
|                               |                |          |               | a new message (chat completion). For information      |
|                               |                |          |               | about the message structure, see                      |
|                               |                |          |               | :ref:`chat_message_structure`.                        |
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

.. _chat_message_structure:

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

The standard parameters for completions routes with type ``llm/v1/embeddings`` are:

+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+
| Query Parameter               | Type           | Required | Default       | Description                                           |
+===============================+================+==========+===============+=======================================================+
| **input**                     | string         | Yes      | N/A           | A string or list of strings for which to generate     |
|                               | or             |          |               | embeddings.                                           |
|                               | array[string]  |          |               |                                                       |
+-------------------------------+----------------+----------+---------------+-------------------------------------------------------+

Additional Query Parameters
---------------------------
In addition to the :ref:`standard_query_parameters`, you can pass any additional parameters supported by the route's provider as part of your query. For example:

- ``logit_bias`` (supported by OpenAI, Cohere)
- ``top_k`` (supported by MosaicML, Anthropic, PaLM, Cohere)
- ``frequency_penalty`` (supported by OpenAI, Cohere, AI21 Labs)
- ``presence_penalty`` (supported by OpenAI, Cohere, AI21 Labs)

The following parameters are not allowed:

- ``stream`` is not supported. Setting this parameter on any provider will not work currently.

Below is an example of submitting a query request to an MLflow AI Gateway route using additional parameters:

.. code-block:: python

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

    query(route="completions-gpt4", data=data)

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

FastAPI Documentation ("/docs")
-------------------------------

FastAPI, the framework used for building the MLflow AI Gateway, provides an automatic interactive API
documentation interface, which is accessible at the "/docs" endpoint (e.g., ``http://my.gateway:9000/docs``).
This interactive interface is very handy for exploring and testing the available API endpoints.

As a convenience, accessing the root URL (e.g., ``http://my.gateway:9000``) redirects to this "/docs" endpoint.

MLflow Python Client APIs
-------------------------
:class:`MlflowGatewayClient <mlflow.gateway.client.MlflowGatewayClient>` is the user-facing client API that is used to interact with the MLflow AI Gateway.
It abstracts the HTTP requests to the Gateway via a simple, easy-to-use Python API.

The fluent API is a higher-level interface that supports setting the Gateway URI once and using simple functions to interact with the AI Gateway Server.

.. _gateway_fluent_api:

Fluent API
~~~~~~~~~~
For the ``fluent`` API, here are some examples:

1. Set the Gateway URI:

   Before using the Fluent API, the gateway URI must be set via :func:`set_gateway_uri() <mlflow.gateway.set_gateway_uri>`.

   Alternatively to directly calling the ``set_gateway_uri`` function, the environment variable ``MLFLOW_GATEWAY_URI`` can be set
   directly, achieving the same session-level persistence for all ``fluent`` API usages.

   .. code-block:: python

       from mlflow.gateway import set_gateway_uri

       set_gateway_uri(gateway_uri="http://my.gateway:7000")

2. Query a route:

   The :func:`query() <mlflow.gateway.query>` function queries the specified route and returns the response from the provider
   in a standardized format. The data structure you send in the query depends on the route.

   .. code-block:: python

       from mlflow.gateway import query

       response = query(
           "embeddings", {"input": ["It was the best of times", "It was the worst of times"]}
       )
       print(response)

.. _gateway_client_api:

Client API
~~~~~~~~~~

To use the ``MlflowGatewayClient`` API, see the below examples for the available API methods:

1. Create an ``MlflowGatewayClient``

   .. code-block:: python

       from mlflow.gateway import MlflowGatewayClient

       gateway_client = MlflowGatewayClient("http://my.gateway:8888")

2. List all routes:

   The :meth:`search_routes() <mlflow.gateway.client.MlflowGatewayClient.search_routes>` method returns a list of all routes.

   .. code-block:: python

       routes = gateway_client.search_routes()
       for route in routes:
           print(route)

3. Query a route:

   The :meth:`query() <mlflow.gateway.client.MlflowGatewayClient.query>` method submits a query to a configured provider route.
   The data structure you send in the query depends on the route.

   .. code-block:: python

       response = gateway_client.query(
           "chat", {"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}]}
       )
       print(response)


LangChain Integration
~~~~~~~~~~~~~~~~~~~~~

`LangChain <https://github.com/hwchase17/langchain>`_ supports `an integration for MLflow AI Gateway <https://python.langchain.com/docs/ecosystem/integrations/mlflow_ai_gateway>`_.
This integration enable users to use prompt engineering, retrieval augmented generation, and other techniques with LLMs in the gateway.

.. code-block:: python
    :caption: Example

    import mlflow
    from langchain import LLMChain, PromptTemplate
    from langchain.llms import MlflowAIGateway

    gateway = MlflowAIGateway(
        gateway_uri="http://127.0.0.1:5000",
        route="completions",
        params={
            "temperature": 0.0,
            "top_p": 0.1,
        },
    )

    llm_chain = LLMChain(
        llm=gateway,
        prompt=PromptTemplate(
            input_variables=["adjective"],
            template="Tell me a {adjective} joke",
        ),
    )
    result = llm_chain.run(adjective="funny")
    print(result)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, "model")

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    print(model.predict([{"adjective": "funny"}]))


.. _gateway_mlflow_models:

MLflow Models
~~~~~~~~~~~~~
Interfacing with MLflow Models can be done in two ways. With the use of a custom PyFunc Model, a query can be issued directly to an AI Gateway endpoint and used in a broader context within a model.
Data may be augmented, manipulated, or used in a mixture of experts paradigm. The other means of utilizing the AI Gateway along with MLflow Models is to define a served MLflow model directly as a
route within the AI Gateway.

Using the AI Gateway to Query a served MLflow Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a full walkthrough and example of using the MLflow serving integration to query a model directly through the MLflow AI Gateway, please see `the full example <https://github.com/mlflow/mlflow/tree/master/examples/gateway/mlflow_serving/README.md>`_.
Within the guide, you will see the entire end-to-end process of serving multiple models from different servers and configuring an MLflow AI Gateway server instance to provide a single unified point to handle queries from.

Using an MLflow Model to Query the AI Gateway
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also build and deploy MLflow Models that call the MLflow AI Gateway.
The example below demonstrates how to use an AI Gateway server from within a custom ``pyfunc`` model.


.. note::
    The custom ``Model`` shown in the example below is utilizing environment variables for the AI Gateway server's uri. These values can also be set manually within the
    definition or can be applied via :func:`mlflow.gateway.get_gateway_uri` after the uri has been set. For the example below, the value for ``MLFLOW_GATEWAY_URI`` is
    ``http://127.0.0.1:5000/``. For an actual deployment use case, this value would be set to the configured and production deployment server.

.. code-block:: python

    import os
    import pandas as pd
    import mlflow


    def predict(data):
        from mlflow.gateway import MlflowGatewayClient

        client = MlflowGatewayClient(os.environ["MLFLOW_GATEWAY_URI"])

        payload = data.to_dict(orient="records")
        return [
            client.query(route="completions-claude", data=query)["choices"][0]["text"]
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

.. _gateway_rest_api:

REST API
~~~~~~~~
The REST API allows you to send HTTP requests directly to the MLflow AI Gateway server. This is useful if you're not using Python or if you prefer to interact with the Gateway using HTTP directly.

Here are some examples for how you might use curl to interact with the Gateway:

1. Get information about a particular route: ``GET /api/2.0/gateway/routes/{name}``
   This endpoint returns a serialized representation of the Route data structure.
   This provides information about the name and type, as well as the model details for the requested route endpoint.

   .. code-block:: bash

       curl -X GET http://my.gateway:8888/api/2.0/gateway/routes/embeddings

2. List all routes: ``GET /api/2.0/gateway/routes/``

   This endpoint returns a list of all routes.

   .. code-block:: bash

       curl -X GET http://my.gateway:8888/api/2.0/gateway/routes/

3. Query a route: ``POST /gateway/{route}/invocations``

   This endpoint allows you to submit a query to a configured provider route. The data structure you send in the query depends on the route. Here are examples for the "completions", "chat", and "embeddings" routes:


   * ``Completions``

     .. code-block:: bash

         curl -X POST http://my.gateway:8888/gateway/completions/invocations \
           -H "Content-Type: application/json" \
           -d '{"prompt": "Describe the probability distribution of the decay chain of U-235"}'


   * ``Chat``

     .. code-block:: bash

         curl -X POST http://my.gateway:8888/gateway/chat/invocations \
           -H "Content-Type: application/json" \
           -d '{"messages": [{"role": "user", "content": "Can you write a limerick about orange flavored popsicles?"}]}'

   * ``Embeddings``

     .. code-block:: bash

         curl -X POST http://my.gateway:8888/gateway/embeddings/invocations \
           -H "Content-Type: application/json" \
           -d '{"input": ["I would like to return my shipment of beanie babies, please", "Can I please speak to a human now?"]}'

**Note:** Remember to replace ``http://my.gateway:8888`` with the URL of your actual MLflow AI Gateway Server.

MLflow AI Gateway API Documentation
===================================

`API documentation <../deployments/api.html>`_

.. _gateway_security:

AI Gateway Security Considerations
==================================

Remember to ensure secure access to the system that the MLflow AI Gateway service is running in to protect access to these keys.

An effective way to secure your MLflow AI Gateway service is by placing it behind a reverse proxy. This will allow the reverse proxy to handle incoming requests and forward them to the MLflow AI Gateway. The reverse proxy effectively shields your application from direct exposure to Internet traffic.

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

                proxy_pass http://localhost:5000;  # Replace with the MLflow AI Gateway service port
            }
        }
    }

In this example, `/etc/nginx/.htpasswd` is a file that contains the username and password for authentication.

These measures, together with a proper network setup, can significantly improve the security of your system and ensure that only authorized users have access to submit requests to your LLM services.

LangChain Integration
=====================

`LangChain <https://github.com/hwchase17/langchain>`_ supports an integration for MLflow AI Gateway. See https://python.langchain.com/docs/ecosystem/integrations/mlflow_ai_gateway for more information.
