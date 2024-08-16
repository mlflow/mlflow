.. _deployments:

========================================
MLflow Deployments Server (Experimental)
========================================

.. important::
   The feature previously known as **MLflow AI Gateway** in experimental status has been moved to
   utilize the **MLflow deployments API**. This major update involves changes to API endpoints and
   standardization for Large Language Models, both custom and SaaS-based. Users currently utilizing
   MLflow AI Gateway should refer to the new documentation for migration guidelines and familiarize
   themselves with the updated API structure. See :ref:`gateway-migration` for migration.

.. warning::

    MLflow Deployments Server does not support Windows.

The MLflow Deployments Server is a powerful tool designed to streamline the usage and management of
various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization.
It offers a high-level interface that simplifies the interaction with these services by providing
a unified endpoint to handle specific LLM related requests.

A major advantage of using the MLflow Deployments Server is its centralized management of API keys.
By storing these keys in one secure location, organizations can significantly enhance their
security posture by minimizing the exposure of sensitive API keys throughout the system. It also
helps to prevent exposing these keys within code or requiring end-users to manage keys safely.

The deployments server is designed to be flexible and adaptable, capable of easily defining and managing endpoints by updating the
configuration file. This enables the easy incorporation
of new LLM providers or provider LLM types into the system without necessitating changes to
applications that interface with the deployments server. This level of adaptability makes the MLflow Deployments Server
Service an invaluable tool in environments that require agility and quick response to changes.

This simplification and centralization of language model interactions, coupled with the added
layer of security for API key management, make the MLflow Deployments Server an ideal choice for
organizations that use LLMs on a regular basis.

.. toctree::
    :maxdepth: 1
    :hidden:

    guides/index

Tutorials and Guides
====================

If you're interested in diving right in to a step by step guide that will get you up and running with the MLflow Deployments Server
as fast as possible, the guides below will be your best first stop.

.. raw:: html

    <a href="guides/index.html" class="download-btn">View the Deployments Server Getting Started Guide</a><br/>

.. _deployments-quickstart:

Quickstart
==========

The following guide will assist you in getting up and running, using a 3-endpoint configuration to
OpenAI services for chat, completions, and embeddings.

Step 1: Install the MLflow Deployments Server
---------------------------------------------
First, you need to install the MLflow Deployments Server on your machine. You can do this using pip from PyPI or from the MLflow repository.

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    pip install 'mlflow[genai]'

Step 2: Set the OpenAI API Key(s) for each provider
---------------------------------------------------
The deployments server needs to communicate with the OpenAI API. To do this, it requires an API key.
You can create an API key from the OpenAI dashboard.

For this example, we're only connecting with OpenAI. If there are additional providers within the
configuration, these keys will need to be set as well.

Once you have the key, you can set it as an environment variable in your terminal:

.. code-block:: sh

    export OPENAI_API_KEY=your_api_key_here

This sets a temporary session-based environment variable. For production use cases, it is advisable
to store this key in the ``.bashrc`` or ``.zshrc`` files so that the key doesn't have to be re-entered upon
system restart.

Step 3: Create a Deployments Server Configuration File
------------------------------------------------------
Next, you need to create a deployments server configuration file. This is a YAML file where you specify the
endpoints that the MLflow Deployments Server should expose. Let's create a file with three endpoints using OpenAI as a provider: completions, chat, and embeddings.

For details about the configuration file's parameters (including parameters for other providers besides OpenAI), see the :ref:`deployments_configuration_details` section below.

.. code-block:: yaml

    endpoints:
      - name: completions
        endpoint_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY
        limit:
          renewal_period: minute
          calls: 10

      - name: chat
        endpoint_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY

      - name: embeddings
        endpoint_type: llm/v1/embeddings
        model:
          provider: openai
          name: text-embedding-ada-002
          config:
            openai_api_key: $OPENAI_API_KEY

Save this file to a location on the system that is going to be running the MLflow Deployments Server.

Step 4: Start the Deployments Server
------------------------------------
You're now ready to start the deployments server!

Use the MLflow Deployments Server ``start-server`` command and specify the path to your configuration file:

.. code-block:: sh

    mlflow deployments start-server --config-path config.yaml --port {port} --host {host} --workers {worker count}

The configuration file can also be set using the ``MLFLOW_DEPLOYMENTS_CONFIG`` environment variable:

.. code-block:: bash

    export MLFLOW_DEPLOYMENTS_CONFIG=/path/to/config.yaml

If you do not specify the host, a localhost address will be used.

If you do not specify the port, port 5000 will be used.

The worker count for gunicorn defaults to 2 workers.

Step 5: Access the Interactive API Documentation
------------------------------------------------
The MLflow Deployments Server provides an interactive API documentation endpoint that you can use to explore
and test the exposed endpoints. Navigate to ``http://{host}:{port}/`` (or ``http://{host}:{port}/docs``) in your browser to access it.

The docs endpoint allow for direct interaction with the endpoints and permits submitting actual requests to the
provider services by click on the "try it now" option within the endpoint definition entry.

Step 6: Send Requests Using the Client API
------------------------------------------
See the :ref:`deployments_client_api` section for further information.

Step 7: Send Requests to Endpoints via REST API
-----------------------------------------------
You can now send requests to the exposed endpoints.
See the :ref:`REST examples <deployments_rest_api>` for guidance on request formatting.

Step 8: Compare Provider Models
-------------------------------
Here's an example of adding a new model from a provider to determine which model instance is better for a given use case.

Firstly, update the :ref:`MLflow Deployments Server config <deployments_configuration>` YAML file with the additional endpoint definition to test:

.. code-block:: yaml

    endpoints:
      - name: completions
        endpoint_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY
      - name: completions-gpt4
        endpoint_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-4
          config:
            openai_api_key: $OPENAI_API_KEY

This updated configuration adds a new completions endpoint ``completions-gpt4`` while still preserving the original ``completions``
endpoint that was configured with the ``gpt-4o-mini``  model.

Once the configuration file is updated, simply save your changes. The deployments server will automatically create the new endpoint with zero downtime.

If you no longer need an endpoint, you can delete it from the configuration YAML and save your changes. The deployments server will automatically remove the endpoint.

Step 9: Use Deployments Server endpoints for model development
--------------------------------------------------------------

Now that you have created several deployments server endpoints, you can create MLflow Models that query these
endpoints to build application-specific logic using techniques like prompt engineering. For more
information, see :ref:`Deployments Server and MLflow Models <deployments_mlflow_models>`.

.. _deployments-concepts:

Concepts
========

There are several concepts that are referred to within the MLflow Deployments Server APIs, the configuration definitions, examples, and documentation.
Becoming familiar with these terms will help to simplify both configuring new endpoints and using the MLflow Deployments Server APIs.

.. _deployments-providers:

Providers
---------
The MLflow Deployments Server is designed to support a variety of model providers.
A provider represents the source of the machine learning models, such as OpenAI, Anthropic, and so on.
Each provider has its specific characteristics and configurations that are encapsulated within the model part of an endpoint in the MLflow Deployments Server.

Supported Provider Models
~~~~~~~~~~~~~~~~~~~~~~~~~
The table below presents a non-exhaustive list of models and a corresponding endpoint type within the MLflow Deployments Server.
With the rapid development of LLMs, there is no guarantee that this list will be up to date at all times. However, the associations listed
below can be used as a helpful guide when configuring a given endpoint for any newly released model types as they become available with a given provider.
``N/A`` means that either the provider or the MLflow Deployments Server implementation currently doesn't support the endpoint type.


+--------------------------+--------------------------+--------------------------+--------------------------+
|  Provider                | Endpoints                                                                      |
+--------------------------+--------------------------+--------------------------+--------------------------+
|                          | llm/v1/completions       | llm/v1/chat              | llm/v1/embeddings        |
+==========================+==========================+==========================+==========================+
| OpenAI §                 | - gpt-3.5-turbo-instruct | - gpt-3.5-turbo          | - text-embedding-ada-002 |
|                          | - davinci-002            | - gpt-4                  |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| MosaicML                 | - mpt-7b-instruct        | - llama2-70b-chat†       | - instructor-large       |
|                          | - mpt-30b-instruct       |                          | - instructor-xl          |
|                          | - llama2-70b-chat†       |                          |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Anthropic                | - claude-instant-1.2     | - claude-instant-1.2     | N/A                      |
|                          | - claude-2.1             | - claude-2.1             |                          |
|                          | - claude-2.0             | - claude-2.0             |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+
| Cohere                   | - command                | - command                | - embed-english-v2.0     |
|                          | - command-light          | - command-light          | - embed-multilingual-v2.0|
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
+--------------------------+--------------------------+--------------------------+--------------------------+
| TogetherAI               | - google/gemma-2b        | - dbrx-instruct          |  - BAAI/bge-large-en-v1.5|
|                          | - microsoft/phi-2        |                          |                          |
+--------------------------+--------------------------+--------------------------+--------------------------+

§ For full compatibility references for ``OpenAI``, see the `OpenAI Model Compatibility Matrix <https://platform.openai.com/docs/models/model-endpoint-compatibility>`_.

† Llama 2 is licensed under the `LLAMA 2 Community License <https://ai.meta.com/llama/license/>`_, Copyright © Meta Platforms, Inc. All Rights Reserved.

Within each model block in the configuration file, the provider field is used to specify the name
of the provider for that model. This is a string value that needs to correspond to a provider the MLflow Deployments Server supports.

.. note::
    `*` MLflow Model Serving will only work for chat or completions if the output return is in an endpoint-compatible format. The
    response must conform to either an output of ``{"predictions": str}`` or ``{"predictions": {"candidates": str}}``. Any complex return type from a model that
    does not conform to these structures will raise an exception at query time.

    `**` Embeddings support is only available for models whose response signatures conform to the structured format of ``{"predictions": List[float]}``
    or ``{"predictions": List[List[float]]}``. Any other return type will raise an exception at query time. ``FeatureExtractionPipeline`` in ``transformers`` and
    models using the ``sentence_transformers`` flavor will return the correct data structures for the embeddings endpoint.

Here's an example of a provider configuration within an endpoint:

.. code-block:: yaml

    endpoints:
      - name: chat
        endpoint_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4
          config:
            openai_api_key: $OPENAI_API_KEY
        limit:
          renewal_period: minute
          calls: 10

In the above configuration, ``openai`` is the `provider` for the model.

As of now, the MLflow Deployments Server supports the following providers:

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

More providers are being added continually. Check the latest version of the MLflow Deployments Server Docs for the
most up-to-date list of supported providers.

If you would like to use a LLM model that is not offered by the above providers, or if you
would like to integrate a private LLM model, you can create a :ref:`provider plugin <deployments_plugin>`
to integrate with the MLflow Deployments Server.

.. _deployments-endpoints:

Endpoints
---------

`Endpoints` are central to how the MLflow Deployments Server functions. Each endpoint acts as a proxy endpoint for the
user, forwarding requests to the underlying :ref:`deployments_models` and :ref:`providers` specified in the configuration file.

an endpoint in the MLflow Deployments Server consists of the following fields:

* **name**: This is the unique identifier for the endpoint. This will be part of the URL when making API calls via the MLflow Deployments Server.

* **type**: The type of the endpoint corresponds to the type of language model interaction you desire. For instance, ``llm/v1/completions`` for text completion operations, ``llm/v1/embeddings`` for text embeddings, and ``llm/v1/chat`` for chat operations.

* **model**: Defines the model to which this endpoint will forward requests. The model contains the following details:

    * **provider**: Specifies the name of the :ref:`provider <providers>` for this model. For example, ``openai`` for OpenAI's ``GPT-4o`` models.
    * **name**: The name of the model to use. For example, ``gpt-4o-mini`` for OpenAI's ``GPT-4o-Mini`` model.
    * **config**: Contains any additional configuration details required for the model. This includes specifying the API base URL and the API key.

* **limit**: Specify the rate limit setting this endpoint will follow. The limit field contains the following fields:

    * **renewal_period**: The time unit of the rate limit, one of [second|minute|hour|day|month|year].
    * **calls**: The number of calls this endpoint will accept within the specified time unit.

Here's an example of an endpoint configuration:

.. code-block:: yaml

    endpoints:
      - name: completions
        endpoint_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY
        limit:
          renewal_period: minute
          calls: 10

In the example above, a request sent to the completions endpoint would be forwarded to the
``gpt-4o-mini`` model provided by ``openai``.

The endpoints in the configuration file can be updated at any time, and the MLflow Deployments Server will
automatically update its available endpoints without requiring a restart. This feature provides you
with the flexibility to add, remove, or modify endpoints as your needs change. It enables 'hot-swapping'
of endpoints, providing a seamless experience for any applications or services that interact with the MLflow Deployments Server.

When defining endpoints in the configuration file, ensure that each name is unique to prevent conflicts.
Duplicate endpoint names will raise an ``MlflowException``.

.. _deployments_models:

Models
------

The ``model`` section within an ``endpoint`` specifies which model to use for generating responses.
This configuration block needs to contain a ``name`` field which is used to specify the exact model instance to be used.
Additionally, a :ref:`provider <providers>` needs to be specified, one that you have an authenticated access api key for.

Different endpoint types are often associated with specific models.
For instance, the ``llm/v1/chat`` and ``llm/v1/completions`` endpoints are generally associated with
conversational models, while ``llm/v1/embeddings`` endpoints would typically be associated with
embedding or transformer models. The model you choose should be appropriate for the type of endpoint specified.

Here's an example of a model name configuration within an endpoint:

.. code-block:: yaml

    endpoints:
      - name: embeddings
        endpoint_type: llm/v1/embeddings
        model:
          provider: openai
          name: text-embedding-ada-002
          config:
            openai_api_key: $OPENAI_API_KEY


In the above configuration, ``text-embedding-ada-002`` is the model used for the embeddings endpoint.

When specifying a model, it is critical that the provider supports the model you are requesting.
For instance, ``openai`` as a provider supports models like ``text-embedding-ada-002``, but other providers
may not. If the model is not supported by the provider, the MLflow Deployments Server will return an HTTP 4xx error
when trying to route requests to that model.

.. important::

    Always check the latest documentation of the specified provider to ensure that the model you want
    to use is supported for the type of endpoint you're configuring.

Remember, the model you choose directly affects the results of the responses you'll get from the
API calls. Therefore, choose a model that fits your use-case requirements. For instance,
for generating conversational responses, you would typically choose a chat model.
Conversely, for generating embeddings of text, you would choose an embedding model.

.. _deployments_configuration:

Configuring the Deployments Server
==================================

The MLflow Deployments Server relies on a user-provided configuration file, written in YAML,
that defines the endpoints and providers available to the server. The configuration file dictates
how the deployments server interacts with various language model providers and determines the end-points that
users can access.

Deployments Server Configuration
--------------------------------

The configuration file includes a series of sections, each representing a unique endpoint.
Each endpoint section has a name, a type, and a model specification, which includes the model
provider, name, and configuration details. The configuration section typically contains the base
URL for the API and an environment variable for the API key.

Here is an example of a single-endpoint configuration:

.. code-block:: yaml

    endpoints:
      - name: chat
        endpoint_type: llm/v1/chat
        model:
          provider: openai
          name: gpt-4o-mini
          config:
            openai_api_key: $OPENAI_API_KEY
        limit:
          renewal_period: minute
          calls: 10


In this example, we define an endpoint named ``chat`` that corresponds to the ``llm/v1/chat`` type, which
will use the ``gpt-4o-mini`` model from OpenAI to return query responses from the OpenAI service, and accept up to 10 requests per minute.

The MLflow Deployments Server configuration is very easy to update.
Simply edit the configuration file and save your changes, and the MLflow Deployments Server will automatically
update the endpoints with zero disruption or down time. This allows you to try out new providers or model types while keeping your applications steady and reliable.

In order to define an API key for a given provider, there are three primary options:

1. Directly include it in the YAML configuration file.
2. Use an environment variable to store the API key and reference it in the YAML configuration file.
3. Define your API key in a file and reference the location of that key-bearing file within the YAML configuration file.

If you choose to include the API key directly, replace ``$OPENAI_API_KEY`` in the YAML file with your
actual API key.

.. warning::

    The MLflow Deployments Server provides direct access to billed external LLM services. It is strongly recommended to restrict access to this server. See the section on :ref:`security <deployments_security>` for guidance.

If you prefer to use an environment variable (recommended), you can define it in your shell
environment. For example:

.. code-block:: bash

     export OPENAI_API_KEY="your_openai_api_key"

**Note:** Replace "your_openai_api_key" with your actual OpenAI API key.

.. _deployments_configuration_details:

Deployments Server Configuration Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MLflow Deployments Server relies on a user-provided configuration file. It defines how the deployments server interacts with various language model providers and dictates the endpoints that users can access.

The configuration file is written in YAML and includes a series of sections, each representing a unique endpoint. Each endpoint section has a name, a type, and a model specification, which includes the provider, model name, and provider-specific configuration details.

Here are the details of each configuration parameter:

General Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **endpoints**: This is a list of endpoint configurations. Each endpoint represents a unique endpoint that maps to a particular language model service.

Each endpoint has the following configuration parameters:

- **name**: This is the name of the endpoint. It needs to be a unique name without spaces or any non-alphanumeric characters other than hyphen and underscore.

- **endpoint_type**: This specifies the type of service offered by this endpoint. This determines the interface for inputs to an endpoint and the returned outputs. Current supported endpoint types are:

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

Top-level model configuration for Amazon Bedrock endpoints must be one of the following two supported authentication modes: `key-based` or `role-based`.

+--------------------------+----------+------------------------------+-------------------------------------------------------+
| Configuration Parameter  | Required | Default                      | Description                                           |
+==========================+==========+==============================+=======================================================+
| **aws_config**           | No       |                              | An object with either the key-based or role-based     |
|                          |          |                              | schema below.                                         |
+--------------------------+----------+------------------------------+-------------------------------------------------------+


To use key-based authentication, define an Amazon Bedrock endpoint with the required fields below.
.. note::

  If using a configured endpoint purely for development or testing, utilizing an IAM User role or a temporary short-lived standard IAM role are recommended; while for production deployments, a standard long-expiry IAM role is recommended to ensure that the endpoint is capable of handling authentication for a long period. If the authentication expires and a new set of keys need to be supplied, the endpoint must be recreated in order to persist the new keys.

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

Alternatively, for role-based authentication, an Amazon Bedrock endpoint can be defined and initialized with an a IAM Role  ARN that is authorized to access Bedrock.  The MLflow Deployments Server will attempt to assume this role with using the standard credential provider chain and will renew the role credentials if they have expired.

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

Querying the Deployments Server
===============================

Once the MLflow Deployments Server has been configured and started, it is ready to receive traffic from users.

.. _standard_deployments_parameters:

Standard Query Parameters
-------------------------

The MLflow Deployments Server defines standard parameters for chat, completions, and embeddings that can be
used when querying any endpoint regardless of its provider. Each parameter has a standard range and
default value. When querying an endpoint with a particular provider, the MLflow Deployments Server automatically
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
In addition to the :ref:`standard_query_parameters`, you can pass any additional parameters supported by the endpoint's provider as part of your query. For example:

- ``logit_bias`` (supported by OpenAI, Cohere)
- ``top_k`` (supported by MosaicML, Anthropic, PaLM, Cohere)
- ``frequency_penalty`` (supported by OpenAI, Cohere, AI21 Labs)
- ``presence_penalty`` (supported by OpenAI, Cohere, AI21 Labs)
- ``stream`` (supported by OpenAI, Cohere)

Below is an example of submitting a query request to an MLflow Deployments Server endpoint using additional parameters:

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

FastAPI, the framework used for building the MLflow Deployments Server, provides an automatic interactive API
documentation interface, which is accessible at the "/docs" endpoint (e.g., ``http://my.deployments:9000/docs``).
This interactive interface is very handy for exploring and testing the available API endpoints.

As a convenience, accessing the root URL (e.g., ``http://my.deployments:9000``) redirects to this "/docs" endpoint.

MLflow Python Client APIs
-------------------------
:class:`MlflowDeploymentClient <mlflow.deployments.MlflowDeploymentClient>` is the user-facing client API that is used to interact with the MLflow Deployments Server.
It abstracts the HTTP requests to the Deployments Server via a simple, easy-to-use Python API.

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
This integration enable users to use prompt engineering, retrieval augmented generation, and other techniques with LLMs in the deployments server.

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
Interfacing with MLflow Models can be done in two ways. With the use of a custom PyFunc Model, a query can be issued directly to a deployments server endpoint and used in a broader context within a model.
Data may be augmented, manipulated, or used in a mixture of experts paradigm. The other means of utilizing the MLflow Deployments Server along with MLflow Models is to define a served MLflow model directly as
an endpoint within a deployments server.

Using the Deployments Server to Query a served MLflow Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a full walkthrough and example of using the MLflow serving integration to query a model directly through the MLflow Deployments Server, please see `the full example <https://github.com/mlflow/mlflow/tree/master/examples/deployments/mlflow_serving/README.md>`_.
Within the guide, you will see the entire end-to-end process of serving multiple models from different servers and configuring an MLflow Deployments Server instance to provide a single unified point to handle queries from.

Using an MLflow Model to Query the Deployments Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also build and deploy MLflow Models that call the MLflow Deployments Server.
The example below demonstrates how to use a deployments server from within a custom ``pyfunc`` model.


.. note::
    The custom ``Model`` shown in the example below is utilizing environment variables for the deployments server's uri. These values can also be set manually within the
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
The REST API allows you to send HTTP requests directly to the MLflow Deployments Server. This is useful if you're not using Python or if you prefer to interact with a deployments server using HTTP directly.

Here are some examples for how you might use curl to interact with the MLflow Deployments Server:

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

**Note:** Remember to replace ``my.deployments:8888`` with the URL of your actual MLflow Deployments Server.

.. _deployments_plugin:

Plugin LLM Provider (Experimental)
==================================

.. attention::
    This feature is in active development and is marked as Experimental. It may change in a future release without warning.

The MLflow Deployments Server supports the use of custom language model providers through the use of plugins.
A plugin is a Python package that provides a custom implementation of a language model provider.
This allows users to integrate their own language model services with the MLflow Deployments Server.

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

Finally, you need to install the plugin package in the same environment as the MLflow Deployments Server.

.. important::

    Only install plugin packages from sources that you trust. Starting a server with a plugin provider will
    execute any arbitrary code that is defined within the plugin package.

Then, you can specify the plugin provider according to the entry point name
in the MLflow Deployments Server configuration file.

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

MLflow Deployments Server API Documentation
===========================================

`API documentation <./api.html>`_

OpenAI Compatibility
====================

MLflow Deployments Server is compatible with OpenAI API and supports the ``chat``, ``completions``, and ``embeddings`` APIs.
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

    mlflow deployments start-server --config-path /path/to/config.yaml --port 7000

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

See `Unity Catalog Integration <./uc_integration.html>`_ for how to integrate the MLflow Deployments Server with Unity Catalog.

.. _deployments_security:

Deployments Server Security Considerations
==========================================

Remember to ensure secure access to the system that the MLflow Deployments Server is running in to protect access to these keys.

An effective way to secure your deployments server is by placing it behind a reverse proxy. This will allow the reverse proxy to handle incoming requests and forward them to the MLflow Deployments Server. The reverse proxy effectively shields your application from direct exposure to Internet traffic.

A popular choice for a reverse proxy is `Nginx`. In addition to handling the traffic to your application, `Nginx` can also serve static files and load balance the traffic if you have multiple instances of your application running.

Furthermore, to ensure the integrity and confidentiality of data between the client and the server, it's highly recommended to enable HTTPS on your reverse proxy.

In addition to the reverse proxy, it's also recommended to add an authentication layer before the requests reach the MLflow Deployments Server. This could be HTTP Basic Authentication, OAuth, or any other method that suits your needs.

For example, here's a simple configuration for Nginx with Basic Authentication:

.. code-block:: nginx

    http {
        server {
            listen 80;

            location / {
                auth_basic "Restricted Content";
                auth_basic_user_file /etc/nginx/.htpasswd;

                proxy_pass http://localhost:5000;  # Replace with the MLflow Deployments Server port
            }
        }
    }

In this example, `/etc/nginx/.htpasswd` is a file that contains the username and password for authentication.

These measures, together with a proper network setup, can significantly improve the security of your system and ensure that only authorized users have access to submit requests to your LLM services.
