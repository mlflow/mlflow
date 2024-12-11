Tasks in MLflow Transformers Flavor
===================================

This page provides an overview of how to use the ``task`` parameter in the MLflow Transformers flavor to control the inference interface of the model.

.. contents::
   :local:
   :depth: 2

Overview
--------

In the MLflow Transformers flavor, ``task`` plays a crucial role in determining the input and output format of the model. The ``task`` is a fundamental concept in the Transformers library, which describe the structure of each model's API (inputs and outputs) and are used to determine which Inference API and widget we want to display for any given model.

MLflow utilizes this concept to determine the input and output format of the model, persists the correct `Model Signature <https://mlflow.org/docs/latest/models.html#model-signatures-and-input-examples>`_, and provides a consistent `Pyfunc Inference API <https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#inference-api>`_ for serving different types of models. Additionally, on top of the native Transformers task types, MLflow defines a few additional task types to support more complex use cases, such as chat-style applications.


Native Transformers Task Types
------------------------------

For native Transformers tasks, MLflow will automatically infer the task type from the pipeline when you save a pipeline with :py:func:`mlflow.transformers.log_model()`. You can also specify the task type explicitly by passing the ``task`` parameter. The full list of supported task types is available in the `Transformers documentation <https://huggingface.co/tasks>`_, but note that **not all task types are supported in MLflow**.

.. code-block:: python

    import mlflow
    import transformers

    pipeline = transformers.pipeline("text-generation", model="gpt2")

    with mlflow.start_run():
        model_info = mlflow.transformers.save_model(
            transformers_model=pipeline,
            artifact_path="model",
            save_pretrained=False,
        )

    print(f"Inferred task: {model_info.flavors['transformers']['task']}")
    # >> Inferred task: text-generation


Advanced Tasks for OpenAI-Compatible Inference
----------------------------------------------

In addition to the native Transformers task types, MLflow defines a few additional task types. Those advanced task types allows you to extend the Transformers pipeline with OpenAI-compatible inference interface, to serve models for specific use cases.
In addition to the native Transformers task types, MLflow defines several additional task types. These advanced task types allow you to extend the Transformers pipeline with an OpenAI-compatible inference interface to serve models for specific use cases.

For example, the Transformers ``text-generation`` pipeline inputs and outputs a single string or a list of strings. However, when serving a model, it is often necessary to have a more structured input and output format. For instance, in a chat-style application, the input may be a list of messages.

To support these use cases, MLflow defines a set of advanced task types prefixed with ``llm/v1``:

- ``"llm/v1/chat"`` for chat-style applications
- ``"llm/v1/completions"`` for generic completions
- ``"llm/v1/embeddings"`` for text embeddings generation

The required step to use these advanced task types is just to specify the ``task`` parameter as an ``llm/v1`` task when logging the models.

.. code-block:: python

    import mlflow

    with mlflow.start_run():
        mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="model",
            task="llm/v1/chat",  # <= Specify the llm/v1 task type
            # Optional, recommended for large models to avoid creating a local copy of the model weights
            save_pretrained=False,
        )

.. note::

    This feature is only available in MLflow 2.11.0 and above. Also, the ``llm/v1/chat`` task type is only available for models saved with ``transformers >= 4.34.0``.

Input and Output Formats
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Task
     - Supported pipeline
     - Input
     - Output
   * - ``llm/v1/chat``
     - ``text-generation``
     - `Chat API spec <https://mlflow.org/docs/latest/llms/deployments/index.html#chat>`_
     - Returns a `Chat Completion <https://platform.openai.com/docs/api-reference/chat/object>`_ object in the json format.
   * - ``llm/v1/completions``
     - ``text-generation``
     - `Completions API spec <https://mlflow.org/docs/latest/llms/deployments/index.html#completions>`_
     - Returns a `Completion <https://platform.openai.com/docs/guides/text-generation/completions-api>`_ object in the json format.
   * - ``llm/v1/embeddings``
     - ``feature-extraction``
     - `Embeddings API spec <https://mlflow.org/docs/latest/llms/deployments/index.html#embeddings>`_
     - Returns a list of `Embedding <https://platform.openai.com/docs/api-reference/embeddings/object>`_ object. Additionally, the model returns ``usage`` field, which contains the number of tokens used for the embeddings generation.

.. note::

    The Completion API is considered as legacy, but it is still supported in MLflow for backward compatibility. We recommend using the Chat API for compatibility with the latest APIs from OpenAI and other model providers.

Code Example of Using ``llm/v1`` Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code snippet demonstrates how to log a Transformers pipeline with the ``llm/v1/chat`` task type, and use the model for chat-style inference. Check out the `notebook tutorial <tutorials/conversational/pyfunc-chat-model.html>`_ to see more examples in action!

.. code-block:: python

    import mlflow
    import transformers

    pipeline = transformers.pipeline("text-generation", "gpt2")

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="model",
            task="llm/v1/chat",
            input_example={
                "messages": [
                    {"role": "system", "content": "You are a bot."},
                    {"role": "user", "content": "Hello, how are you?"},
                ]
            },
            save_pretrained=False,
        )

    # Model metadata logs additional field "inference_task"
    print(model_info.flavors["transformers"]["inference_task"])
    # >> llm/v1/chat

    # The original native task type is also saved
    print(model_info.flavors["transformers"]["task"])
    # >> text-generation

    # Model signature is set to the chat API spec
    print(model_info.signature)
    # >> inputs:
    # >>   ['messages': Array({content: string (required), name: string (optional), role: string (required)}) (required), 'temperature': double (optional), 'max_tokens': long (optional), 'stop': Array(string) (optional), 'n': long (optional), 'stream': boolean (optional)]
    # >> outputs:
    # >>   ['id': string (required), 'object': string (required), 'created': long (required), 'model': string (required), 'choices': Array({finish_reason: string (required), index: long (required), message: {content: string (required), name: string (optional), role: string (required)} (required)}) (required), 'usage': {completion_tokens: long (required), prompt_tokens: long (required), total_tokens: long (required)} (required)]
    # >> params:
    # >>     None

    # The model can be served with the OpenAI-compatible inference API
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    prediction = pyfunc_model.predict(
        {
            "messages": [
                {"role": "system", "content": "You are a bot."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "temperature": 0.5,
            "max_tokens": 200,
        }
    )
    print(prediction)
    # >> [{'choices': [{'finish_reason': 'stop',
    # >>               'index': 0,
    # >>               'message': {'content': 'I'm doing well, thank you for asking.', 'role': 'assistant'}},
    # >>   'created': 1719875820,
    # >>   'id': '355c4e9e-040b-46b0-bf22-00e93486100c',
    # >>   'model': 'gpt2',
    # >>   'object': 'chat.completion',
    # >>   'usage': {'completion_tokens': 7, 'prompt_tokens': 13, 'total_tokens': 20}}]

Note that the input and output modifications only apply when the model is loaded with :py:func:`mlflow.pyfunc.load_model()` (e.g. when
serving the model with the ``mlflow models serve`` CLI tool). If you want to load just the raw pipeline, you can
use :py:func:`mlflow.transformers.load_model()`.

Provisioned Throughput on Databricks Model Serving
--------------------------------------------------

`Provisioned Throughput <https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html>`_ on Databricks Model Serving is a capability that optimizes inference performance for foundation models with performance guarantees. To serve Transformers models with provisioned throughput, specify ``llm/v1/xxx`` task type when logging the model. MLflow logs the required metadata to enable provisioned throughput on Databricks Model Serving.


.. tip::

    When logging large models, you can use ``save_pretrained=False`` to avoid creating a local copy of the model weights for saving time and disk space. Please refer to the :ref:`documentation <transformers-save-pretrained-guide>` for more details.

FAQ
---

How to override the default query parameters for the OpenAI-compatible inference?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When serving the model saved with the ``llm/v1`` task type, MLflow uses the same default value as OpenAI APIs for the parameters like ``temperature`` and ``stop``. You can override them by either passing the values at inference time, or by setting different default values when logging the model.

1. At inference time: You can pass the parameters as part of the input dictionary when calling the ``predict()`` method, just like how you pass the input messages.
2. When logging the model: You can override the default values for the parameters by saving a ``model_config`` parameter when logging the model.

.. code-block:: python

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="model",
            task="llm/v1/chat",
            model_config={
                "temperature": 0.5,  # <= Set the default temperature
                "stop": ["foo", "bar"],  # <= Set the default stop sequence
            },
            save_pretrained=False,
        )


.. attention::

    The ``stop`` parameter can be used to specify the stop sequence for the ``llm/v1/chat`` and ``llm/v1/completions`` tasks. We emulate the behavior of the ``stop`` parameter in the OpenAI APIs by passing the `stopping_criteria <https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.generate.stopping_criteria>`_ to the Transformers pipeline, with the token IDs of the given stop sequence. However, the behavior may not be stable because the tokenizer does not always generate the same token IDs for the same sequence in different sentences, especially for ``sentence-piece`` based tokenizers.
