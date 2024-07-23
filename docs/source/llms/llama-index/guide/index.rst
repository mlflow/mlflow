LlamaIndex within MLflow (Experimental)
=======================================

.. attention::
   The ``llama_index`` flavor is currently under active development and is marked as Experimental. Public APIs are evolving, and new features are being added to enhance its functionality.


LlamaIndex Overview
-------------------
LlamaIndex is a comprehensive tool that offers robust support for Retrieval-Augmented Generation (RAG), 
enabling efficient and effective information retrieval combined with text generation. It provides 
native interaction with a variety of Large Language Models (LLMs) and embedding models, facilitating
seamless integration and usage of advanced machine learning models. Additionally, LlamaIndex 
features strong support for callbacks, allowing users to easily manage and monitor the execution of
processes. The tool also boasts strong community integration through Llama Hub, which includes a 
rich collection of data loaders, agents, datasets, and other enhancements.


LlamaIndex with MLflow
----------------------
MLflow enhances LlamaIndex's functionality by supporting the entire model development lifecycle. 
Key benefits include the ability to track runs and experiments, manage model versioning, and 
maintain environment tracking via Conda and Python environments. 

For generative AI specifically, MLflow boasts additional functionality listed below:

* Tracing: A method of tracking agent chain lineage and the associated duration of each step, which facilitates model explainability and debugging.  
* Prompt Engineering UI: A user interface for designing, testing, and refining prompts for language models.
* MLflow Deployments Server: A highly performant and modular model deployment framework that easily allows you to switch out base LLMs.
* GenAI Evaluation: Tools for assessing generative AI model outputs, enabling quality, relevance, and accuracy evaluation.


FAQ
---

I have an index logged with ``query`` engine type. Can I load it back a ``chat`` engine?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While it is not possible to update the engine type of the logged model in-place,
you can always load the index back and re-log it with the desired engine type. This process
does **not require re-creating the index**, so it is an efficient way to switch between
different engine types.

.. code-block:: python

    import mlflow

    # Log the index with the query engine type first
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index,
            artifact_path="index-query",
            engine_type="query",
        )

    # Load the index back and re-log it with the chat engine type
    index = mlflow.llama_index.load_model(model_info.model_uri)
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index,
            artifact_path="index-chat",
            # Specify the chat engine type this time
            engine_type="chat",
        )

Alternatively, you can leverage their standard inference APIs on the loaded LlamaIndex native index object, specifically:

* ``index.as_chat_engine().chat("hi")``
* ``index.as_query_engine().query("hi")``
* ``index.as_retriever().retrieve("hi")``


How to use different LLMs for inference with the loaded engine?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When saving the index to MLflow, it persists the global `Settings <https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/>`_ object as a part of the model. This object contains settings such as LLM and embedding
models to be used by engines.

.. code-block:: python

    import mlflow
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI("gpt-4o-mini")

    # MLflow saves GPT-3.5-turbo as the LLM to use for inference
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index, artifact_path="index", engine_type="chat"
        )

Then later when you load the index back, the persisted settings are also applied globally. This means that the loaded engine will use the same LLM as when it was logged.

However, sometimes you may want to use a different LLM for inference. In such cases, you can update the global ``Settings`` object directly after loading the index.

.. code-block:: python

    import mlflow

    # Load the index back
    loaded_index = mlflow.llama_index.load_model(model_info.model_uri)

    assert Settings.llm.model == "gpt-4o-mini"


    # Update the settings to use GPT-4 instead
    Settings.llm = OpenAI("gpt-4")
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("What is the capital of France?")
