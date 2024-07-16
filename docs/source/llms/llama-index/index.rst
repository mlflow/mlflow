MLflow LlamaIndex Flavor
========================

.. attention::
    The ``llama_index`` flavor is under active development and is marked as Experimental. Public APIs are
    subject to change, and new features may be added as the flavor evolves.

Introduction
------------

**LlamaIndex** 🦙 is a powerful data centric framework designed to seamlessly connect custom data sources to large language models (LLMs).
It offers a comprehensive suite of data structures and tools that simplify the process of ingesting, structuring, and
accessing private or domain-specific data for use with LLMs. LlamaIndex excels in enabling context-aware AI applications
by providing efficient indexing and retrieval mechanisms, making it easier to build advanced QA systems, chatbots,
and other AI-driven applications that require integration of external knowledge.

.. figure:: ../../_static/images/llms/llama-index/llama-index-gateway.png
    :alt: Overview of LlamaIndex and MLflow integration
    :width: 70%
    :align: center


Why use LlamaIndex with MLflow?
-------------------------------

The integration of the LlamaIndex library with MLflow provides a seamless experience for managing and deploying LlamaIndex engines. The following are some of the key benefits of using LlamaIndex with MLflow:

* `MLflow Tracking <../../tracking.html>`_ allows you to track your index to MLflow and manage a lot of moving parts in your LlamaIndex project, such as prompts, LLMs, retrievers, tools, global configurations, and more.

* `MLflow Model <../../models.html>`_ packages your LlamaIndex engine with all its dependency versions, input and output interfaces, and other essential metadata. This allows you to deploy your LlamaIndex engine with ease, knowing that the environment is consistent across different stages of the ML lifecycle.

* `MLflow Evaluate <../llm-evaluate/index.html>`_ provides native capabilities within MLflow to **evaluate** language models. This capability facilitates the efficient assessment of inference results from your LlamaIndex engine, ensuring robust performance analytics and facilitating quick iterations.

* `MLflow Tracing <../tracing/index.html>`_ is a powerful **observability** tool for monitoring and debugging what happens inside the LlamaIndex models, helping you identifying potential bottlenecks or issues quickly. With its powerful automatic logging capability, you can instrument your LlamaIndex application without any code change but just running a single command.



Getting Started
---------------

TBA


Concepts
--------

``Index``
^^^^^^^^^

The ``Index`` object is the core foundation in LlamaIndex, which is a collection of documents that are indexed for fast information retrieval. such as Retrieval-Augmented generation (RAG) and Agents use cases. The ``Index`` object can be logged to MLflow experiments and loaded back as an inference engine.

``Engine``
^^^^^^^^^^

The ``Engine`` is a generic interface built on top of the ``Index`` object, which provides a set of APIs to interact with the index. LlamaIndex provides two types of engines: ``QueryEngine`` and ``ChatEngine``. The ``QueryEngine`` simply takes a single
query and returns a response based on the index. The ``ChatEngine`` is designed for conversational agents, which keeps track of the conversation history as well.


Usage
-----

.. toctree::
    :maxdepth: 2

Saving and Loading Index in MLflow Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating an Index
~~~~~~~~~~~~~~~~~

The ``index`` object is a centerpiece of the LlamaIndex and MLflow integration. With LlamaIndex, you can create an index from a collection of documents. Following code creates a sample index from Paul Graham's essay data available on the LlamaIndex repository.

.. code-block:: shell

    mkdir -p data
    curl -L https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -o ./data/paul_graham_essay.txt

.. code-block:: python

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)

Logging the Index to MLflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can log the ``index`` object to the MLflow experiment using the :py:func:`mlflow.llama_index.log_model` function.

One key step here is to specify the ``engine_type`` parameter. The choice of engine type does not affect the index itself,
but dictates the interface of how you query the index when you load it back for inference.

1. QueryEngine (``engine_type="query"``) is designed for simple query-response system that takes a single query string and returns a response.
2. ChatEngine (``engine_type="chat"``) is designed for conversational agent that keeps track of the conversation history and answer tp the user query based on the context.
3. Retriever (``engine_type="retriever"``) is a lower level component that returns the top-k relevant documents matching the query.


The following code is an example of logging the index to MLflow with chat engine type.

.. code-block:: python

    import mlflow

    mlflow.set_experiment("llama-index-demo")

    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model(
            index,
            artifact_path="index",
            engine_type="chat",
            input_example="What did the author do growing up?",
        )

.. figure:: ../../_static/images/llms/llama-index/llama-index-artifacts.png
    :alt: MLflow artifacts for the LlamaIndex index
    :width: 80%
    :align: center

.. tip::

    Under the hood, MLflow calls ``as_query_engine()`` / ``as_chat_engine()`` / ``as_retriever()`` method on the index object to convert it to the respective engine instance.

Loading the Index Back for inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The saved index can be loaded back for inference using the :py:func:`mlflow.pyfunc.load_model` function. This function
gives an MLflow Python Model backed by the LlamaIndex engine, with the engine type specified during logging.

.. code-block:: python

    import mlflow

    model = mlflow.pyfunc.load_model(model_info.model_uri)

    response = model.predict("What was the first program the author wrote?")
    print(response)
    # >> The first program the author wrote was on the IBM 1401 ...

    # The chat engine keeps track of the conversation history
    response = model.predict("How did the author feel about it?")
    print(response)
    # >> The author felt puzzled by the first program ...


.. tip::

    To load the index itself back instead of the engine, use the :py:func:`mlflow.llama_index.load_model` function.

    .. code-block:: python

        index = mlflow.llama_index.load_model("runs:/<run_id>/index")


Enable Tracing
^^^^^^^^^^^^^^

You can easily enable tracing for your LlamaIndex engine calling the :py:func:`mlflow.llama_index.autolog` function. This function automatically logs the input and output of the LlamaIndex engine to MLflow, providing you with a detailed view of the engine's behavior.

.. code-block:: python

    import mlflow

    mlflow.llama_index.autolog()

    chat_engine = index.as_chat_engine()
    response = chat_engine.chat("What was the first program the author wrote?")

Then you can navigate to the MLflow UI, select the experiment, and open the "Traces" tab to find the logged trace for the prediction made by the engine. It is impressive to see how the chat engine coordinates and executes numbers of tasks to answer your question!

.. figure:: ../../_static/images/llms/llama-index/llama-index-trace.png
    :alt: Trace view in MLflow UI
    :width: 80%
    :align: center

You can disable tracing by running the same function with the ``disable`` parameter set to ``True``:

.. code-block:: python

    mlflow.llama_index.autolog(disable=True)


.. note::

    The tracing supports async prediction and streaming response, however, it does not
    support the combination of async and streaming, such as ``astream_chat`` method.
