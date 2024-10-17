LangChain within MLflow (Experimental)
======================================

.. attention::
   The ``langchain`` flavor is currently under active development and is marked as Experimental. Public APIs are evolving, and new features are being added to enhance its functionality.

Overview
--------
`LangChain <https://www.langchain.com/>`_ is a Python framework for creating applications powered by language models. It offers unique features for developing context-aware 
applications that utilize language models for reasoning and generating responses. This integration with MLflow streamlines the development and 
deployment of complex NLP applications.

LangChain's Technical Essence
-----------------------------

- **Context-Aware Applications**: LangChain specializes in connecting language models to various sources of context, enabling them to produce more relevant and accurate outputs.
- **Reasoning Capabilities**: It uses the power of language models to reason about the given context and take appropriate actions based on it.
- **Flexible Chain Composition**: The LangChain Expression Language (LCEL) allows for easy construction of complex chains from basic components, supporting functionalities like streaming, parallelism, and logging.

Building Chains with LangChain
------------------------------

- **Basic Components**: LangChain facilitates chaining together components like prompt templates, models, and output parsers to create complex workflows.
- **Example - Joke Generator**:
  - A basic chain can take a topic and generate a joke using a combination of a prompt template, a ChatOpenAI model, and an output parser.
  - The components are chained using the `|` operator, similar to a Unix pipe, allowing the output of one component to feed into the next.
- **Advanced Use Cases**:
  - LangChain also supports more complex setups, like Retrieval-Augmented Generation (RAG) chains, which can add context when responding to questions.

Integration with MLflow
-----------------------

- **Simplified Logging and Loading**: MLflow's `langchain` flavor provides functions like `log_model()` and `load_model()`, enabling easy logging and retrieval of LangChain models within the MLflow ecosystem.
- **Simplified Deployment**: LangChain models logged in MLflow can be interpreted as generic Python functions, simplifying their deployment and use in diverse applications. With dependency management incorporated directly into your logged model, you can deploy your application knowing that the environment that you used to train the model is what will be used to serve it.
- **Versatile Model Interaction**: The integration allows developers to leverage LangChain's unique features in conjunction with MLflow's robust model tracking and management capabilities.
- **Autologging**: MLflow's `langchain` flavor provides autologging of LangChain models, which automatically logs artifacts, metrics and models for inference.

The ``langchain`` model flavor enables logging of `LangChain models <https://github.com/hwchase17/langchain>`_ in MLflow format via
the :py:func:`mlflow.langchain.save_model()` and :py:func:`mlflow.langchain.log_model()` functions. Use of these
functions also adds the ``python_function`` flavor to the MLflow Models that they produce, allowing the model to be
interpreted as a generic Python function for inference via :py:func:`mlflow.pyfunc.load_model()`.

You can also use the :py:func:`mlflow.langchain.load_model()` function to load a saved or logged MLflow
Model with the ``langchain`` flavor as a dictionary of the model's attributes.

Basic Example: Logging a LangChain ``LLMChain`` in MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../../examples/langchain/simple_chain.py
    :language: python

The output of the example is shown below:

.. code-block:: python
    :caption: Output

    ["\n\nColorful Cozy Creations."]

What the Simple LLMChain Example Showcases
""""""""""""""""""""""""""""""""""""""""""

- **Integration Flexibility**: The example highlights how LangChain's LLMChain, consisting of an OpenAI model and a custom prompt template, can be easily logged in MLflow.
- **Simplified Model Management**: Through MLflow's `langchain` flavor, the chain is logged, enabling version control, tracking, and easy retrieval.
- **Ease of Deployment**: The logged LangChain model is loaded using MLflow's `pyfunc` module, illustrating the straightforward deployment process for LangChain models within MLflow.
- **Practical Application**: The final prediction step demonstrates the model's functionality in a real-world scenario, generating a company name based on a given product.


Logging a LangChain Agent with MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What is an Agent?
"""""""""""""""""

Agents in LangChain leverage language models to dynamically determine and execute a sequence of actions, contrasting with the hardcoded sequences in chains. 
To learn more about Agents and see additional examples within LangChain, you can `read the LangChain docs on Agents <https://python.langchain.com/docs/modules/agents/>`_.

Key Components of Agents
""""""""""""""""""""""""

Agent
'''''
- The core chain driving decision-making, utilizing a language model and a prompt.
- Receives inputs like tool descriptions, user objectives, and previously executed steps.
- Outputs the next action set (AgentActions) or the final response (AgentFinish).

Tools
'''''
- Functions invoked by agents to fulfill tasks.
- Essential to provide appropriate tools and accurately describe them for effective use.

Toolkits
''''''''
- Collections of tools tailored for specific tasks.
- LangChain offers a range of built-in toolkits and supports custom toolkit creation.

AgentExecutor
'''''''''''''
- The runtime environment executing agent decisions.
- Handles complexities such as tool errors and agent output parsing.
- Ensures comprehensive logging and observability.

Additional Agent Runtimes
'''''''''''''''''''''''''
- Beyond AgentExecutor, LangChain supports experimental runtimes like Plan-and-execute Agent, Baby AGI, and Auto GPT.
- Custom runtime logic creation is also facilitated.


An Example of Logging an LangChain Agent
""""""""""""""""""""""""""""""""""""""""

This example illustrates the process of logging a LangChain Agent in MLflow, highlighting the integration of LangChain's complex agent functionalities with MLflow's robust model management.


.. literalinclude:: ../../../../../examples/langchain/simple_agent.py
    :language: python

The output of the example above is shown below:

.. code-block:: python
    :caption: Output

    ["1.1044000282035853"]

What the Simple Agent Example Showcases
"""""""""""""""""""""""""""""""""""""""

- **Complex Agent Logging**: Demonstrates how LangChain's sophisticated agent, which utilizes multiple tools and a language model, can be logged in MLflow.
- **Integration of Advanced Tools**: Showcases the use of additional tools like 'serpapi' and 'llm-math' with a LangChain agent, emphasizing the framework's capability to integrate complex functionalities.
- **Agent Initialization and Usage**: Details the initialization process of a LangChain agent with specific tools and model settings, and how it can be used to perform complex queries.
- **Efficient Model Management and Deployment**: Illustrates the ease with which complex LangChain agents can be managed and deployed using MLflow, from logging to prediction.

Real-Time Streaming Outputs with LangChain and GenAI LLMs
---------------------------------------------------------

.. note::
  Stream responses via the ``predict_stream`` API are only available in MLflow versions >= 2.12.2. Previous versions of MLflow do not support streaming responses.

Overview of Streaming Output Capabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LangChain integration within MLflow enables real-time streaming outputs from various GenAI language models (LLMs) that support such functionality. 
This feature is essential for applications that require immediate, incremental responses, facilitating dynamic interactions such as conversational 
agents or live content generation.

Supported Streaming Models
^^^^^^^^^^^^^^^^^^^^^^^^^^
LangChain is designed to work seamlessly with any LLM that offers streaming output capabilities. This includes certain models from providers 
like OpenAI (e.g., specific versions of ChatGPT), as well as other LLMs from different vendors that support similar functionalities.

Using ``predict_stream`` for Streaming Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``predict_stream`` method within the MLflow pyfunc LangChain flavor is designed to handle synchronous inputs and provide outputs in a streaming manner. This method is particularly 
useful for maintaining an engaging user experience by delivering parts of the model's response as they become available, rather than waiting for the 
entire completion of the response generation.

Example Usage
^^^^^^^^^^^^^
The following example demonstrates setting up and using the ``predict_stream`` function with a LangChain model managed in MLflow, highlighting 
the real-time response generation:

.. code-block:: python

    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import OpenAI
    import mlflow


    template_instructions = "Provide brief answers to technical questions about {topic} and do not answer non-technical questions."
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=template_instructions,
    )
    chain = LLMChain(llm=OpenAI(temperature=0.05), prompt=prompt)

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(chain, "tech_chain")

    # Assuming the model is already logged in MLflow and loaded
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

    # Simulate a single synchronous input
    input_data = "Hello, can you explain streaming outputs?"

    # Generate responses in a streaming fashion
    response_stream = loaded_model.predict_stream(input_data)
    for response_part in response_stream:
        print("Streaming Response Part:", response_part)
        # Each part of the response is handled as soon as it is generated

Advanced Integration with Callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LangChain's architecture also supports the use of callbacks within the streaming output context. These callbacks can be used to enhance 
functionality by allowing actions to be triggered during the streaming process, such as logging intermediate responses or modifying them before delivery.

.. note:: 

  Most uses of callback handlers involve logging of traces involved in the various calls to services and tools within a Chain or Retriever. For purposes
  of simplicity, a simple ``stdout`` callback handler is shown below. Real-world callback handlers must be subclasses of the ``BaseCallbackHandler`` class 
  from LangChain.

.. code-block:: python

    from langchain_core.callbacks import StdOutCallbackHandler

    handler = StdOutCallbackHandler()

    # Attach callback to enhance the streaming process
    response_stream = loaded_model.predict_stream(input_data, callback_handlers=[handler])
    for enhanced_response in response_stream:
        print("Enhanced Streaming Response:", enhanced_response)

These examples and explanations show how developers can utilize the real-time streaming output capabilities of LangChain models within MLflow, 
enabling the creation of highly responsive and interactive applications.


Enhanced Management of RetrievalQA Chains with MLflow
-----------------------------------------------------

LangChain's integration with MLflow introduces a more efficient way to manage and utilize the ``RetrievalQA`` chains, a key aspect of LangChain's capabilities.
These chains adeptly combine data retrieval with question-answering processes, leveraging the strength of language models.

Key Insights into RetrievalQA Chains
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **RetrievalQA Chain Functionality**: These chains represent a sophisticated LangChain feature where information retrieval is seamlessly blended with language 
  model-based question answering. They excel in scenarios requiring the language
  model to consult specific data or documents for accurate responses.

- **Role of the Retrieval Object**: At the core of RetrievalQA chains lies the retriever object, tasked with sourcing relevant documents or data in response
  to queries.

Detailed Overview of the RAG Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Document Loaders**: Facilitate loading documents from a diverse array of sources, boasting over 100 loaders and integrations.

- **Document Transformers**: Prepare documents for retrieval by transforming and segmenting them into manageable parts.

- **Text Embedding Models**: Generate semantic embeddings of texts, enhancing the relevance and efficiency of data retrieval.

- **Vector Stores**: Specialized databases that store and facilitate the search of text embeddings.

- **Retrievers**: Employ various retrieval techniques, ranging from simple semantic searches to more sophisticated methods like the Parent Document Retriever and
  Ensemble Retriever.

Clarifying Vector Database Management with MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Traditional LangChain Serialization**: LangChain typically requires manual management for the serialization of retriever objects, including handling of the vector database.

- **MLflow's Simplification**: The ``langchain`` flavor in MLflow substantially simplifies this process. It automates serialization, managing the contents of 
  the ``persist_dir`` and the pickling of the ``loader_fn`` function.

Key MLflow Components and VectorDB Logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **persist_dir**: The directory where the retriever object, including the vector database, is stored.

2. **loader_fn**: The function for loading the retriever object from its storage location.

Important Considerations
^^^^^^^^^^^^^^^^^^^^^^^^

- **VectorDB Logging**: MLflow, through its ``langchain`` flavor, does manage the vector database as part of the retriever object. However, the vector 
  database itself is not explicitly logged as a separate entity in MLflow.

- **Runtime VectorDB Maintenance**: It's essential to maintain consistency in the vector database between the training and runtime environments. 
  While MLflow manages the serialization of the retriever object, ensuring that the same vector database is accessible during runtime remains crucial 
  for consistent performance.



An Example of logging a LangChain RetrievalQA Chain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../../examples/langchain/retrieval_qa_chain.py
    :language: python

The output of the example above is shown below:

.. code-block:: python
    :caption: Output (truncated)

    [" The president said..."]

.. _log-retriever-chain:

Logging and Evaluating a LangChain Retriever in MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `langchain` flavor in MLflow extends its functionalities to include the logging and individual evaluation of retriever objects. This capability is particularly valuable for assessing the quality of documents retrieved by a retriever without needing to process them through a large language model (LLM).

Purpose of Logging Individual Retrievers
""""""""""""""""""""""""""""""""""""""""
- **Independent Evaluation**: Allows for the assessment of a retriever's performance in fetching relevant documents, independent of their subsequent use in LLMs.
- **Quality Assurance**: Facilitates the evaluation of the retriever's effectiveness in sourcing accurate and contextually appropriate documents.

Requirements for Logging Retrievers in MLflow
"""""""""""""""""""""""""""""""""""""""""""""
- **persist_dir**: Specifies where the retriever object is stored.
- **loader_fn**: Details the function used to load the retriever object from its storage location.
- These requirements align with those for logging RetrievalQA chains, ensuring consistency in the process.


An example of logging a LangChain Retriever
"""""""""""""""""""""""""""""""""""""""""""

.. literalinclude:: ../../../../../examples/langchain/retriever_chain.py
    :language: python

The output of the example above is shown below:

.. code-block:: python
    :caption: Output (truncated)

    [
        [
            {
                "page_content": "Tonight. I call...",
                "metadata": {"source": "/state.txt"},
            },
            {
                "page_content": "A former top...",
                "metadata": {"source": "/state.txt"},
            },
        ]
    ]

MLflow Langchain Autologging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please refer to the `MLflow Langchain Autologging <../autologging.html>`_ documentation for more details on how to enable autologging for Langchain models.
