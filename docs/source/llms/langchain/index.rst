MLflow LangChain Flavor
=======================

.. attention::
    The ``langchain`` flavor is under active development and is marked as Experimental. Public APIs are 
    subject to change, and new features may be added as the flavor evolves.

Welcome to the developer guide for the integration of `LangChain <https://www.langchain.com/>`_ with MLflow. This guide serves as a comprehensive 
resource for understanding and leveraging the combined capabilities of LangChain and MLflow in developing advanced language model applications.

What is LangChain?
------------------
LangChain is a versatile framework designed for building applications powered by language models. It excels in creating context-aware applications 
that utilize language models for reasoning and generating responses, enabling the development of sophisticated NLP applications.

Supported Elements in MLflow LangChain Integration
--------------------------------------------------
- `LLMChain <https://python.langchain.com/docs/modules/chains/foundational/llm_chain>`_
- `Agents <https://python.langchain.com/docs/modules/agents/>`_
- `RetrievalQA <https://js.langchain.com/docs/modules/chains/popular/vector_db_qa>`_
- `Retrievers <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_


.. attention::

   Logging chains/agents that include `ChatOpenAI <https://python.langchain.com/docs/integrations/chat/openai>`_ and `AzureChatOpenAI <https://python.langchain.com/docs/integrations/chat/azure_chat_openai>`_ requires ``MLflow>=2.12.0`` and ``LangChain>=0.0.307``.


Why use MLflow with LangChain?
------------------------------
Aside from the benefits of using MLflow for managing and deploying machine learning models, the integration of LangChain with MLflow provides a number of 
benefits that are associated with using LangChain within the broader MLflow ecosystem. 

- **MLflow Evaluate**: With the native capabilities within MLflow to evaluate language models, you can easily utilize automated evaluation algorithms on the results of your LangChain application's inference results. This integration facilitates the efficient assessment of inference results from your LangChain application, ensuring robust performance analytics.
- **Simplified Experimentation**: LangChain's flexibility in experimenting with various agents, tools, and retrievers becomes even more powerful when paired with MLflow. This combination allows for rapid experimentation and iteration. You can effortlessly compare runs, making it easier to refine models and accelerate the journey from development to production deployment.
- **Robust Dependency Management**: Deploy your LangChain application with confidence, leveraging MLflow's ability to manage and record all external dependencies. This ensures consistency between development and deployment environments, reducing deployment risks and simplifying the process.

Capabilities of LangChain and MLflow
------------------------------------
- **Efficient Development**: Streamline the development of NLP applications with LangChain's modular components and MLflow's robust tracking features.
- **Flexible Integration**: Leverage the versatility of LangChain within the MLflow ecosystem for a range of NLP tasks, from simple text generation to complex data retrieval and analysis.
- **Advanced Functionality**: Utilize LangChain's advanced features like context-aware reasoning and dynamic action selection in agents, all within MLflow's scalable platform.

Overview of Chains, Agents, and Retrievers
------------------------------------------
- **Chains**: Sequences of actions or steps hardcoded in code. Chains in LangChain combine various components like prompts, models, and output parsers to create a flow of processing steps.

The figure below shows an example of interfacing directly with a SaaS LLM via API calls with no context to the history of the conversation in the top portion. The 
bottom portion shows the same queries being submitted to a LangChain chain that incorporates a conversation history state such that the entire conversation's history 
is included with each subsequent input. Preserving conversational context in this manner is key to creating a "chat bot".

.. figure:: ../../_static/images/tutorials/llms/stateful-chains.png
    :alt: The importance of stateful storage of conversation history for chat applications
    :width: 90%
    :align: center

- **Agents**: Dynamic constructs that use language models to choose a sequence of actions. Unlike chains, agents decide the order of actions based on inputs, tools available, and intermediate outcomes.

.. figure:: ../../_static/images/tutorials/llms/langchain-agents.png
    :alt: Complex LLM queries with LangChain agents
    :width: 90%
    :align: center

- **Retrievers**: Components in RetrievalQA chains responsible for sourcing relevant documents or data. Retrievers are key in applications where LLMs need to reference specific external information for accurate responses.

.. figure:: ../../_static/images/tutorials/llms/langchain-retrievalqa.png
   :alt: MLflow LangChain RetrievalQA architecture
   :width: 80%
   :align: center

Getting Started with the MLflow LangChain Flavor - Tutorials and Guides
-----------------------------------------------------------------------

.. toctree::
    :maxdepth: 2
    :hidden:

    notebooks/langchain-quickstart.ipynb
    notebooks/langchain-retriever.ipynb

Introductory Tutorial
^^^^^^^^^^^^^^^^^^^^^

In this introductory tutorial, you will learn the most fundamental components of LangChain and how to leverage the integration with MLflow to store, retrieve, and 
use a chain. 

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="notebooks/langchain-quickstart.html">
                    <div class="header">
                        LangChain Quickstart
                    </div>
                    <p>
                        Get started with MLflow and LangChain by exploring the simplest possible chain configuration of a prompt and model chained to create 
                        a single-purpose utility application.
                    </p>
                </a>
            </div>
        </article>
    </section>


Advanced Tutorials
^^^^^^^^^^^^^^^^^^

In these tutorials, you can learn about more complex usages of LangChain with MLflow. It is highly advised to read through the introductory tutorial prior to 
exploring these more advanced use cases. 

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="notebooks/langchain-retriever.html">
                    <div class="header">
                        RAG tutorial with LangChain
                    </div>
                    <p>
                        Learn how to build a LangChain RAG with MLflow integration to answer highly specific questions about the legality of business ventures.
                    </p>
                </a>
            </div>
        </article>
    </section>


`Detailed Documentation <guide/index.html>`_
--------------------------------------------

To learn more about the details of the MLflow LangChain flavor, read the detailed guide below.

.. raw:: html

    <a href="guide/index.html" class="download-btn">View the Comprehensive Guide</a>

.. toctree::
    :maxdepth: 1
    :hidden:

    guide/index.rst

FAQ
---

I can't load my chain!
^^^^^^^^^^^^^^^^^^^^^^

- **Allowing for Dangerous Deserialization**: Pickle opt-in logic in LangChain will prevent components from being loaded via MLflow. You might see an error like this:

    .. code-block:: text

        ValueError: This code relies on the pickle module. You will need to set allow_dangerous_deserialization=True if you want to opt-in to 
        allow deserialization of data using pickle. Data can be compromised by a malicious actor if not handled properly to include a malicious 
        payload that when deserialized with pickle can execute arbitrary code on your machine. 

    A change within LangChain that `forces users to opt-in to pickle deserialization <https://github.com/langchain-ai/langchain/pull/18696>`_ can create 
    some issues with loading chains, vector stores, retrievers, and agents that have been logged using MLflow. Because the option is not exposed per component
    to set this argument on the loader function, you will need to ensure that you are setting this option directly within the defined loader function when 
    logging the model. LangChain components that do not set this value will be saved without issue, but a ``ValueError`` will be raised when loading if unset. 

    To fix this, simply re-log your model, specifying the option ``allow_dangerous_deserialization=True`` in your defined loader function. See the tutorial 
    `for LangChain retrievers <notebooks/langchain-retriever.html#Establishing-RetrievalQA-Chain-and-Logging-with-MLflow>`_ for an example of specifying this
    option when logging a ``FAISS`` vector store instance within a ``loader_fn`` declaration.


I can't save my chain, agent, or retriever with MLflow.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Serialization Challenges with Cloudpickle**: Serialization with cloudpickle can encounter limitations depending on the complexity of the objects. 

    Some objects, especially those with intricate internal states or dependencies on external system resources, are not inherently pickleable. This limitation 
    arises because serialization essentially requires converting an object to a byte stream, which can be complex for objects tightly coupled with system states 
    or those having external I/O operations. Try upgrading PyDantic to 2.x version to resolve this issue.

- **Verifying Native Serialization Support**: Ensure that the langchain object (chain, agent, or retriever) is serializable natively using langchain APIs if saving or logging with MLflow doesn't work. 

    Due to their complex structures, not all langchain components are readily serializable. If native serialization 
    is not supported and MLflow doesn't support saving the model, you can file an issue `in the LangChain repository <https://github.com/langchain-ai/langchain/issues>`_ or 
    ask for guidance in the `LangChain Discussions board <https://github.com/langchain-ai/langchain/discussions>`_.

- **Keeping Up with New Features in MLflow**: MLflow might not immediately support the latest LangChain features immediately. 

    If a new feature is not supported in MLflow, consider `filing a feature request on the MLflow GitHub issues page <https://github.com/mlflow/mlflow/issues>`_. 
    With the rapid pace of changes in libraries that are in heavy active development (such as `LangChain's release velocity <https://pypi.org/project/langchain/#history>`_),
    breaking changes, API refactoring, and fundamental functionality support for even existing features can cause integration issues. If there is a chain, agent,
    retriever, or any future structure within LangChain that you'd like to see supported, please let us know!

I'm getting an AttributeError when saving my model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Handling Dependency Installation in LangChain and MLflow**: LangChain and MLflow do not automatically install all dependencies. 

    Other packages that might be required for specific agents, retrievers, or tools may need to be explicitly defined when saving or logging your model. 
    If your model relies on these external component libraries (particularly for tools) that not included in the standard LangChain package, these dependencies 
    will not be automatically logged as part of the model at all times (see below for guidance on how to include them).

- **Declaring Extra Dependencies**: Use the ``extra_pip_requirements`` parameter when saving and logging. 

    When saving or logging your model that contains external dependencies that are not part of the core langchain installation, you will need these additional 
    dependencies. The model flavor contains two options for declaring these dependencies: ``extra_pip_requirements`` and ``pip_requirements``. While specifying 
    ``pip_requirements`` is entirely valid, we recommend using ``extra_pip_requirements`` as it does not rely on defining all of the core dependent packages that 
    are required to use the langchain model for inference (the other core dependencies will be inferred automatically).

I can't load the model logged by mlflow langchain autologging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Model contains langchain retrievers**: LangChain retrievers are not supported by MLflow autologging. 

    If your model contains a retriever, you will need to manually log the model using the ``mlflow.langchain.log_model`` API.
    As loading those models requires specifying `loader_fn` and `persist_dir` parameters, please check examples in 
    `retriever_chain <https://github.com/mlflow/mlflow/blob/master/examples/langchain/retriever_chain.py>`_

- **Can't pickle certain objects**: Try manually logging the model.

    For certain models that LangChain does not support native saving or loading, we will pickle the object when saving it. Due to this functionality, your cloudpickle version must be 
    consistent between the saving and loading environments to ensure that object references resolve properly. For further guarantees of correct object representation, you should ensure that your
    environment has `pydantic` installed with at least version 2. 

How does MLflow langchain autologging interact with callbacks?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Model inference with default callbacks**:

    If you invoke a langchain model with `invoke`, `__call__`, `batch`, `stream` or `get_relevant_documents` (for BaseRetriever) functions directly, MLflow autologging will inject
    a callback into the inference call to collect metrics and artifacts that can be generated from the call chain.

- **Model inference with user-specified callbacks**:

    If your inference call already includes callbacks in the config, e.g. `model.invoke(input, config=RunnableConfig(callbacks=customer_callbacks))`, then MLflow autologging
    still preserves your callbacks and appends a callback after them. `RunnableConfig callbacks parameter <https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig>`_ 
    supports both `BaseCallbackManager` or `List[BaseCallbackHandler]`, in either case MLflow autologging appends a callback to collect metrics and artifacts.
