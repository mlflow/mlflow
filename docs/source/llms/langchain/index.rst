MLflow LangChain Flavor
=======================

.. attention::
    The ``langchain`` flavor is under active development and is marked as Experimental. Public APIs are 
    subject to change, and new features may be added as the flavor evolves.

Welcome to the developer guide for the integration of `LangChain <https://www.langchain.com/>`_ with MLflow. This guide serves as a comprehensive 
resource for understanding and leveraging the combined capabilities of LangChain and MLflow in developing advanced language model applications.

`LangChain <https://www.langchain.com/>`_ is a versatile framework designed for building applications powered by language models. It excels in creating context-aware applications 
that utilize language models for reasoning and generating responses, enabling the development of sophisticated NLP applications.

`LangGraph <https://langchain-ai.github.io/langgraph/>`_ is a complementary agent-based framework from the creators of Langchain, supporting the creation of
stateful agent and multi-agent GenAI applications. LangGraph utilizes LangChain in order to interface with GenAI agent components.

Why use MLflow with LangChain?
------------------------------
Aside from the benefits of using MLflow for managing and deploying machine learning models, the integration of LangChain with MLflow provides a number of
benefits that are associated with using LangChain within the broader MLflow ecosystem.

Experiment Tracking
^^^^^^^^^^^^^^^^^^^
LangChain's flexibility in experimenting with various agents, tools, and retrievers becomes even more powerful when paired with `MLflow Tracking <../../tracking.html>`_. This combination allows for rapid experimentation and iteration. You can effortlessly compare runs, making it easier to refine models and accelerate the journey from development to production deployment.

Dependency Management
^^^^^^^^^^^^^^^^^^^^^
Deploy your LangChain application with confidence, leveraging MLflow's ability to `manage and record all external dependencies <../../model/dependencies.html>`_ automatically. This ensures consistency between development and production environments, reducing deployment risks with less manual intervention.

MLflow Evaluate
^^^^^^^^^^^^^^^
`MLflow Evaluate <../llm-evaluate/index.html>`_ provides native capabilities within MLflow to evaluate language models. With this feature you can easily utilize automated evaluation algorithms on the results of your LangChain application's inference results. This capability facilitates the efficient assessment of inference results from your LangChain application, ensuring robust performance analytics.

Observability
^^^^^^^^^^^^^
`MLflow Tracing <../tracing/index.html>`_ is a new feature of MLflow that allows you to trace how data flows through your LangChain chain/agents/etc. This feature provides a visual representation of the data flow, making it easier to understand the behavior of your LangChain application and identify potential bottlenecks or issues. With its powerful `Automatic Tracing <../tracing/index.html#automatic-tracing>`_ capability, you can instrument your LangChain application without any code change but just running ``mlflow.langchain.autolog()`` command once.


Automatic Logging
------------------

Autologging is a powerful one stop solution to achieve all the above benefits with just one line of code ``mlflow.langchain.autolog()``. By enabling autologging, you can automatically log all the components of your LangChain application, including chains, agents, and retrievers, with minimal effort. This feature simplifies the process of tracking and managing your LangChain application, allowing you to focus on developing and improving your models. For more information on how to use this feature, refer to the `MLflow LangChain Autologging Documentation <autologging.html>`_.

.. toctree::
    :maxdepth: 1
    :hidden:

    autologging.rst


Supported Elements in MLflow LangChain Integration
--------------------------------------------------
- `LLMChain <https://python.langchain.com/docs/modules/chains/foundational/llm_chain>`_
- `Agents <https://python.langchain.com/docs/modules/agents/>`_
- `RetrievalQA <https://js.langchain.com/docs/modules/chains/popular/vector_db_qa>`_
- `Retrievers <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_


.. attention::

   Logging chains/agents that include `ChatOpenAI <https://python.langchain.com/docs/integrations/chat/openai>`_ and `AzureChatOpenAI <https://python.langchain.com/docs/integrations/chat/azure_chat_openai>`_ requires ``MLflow>=2.12.0`` and ``LangChain>=0.0.307``.


Overview of Chains, Agents, and Retrievers
------------------------------------------

.. tabs::

    .. tab:: Chain

        Sequences of actions or steps hardcoded in code. Chains in LangChain combine various components like prompts, models, and output parsers to create a flow of processing steps.

        The figure below shows an example of interfacing directly with a SaaS LLM via API calls with no context to the history of the conversation in the top portion. The 
        bottom portion shows the same queries being submitted to a LangChain chain that incorporates a conversation history state such that the entire conversation's history 
        is included with each subsequent input. Preserving conversational context in this manner is key to creating a "chat bot".

        .. figure:: ../../_static/images/tutorials/llms/stateful-chains.png
            :alt: The importance of stateful storage of conversation history for chat applications
            :width: 70%
            :align: center

    .. tab:: Agents

        Dynamic constructs that use language models to choose a sequence of actions. Unlike chains, agents decide the order of actions based on inputs, tools available, and intermediate outcomes.

        .. figure:: ../../_static/images/tutorials/llms/langchain-agents.png
            :alt: Complex LLM queries with LangChain agents
            :width: 80%
            :align: center

    .. tab:: Retrievers

        Components in RetrievalQA chains responsible for sourcing relevant documents or data. Retrievers are key in applications where LLMs need to reference specific external information for accurate responses.

        .. figure:: ../../_static/images/tutorials/llms/langchain-retrievalqa.png
            :alt: MLflow LangChain RetrievalQA architecture
            :width: 70%
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

.. tip::

    If you're encountering issues with logging or saving LangChain components with MLflow, see the `models from code <../../models.html#models-from-code>`_
    feature documentation to determine if logging your model from a script file provides a simpler and more robust logging solution!

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


How can I use a streaming API with LangChain?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Streaming with LangChain Models**: Ensure that the LangChain model supports a streaming response and use an MLflow version >= 2.12.2.

    As of the MLflow 2.12.2 release, LangChain models that support streaming responses that have been saved using MLflow 2.12.2 (or higher) can be loaded and used for 
    streamable inference using the ``predict_stream`` API. Ensure that you are consuming the return type correctly, as the return from these models is a ``Generator`` object.
    To learn more, refer to the `predict_stream guide <https://mlflow.org/docs/latest/models.html#how-to-load-and-score-python-function-models>`_.

How can I log my chain from code?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Models from Code**: MLflow 2.12.2 introduced the ability to log LangChain models directly from a code definition. 

    In order to use this feature, you will utilize the :py:func:`mlflow.models.set_model` API to define the chain that you would like to log as an MLflow model. 
    After having this set within your code that defines your chain, when logging your model, you will specify the **path** to the file that defines your chain. 

    For example, here is a simple chain defined in a file named ``langchain_code_chain.py``:

    .. code-block:: python
        
        import os
        from operator import itemgetter

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableLambda
        from langchain_openai import OpenAI

        import mlflow

        mlflow.set_experiment("Homework Helper")

        mlflow.langchain.autolog()

        prompt = PromptTemplate(
            template="You are a helpful tutor that evaluates my homework assignments and provides suggestions on areas for me to study further."
            " Here is the question: {question} and my answer which I got wrong: {answer}",
            input_variables=["question", "answer"],
        )


        def get_question(input):
            default = "What is your name?"
            if isinstance(input_data[0], dict):
                return input_data[0].get("content").get("question", default)
            return default


        def get_answer(input):
            default = "My name is Bobo"
            if isinstance(input_data[0], dict):
                return input_data[0].get("content").get("answer", default)
            return default


        model = OpenAI(temperature=0.95)

        chain = (
            {
                "question": itemgetter("messages") | RunnableLambda(get_question),
                "answer": itemgetter("messages") | RunnableLambda(get_answer),
            }
            | prompt
            | model
            | StrOutputParser()
        )

        mlflow.models.set_model(chain)

    From a different file (in this case, a Jupyter Notebook), logging the model directly via supplying the path to the file that defines the chain:

    .. code-block:: python

        from pprint import pprint

        import mlflow

        chain_path = "langchain_code_chain.py"

        with mlflow.start_run():
            info = mlflow.langchain.log_model(lc_model=chain_path, artifact_path="chain")

        # Load the model and run inference
        homework_chain = mlflow.langchain.load_model(model_uri=info.model_uri)

        exam_question = {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "question": "What is the primary function of control rods in a nuclear reactor?",
                        "answer": "To stir the primary coolant so that the neutrons are mixed well.",
                    },
                },
            ]
        }

        response = homework_chain.invoke(exam_question)

        pprint(response)
    
    The model will be logged as a script within the MLflow UI:

    .. figure:: ../../_static/images/tutorials/llms/langchain-code-model.png
            :alt: Logging a LangChain model from a code script file
            :width: 100%
            :align: center


How can I log an agent built with LangGraph to MLflow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LangGraph integration with MLflow is designed to utilize the `Models From Code feature <../../model/models-from-code.html>`_
in MLflow to broaden and simplify the support of agent serialization.

To log a LangGraph agent, you can define your agent code within a script, as shown below, saved to a file ``langgraph.py``:

.. code-block:: python

    from typing import Literal

    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    import mlflow


    @tool
    def get_weather(city: Literal["seattle", "sf"]):
        """Use this to get weather information."""
        if city == "seattle":
            return "It's probably raining. Again."
        elif city == "sf":
            return "It's always sunny in sf"


    llm = ChatOpenAI()
    tools = [get_weather]
    graph = create_react_agent(llm, tools)

    # specify the Agent as the model interface to be loaded when executing the script
    mlflow.models.set_model(graph)

When you're ready to log this agent script definition to MLflow, you can refer to 
this saved script directly when defining the model:

.. code-block:: python

    import mlflow

    input_example = {
        "messages": [{"role": "user", "content": "what is the weather in seattle today?"}]
    }

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model="./langgraph.py",  # specify the path to the LangGraph agent script definition
            artifact_path="langgraph",
            input_example=input_example,
        )

When the agent is loaded from MLflow, the script will be executed and the defined agent will be
made available for use for invocation.

The agent can be loaded and used for inference as follows:

.. code-block:: python

    agent = mlflow.langchain.load_model(model_info.model_uri)
    query = {
        "messages": [
            {
                "role": "user",
                "content": "Should I bring an umbrella today when I go to work in San Francisco?",
            }
        ]
    }
    agent.invoke(query)
