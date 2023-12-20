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
- **LLMChain Logging**: Integrate and log language model chains that interface with both SaaS and self-managed LLM models, for streamlined management and deployment.
- **Agent Logging**: Log complex LangChain agents, allowing language models to dynamically determine action sequences.
- **RetrievalQA Chain Logging**: Seamlessly manage RetrievalQA chains (RAG), combining retrieval capabilities with question-answering.
- **Retriever Logging**: Independently log and evaluate retrievers, assessing the quality of documents retrieved without LLM processing.

Capabilities of LangChain and MLflow
------------------------------------
- **Efficient Development**: Streamline the development of NLP applications with LangChain's modular components and MLflow's robust tracking features.
- **Flexible Integration**: Leverage the versatility of LangChain within the MLflow ecosystem for a range of NLP tasks, from simple text generation to complex data retrieval and analysis.
- **Advanced Functionality**: Utilize LangChain's advanced features like context-aware reasoning and dynamic action selection in agents, all within MLflow's scalable platform.

Overview of Chains, Agents, and Retrievers
------------------------------------------
- **Chains**: Sequences of actions or steps hardcoded in code. Chains in LangChain combine various components like prompts, models, stateful memory, and output parsers to create a flow of processing steps.

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

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/langchain/notebooks/langchain-quickstart.ipynb" class="notebook-download-btn">Download the Introductory Notebook</a><br>


Advanced Tutorials
^^^^^^^^^^^^^^^^^^

In these tutorials, you can learn about more complex usages of LangChain with MLflow. It is highly advised to read through the introductory tutorial prior to 
exploring these more advanced use cases. 

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="notebooks/langchain-agent.html">
                    <div class="header">
                        LangChain Agents
                    </div>
                    <p>
                        Learn how to build a LangChain agent that can query a web search engine and perform calculations based on complex questions using MLflow.
                    </p>
                </a>
            </div>
        </article>
    </section>

Download the Advanced Tutorial Notebooks
----------------------------------------

To download the advanced LangChain tutorial notebooks to run in your environment, click the respective links below:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/langchain/notebooks/langchain-agent.ipynb" class="notebook-download-btn">Download the LangChain Agents Notebook</a><br>
    

.. toctree::
    :maxdepth: 2
    :hidden:

    notebooks/langchain-quickstart.ipynb
    notebooks/langchain-agent.ipynb

`Detailed Documentation <guide/index.html>`_
--------------------------------------------

To learn more about the details of the MLflow LangChain flavor, read the detailed guide below.

.. raw:: html

    <a href="guide/index.html" class="download-btn">View the Comprehensive Guide</a>

.. toctree::
    :maxdepth: 1
    :hidden:

    guide/index.rst
