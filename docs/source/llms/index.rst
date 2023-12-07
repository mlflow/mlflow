LLMs
====

LLMs, or Large Language Models, have rapidly become a cornerstone in the machine learning domain, offering 
immense capabilities ranging from natural language understanding to code generation and more. 
However, harnessing the full potential of LLMs often involves intricate processes, from interfacing with 
multiple providers to fine-tuning specific models to achieve desired outcomes. 

Such complexities can easily become a bottleneck for developers and data scientists aiming to integrate LLM 
capabilities into their applications.

**MLflow's Support for LLMs** aims to alleviate these challenges by introducing a suite of features and tools designed with the end-user in mind:

`MLflow Deployments Server for LLMs <deployments/index.html>`_
--------------------------------------------------------------

.. toctree::
    :maxdepth: 1
    :hidden:

    deployments/index

Serving as a unified interface, the `MLflow Deployments Server <deployments/index.html>`_ simplifies interactions with multiple LLM providers, such as 
`OpenAI <https://openai.com/>`_, `MosaicML <https://www.mosaicml.com/>`_, `Cohere <https://cohere.com/>`_, `Anthropic <https://www.anthropic.com/>`_, 
`PaLM 2 <https://ai.google/discover/palm2/>`_, `AWS Bedrock <https://aws.amazon.com/bedrock/>`_, and `AI21 Labs <https://www.ai21.com/>`_. 

In addition to supporting the most popular SaaS LLM providers, the MLflow Deployments Server (previously known as "MLflow AI Gateway")
provides an integration to MLflow model serving, allowing you to serve your own LLM or a fine-tuned foundation model within your own serving infrastructure.

.. note:: 
    The MLflow Deployments Server is in active development and has been marked as **Experimental**. 
    APIs may change as this new feature is refined and its functionality is expanded based on feedback.

Benefits of the MLflow Deployments Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Unified Endpoint**: No more juggling between multiple provider APIs.

- **Simplified Integrations**: One-time setup, no repeated complex integrations.

- **Secure Credential Management**: 

  - Centralized storage prevents scattered API keys.
  - No hardcoding or user-handled keys.

- **Consistent API Experience**: 

  - Uniform API across all providers.
  - Easy-to-use REST endpoints and Client API.

- **Seamless Provider Swapping**: 

  - Swap providers without touching your code.
  - Zero downtime provider, model, or route swapping.


Explore the Native Provider integrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MLflow Deployments Server supports a large range of foundational models from popular SaaS model vendors, as well as providing a means of self-hosting your 
own open source model via an integration with MLflow model serving. To learn more about how to get started using the MLflow Deployments Server to simplify the 
configuration and management of your LLM serving needs, select the provider that you're interested in below: 

.. raw:: html

    <section>
        <div class="logo-grid">
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/openai-logo.png" alt="OpenAI Logo"/>
                </div>
            </a>
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/mosaicml-logo.svg" alt="MosaicML Logo"/>
                </div>
            </a>
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/anthropic-logo.svg" alt="Anthropic Logo"/>
                </div>
            </a>
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/cohere-logo.png" alt="Cohere Logo"/>
                </div>
            </a>
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/mlflow-logo.svg" alt="MLflow Logo"/>
                </div>
            </a>
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/aws-logo.svg" alt="AWS Logo" style="max-height: 3rem;"/>
                </div>
            </a>
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/PaLM-logo.png" alt="PaLM Logo"/>
                </div>
            </a>
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/ai21labs-logo.svg" alt="ai21Labs Logo"/>
                </div>
            </a>
            <a href="deployments/index.html#providers">
                <div class="logo-card">
                    <img src="../_static/images/logos/huggingface-logo.svg" alt="Hugging Face Logo"/>
                </div>
            </a>
        </div>
    </section>

`Getting Started Examples for each Provider <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/README.md>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're interested in learning about how to set up the MLflow Deployments Server for a specific provider, follow the links below for our up-to-date 
documentation on GitHub. 

Each link will take you to a README file that will explain how to set up a route for the provider. In the same directory as 
the README, you will find a runnable example of how to query the routes that the example creates, providing you with a quick reference for getting started 
with your favorite provider!

* `OpenAI quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/openai/README.md>`_
* `MosaicML quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/mosaicml/README.md>`_
* `Anthropic quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/anthropic/README.md>`_
* `Cohere quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/cohere/README.md>`_
* `MLflow quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/mlflow_serving/README.md>`_
* `AWS Bedrock quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/bedrock/README.md>`_
* `AI21 Labs quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/ai21labs/README.md>`_ 
* `PaLM 2 quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/palm/README.md>`_
* `Azure OpenAI quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/azure_openai/README.md>`_
* `Hugging Face Text Generation Interface (TGI) quickstart <https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/huggingface/readme.md>`_

.. note::
    The **MLflow** and **Hugging Face TGI** providers are for self-hosted LLM serving of either foundation open-source LLM models, fine-tuned open-source 
    LLM models, or your own custom LLM. The example documentation for these providers will show you how to get started with these, using free-to-use open-source 
    models from the `Hugging Face Hub <https://huggingface.co/docs/hub/index>`_.

`LLM Evaluation <llm-evaluate/index.html>`_
-------------------------------------------

.. toctree::
    :maxdepth: 1
    :hidden:

    llm-evaluate/index

Navigating the vast landscape of Large Language Models (LLMs) can be daunting. Determining the right model, prompt, or service that aligns 
with a project's needs is no small feat. Traditional machine learning evaluation metrics often fall short when it comes to assessing the 
nuanced performance of generative models.

Enter `MLflow LLM Evaluation <llm-evaluate/index.html>`_. This feature is designed to simplify the evaluation process, 
offering a streamlined approach to compare foundational models, providers, and prompts.

Benefits of MLflow's LLM Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Simplified Evaluation**: Navigate the LLM space with ease, ensuring the best fit for your project with standard metrics that can be used to compare generated text.

- **Use-Case Specific Metrics**: Leverage MLflow's :py:func:`mlflow.evaluate` API for a high-level, frictionless evaluation experience.

- **Customizable Metrics**: Beyond the provided metrics, MLflow supports a plugin-style for custom scoring, enhancing the evaluation's flexibility.

- **Comparative Analysis**: Effortlessly compare foundational models, providers, and prompts to make informed decisions.

- **Deep Insights**: Dive into the intricacies of generative models with a comprehensive suite of LLM-relevant metrics.

MLflow's LLM Evaluation is designed to bridge the gap between traditional machine learning evaluation and the unique challenges posed by LLMs.


`Prompt Engineering UI <prompt-engineering/index.html>`_
--------------------------------------------------------

.. toctree::
    :maxdepth: 1
    :hidden:

    prompt-engineering/index

Effective utilization of LLMs often hinges on crafting the right prompts. 
The development of a high-quality prompt is an iterative process of trial and error, where subsequent experimentation is not guaranteed to 
result in cumulative quality improvements. With the volume and speed of iteration through prompt experimentation, it can quickly become very 
overwhelming to remember or keep a history of the state of different prompts that were tried.

Serving as a powerful tool for prompt engineering, the `MLflow Prompt Engineering UI <prompt-engineering/index.html>`_ revolutionizes the 
way developers interact with and refine LLM prompts. 

Benefits of the MLflow Prompt Engineering UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Iterative Development**: Streamlined process for trial and error without the overwhelming complexity.

- **UI-Based Prototyping**: Prototype, iterate, and refine prompts without diving deep into code.

- **Accessible Engineering**: Makes prompt engineering more user-friendly, speeding up experimentation.

- **Optimized Configurations**: Quickly hone in on the best model configurations for tasks like question answering or document summarization.

- **Transparent Tracking**: 

  - Every model iteration and configuration is meticulously tracked.
  - Ensures reproducibility and transparency in your development process.

.. note:: 
    The MLflow Prompt Engineering UI is in active development and has been marked as **Experimental**. 
    Features and interfaces may evolve as feedback is gathered and the tool is refined.


Native MLflow Flavors for LLMs
------------------------------

Harnessing the power of LLMs becomes effortless with flavors designed specifically for working with LLM libraries and frameworks.

Benefits of MLflow's Native Flavors for LLMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Support for Popular Packages**: 

  - Native integration with packages like `transformers <https://huggingface.co/docs/transformers/index>`_, 
    `sentence-transformers <https://www.sbert.net/>`_, `open-ai <https://platform.openai.com/docs/libraries/python-library>`_ , and 
    `langchain <https://www.langchain.com/>`_.
  - Standardized interfaces for tasks like saving, logging, and managing inference configurations.

- **PyFunc Compatibility**: 

  - Load models as PyFuncs for broad compatibility across serving infrastructures.
  - Strengthens the MLOps process for LLMs, ensuring smooth deployments.

- **Cohesive Ecosystem**: 

  - All essential tools and functionalities consolidated under MLflow.
  - Focus on deriving value from LLMs without getting bogged down by interfacing and optimization intricacies.

Explore the Native LLM Flavors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Select the integration below to read the documentation on how to leverage MLflow's native integration with these popular libraries:

.. raw:: html

    <section>
    <div class="logo-grid">
        
        <a href="../models.html#transformers-transformers-experimental">
            <div class="logo-card">
                <img src="../_static/images/logos/huggingface-logo.svg" alt="HuggingFace Logo"/>
            </div>
        </a>
        
        <a href="sentence-transformers/guide/index.html">
            <div class="logo-card">
                <img src="../_static/images/logos/sentence-transformers-logo.png" alt="Sentence Transformers Logo"/>
            </div>
        </a>
        
        <a href="../models.html#langchain-langchain-experimental">
            <div class="logo-card">
                <img src="../_static/images/logos/langchain-logo.png" alt="LangChain Logo"/>
            </div>
        </a>
        
        <a href="../models.html#openai-openai-experimental">
            <div class="logo-card">
                <img src="../_static/images/logos/openai-logo.png" alt="OpenAI Logo"/>
            </div>
        </a>
    </div>
 </section>

Native Integration Guides and Tutorials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="transformers/index.html">
                    <div class="header">
                        🤗 Transformers
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Transformers library and see example notebooks that leverage 
                        MLflow and Transformers to build Open-Source LLM powered solutions.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="sentence-transformers/index.html">
                    <div class="header">
                        Sentence Transformers
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Sentence Transformers library and see example notebooks that leverage 
                        MLflow and Sentence Transformers to perform operations with encoded text such as semantic search, text similarity, and information retrieval.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. toctree::
    :maxdepth: 1
    :hidden:

    transformers/index
    sentence-transformers/index

Native Integration Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you'd like to directly explore code examples for how to get started with using our official library integrations, you can navigate 
directly to our up-to-date examples on GitHub below:

* **langchain**

    * `Logging and using a Chain <https://github.com/mlflow/mlflow/blob/master/examples/langchain/simple_chain.py>`_
    * `Logging and using an Agent <https://github.com/mlflow/mlflow/blob/master/examples/langchain/simple_agent.py>`_
    * `Logging and using a Retriever Chain <https://github.com/mlflow/mlflow/blob/master/examples/langchain/retriever_chain.py>`_ :sup:`1`
    * `Logging and using a Retrieval QA Chain <https://github.com/mlflow/mlflow/blob/master/examples/langchain/retrieval_qa_chain.py>`_ :sup:`1`

:sup:`1` Demonstrates the use of Retrieval Augmented Generation (RAG) using a Vector Store

* **openai**

    * `Using a Completions endpoint <https://github.com/mlflow/mlflow/blob/master/examples/openai/completions.py>`_
    * `Using a Chat endpoint <https://github.com/mlflow/mlflow/blob/master/examples/openai/chat_completions.py>`_
    * `Performing Embeddings Generation <https://github.com/mlflow/mlflow/blob/master/examples/openai/embeddings.py>`_
    * `Using OpenAI on a Spark DataFrame for Batch Processing <https://github.com/mlflow/mlflow/blob/master/examples/openai/spark_udf.py>`_
    * `Using Azure OpenAI <https://github.com/mlflow/mlflow/blob/master/examples/openai/azure_openai.py>`_


`LLM Tracking in MLflow <llm-tracking/index.html>`_
---------------------------------------------------

.. toctree::
    :maxdepth: 1
    :hidden:

    llm-tracking/index

Empowering developers with advanced tracking capabilities, the `MLflow LLM Tracking System <llm-tracking/index.html>`_ stands out as the 
premier solution for managing and analyzing interactions with Large Language Models (LLMs).

Benefits of the MLflow LLM Tracking System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Robust Interaction Management**: Comprehensive tracking of every LLM interaction for maximum insight.

- **Tailor-Made for LLMs**: 

  - Unique features specifically designed for LLMs.
  - From logging prompts to tracking dynamic data, MLflow has it covered.

- **Deep Model Insight**: 

  - Introduces 'predictions' as a core entity, alongside the existing artifacts, parameters, and metrics.
  - Gain unparalleled understanding of text-generating model behavior and performance.

- **Clarity and Repeatability**: 

  - Ensures consistent and transparent tracking across all LLM interactions.
  - Facilitates informed decision-making and optimization in LLM deployment and utilization.


Tutorials and Use Case Guides for LLMs in MLflow
------------------------------------------------

Interested in learning how to leverage MLflow for your LLM projects? 

Look in the tutorials and guides below to learn more about interesting use cases that could help to make your journey into leveraging LLMs a bit easier!

Note that there are additional tutorials within the `Native Integration Guides and Tutorials section above <#native-integration-guides-and-tutorials>`_, so be sure to check those out as well!

.. toctree::
    :maxdepth: 1
    :hidden:
  
    rag/index
    custom-pyfunc-for-llms/index
    llm-evaluate/notebooks/index


.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="llm-evaluate/index.html" >
                    <div class="header">
                        Evaluating LLMs
                    </div>
                    <p>
                        Learn how to evaluate LLMs with MLflow.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="custom-pyfunc-for-llms/index.html" >
                    <div class="header">
                        Using Custom PyFunc with LLMs
                    </div>
                    <p>
                        Explore the nuances of packaging, customizing, and deploying advanced LLMs in MLflow using custom PyFuncs. 
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="rag/index.html" >
                    <div class="header">
                        Evaluation for RAG
                    </div>
                    <p>
                        Learn how to evaluate Retrieval Augmented Generation applications by leveraging LLMs to generate a evaluation dataset and evaluate it using the built-in metrics in the MLflow Evaluate API.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. toctree::
    :maxdepth: 1
    :hidden:

    gateway/index
