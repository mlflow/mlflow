LLMs
====

LLMs, or Large Language Models, have rapidly become a cornerstone in the machine learning domain, offering 
immense capabilities ranging from natural language understanding to code generation and more. 
However, harnessing the full potential of LLMs often involves intricate processes, from interfacing with 
multiple providers to fine-tuning specific models to achieve desired outcomes. 

Such complexities can easily become a bottleneck for developers and data scientists aiming to integrate LLM 
capabilities into their applications.

**MLflow's Support for LLMs** aims to alleviate these challenges by introducing a suite of features and tools designed with the end-user in mind:

`MLflow Tracing <tracing/index.html>`_
--------------------------------------

.. note::
    MLflow Tracing is currently in **Experimental Status** and is subject to change without deprecation warning or notification. 

MLflow offers comprehensive tracing capabilities to monitor and analyze the execution of GenAI applications. This includes automated tracing GenAI frameworks such as
LangChain, OpenAI, LlamaIndex, manual trace instrumentation using high-level fluent APIs, and low-level client APIs for fine-grained control. This functionality 
allows you to capture detailed trace data, enabling better debugging, performance monitoring, and insights into complex workflows. 
Whether through decorators, context managers, or explicit API calls, MLflow provides the flexibility needed to trace and optimize the operations 
of your GenAI models and retain your traced data within the tracking server for further analysis.

- `Automated tracing with GenAI libraries <tracing/index.html#automatic-tracing>`_: Seamless integration with libraries such as LangChain, OpenAI, LlamaIndex, and AutoGen, for automatic trace data collection.
- `Manual trace instrumentation with high-level fluent APIs <tracing/index.html#tracing-fluent-apis>`_: Easy-to-use decorators and context managers for adding tracing with minimal code changes.
- `Low-level client APIs for tracing <tracing/index.html#tracing-client-apis>`_: Thread-safe methods for detailed and explicit control over trace data management.

.. toctree::
    :maxdepth: 1
    :hidden:

    tracing/index
    tracing/overview
    tracing/tracing-schema

To learn more about what tracing is, see our `Tracing Concepts Overview <tracing/overview.html>`_ guide. For an in-depth exploration into the structure of 
MLflow traces and their schema, see the `Tracing Schema <tracing/tracing-schema.html>`_ guide.

`MLflow Deployments Server for LLMs <deployments/index.html>`_
--------------------------------------------------------------

.. toctree::
    :maxdepth: 1
    :hidden:

    deployments/index

Serving as a unified interface, the `MLflow Deployments Server <deployments/index.html>`_ (previously known as "MLflow AI Gateway") 
simplifies interactions with multiple LLM providers. In addition to supporting the most popular SaaS LLM providers, the MLflow Deployments Server 
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


Explore the Native Providers of the MLflow Deployments Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MLflow Deployments Server supports a large range of foundational models from popular SaaS model vendors, as well as providing a means of self-hosting your 
own open source model via an integration with MLflow model serving. 

Please refer to `Supported Provider Models <deployments/index.html#providers>`_ for the full list of supported providers and models.

If you're interested in learning about how to set up the MLflow Deployments Server for a specific provider, follow the links below for our up-to-date 
documentation on GitHub. Each link will take you to a README file that will explain how to set up a route for the provider. In the same directory as 
the README, you will find a runnable example of how to query the routes that the example creates, providing you with a quick reference for getting started 
with your favorite provider!

.. raw:: html

    <section>
        <div class="logo-grid">
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/openai/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/openai-logo.png" alt="OpenAI Logo"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/mosaicml/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/mosaicml-logo.svg" alt="MosaicML Logo"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/openai/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/anthropic-logo.svg" alt="Anthropic Logo"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/cohere/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/cohere-logo.png" alt="Cohere Logo"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/mlflow_models/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/mlflow-logo.svg" alt="MLflow Logo"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/bedrock/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/aws-logo.svg" alt="AWS BedLock Logo" style="max-height: 3rem;"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/palm/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/PaLM-logo.png" alt="PaLM Logo"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/ai21_labs/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/ai21labs-logo.svg" alt="ai21Labs Logo"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/azure_openai/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/azure-ml-logo.png" alt="Azure OpenAI Logo"/>
                </div>
            </a>
            <a href="https://github.com/mlflow/mlflow/blob/master/examples/deployments/deployments_server/huggingface/README.md">
                <div class="logo-card">
                    <img src="../_static/images/logos/huggingface-logo.svg" alt="Hugging Face Logo"/>
                </div>
            </a>
        </div>
    </section>

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

- **Native Support for Popular Packages**: Standardized interfaces for tasks like saving, logging, and managing inference configurations.

- **PyFunc Compatibility**: 

  - Load models as PyFuncs for broad compatibility across serving infrastructures.
  - Strengthens the MLOps process for LLMs, ensuring smooth deployments.
  - Utilize the `Models From Code feature <../model/models-from-code.html>`_ for simplified GenAI application development.

- **Cohesive Ecosystem**: 

  - All essential tools and functionalities consolidated under MLflow.
  - Focus on deriving value from LLMs without getting bogged down by interfacing and optimization intricacies.

.. toctree::
    :maxdepth: 1
    :hidden:

    transformers/index
    openai/index
    sentence-transformers/index
    langchain/index
    llama-index/index

Explore the Native LLM Flavors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Select the integration below to read the documentation on how to leverage MLflow's native integration with these popular libraries:

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="transformers/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/huggingface-logo.svg" alt="HuggingFace Logo"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Transformers 🤗 library and see example notebooks that leverage 
                        MLflow and Transformers to build Open-Source LLM powered solutions.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="openai/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/openai-logo.png" alt="OpenAI Logo" style="width: 80%"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the OpenAI SDK and see example notebooks that leverage 
                        MLflow and OpenAI's advanced LLMs to build interesting and fun applications.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="sentence-transformers/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/sentence-transformers-logo.png"  alt="Sentence Transformers Logo" style="width: 70%"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with the Sentence Transformers library and see example notebooks that leverage 
                        MLflow and Sentence Transformers to perform operations with encoded text such as semantic search, text similarity, and information retrieval.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="langchain/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/langchain-logo.png" alt="LangChain Logo" style="width: 90%"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with LangChain and see example notebooks that leverage 
                        MLflow and LangChain to build LLM-backed applications.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="llama-index/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/llamaindex-logo.svg" alt="LlamaIndex Logo" style="width: 90%"/>
                    </div>
                    <p>
                        Learn about MLflow's native integration with LlamaIndex and see example notebooks that leverage 
                        MLflow and LlamaIndex to build advanced QA systems, chatbots, and other AI-driven applications.
                    </p>
                </a>
            </div>
        </article>
    </section>


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

Note that there are additional tutorials within the `"Explore the Native LLM Flavors" section above <#explore-the-native-llm-flavors>`_, so be sure to check those out as well!

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
                        Learn how to evaluate LLMs with various metrics including LLM-as-a-Judge using <code>mlflow.evaluate()</code> API.
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
                <a href="rag/notebooks/index.html" >
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
