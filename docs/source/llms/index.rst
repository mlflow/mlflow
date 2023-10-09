LLMs
====

LLMs, or Large Language Models, have rapidly become a cornerstone in the machine learning domain, offering 
immense capabilities ranging from natural language understanding to code generation and more. 
However, harnessing the full potential of LLMs often involves intricate processes, from interfacing with 
multiple providers to fine-tuning specific models to achieve desired outcomes. 

Such complexities can easily become a bottleneck for developers and data scientists aiming to integrate LLM 
capabilities into their applications.

**MLflow's Support for LLMs** aims to alleviate these challenges by introducing a suite of features and tools designed with the end-user in mind:

.. raw:: html

    <div class="redirect-header">
        <a href="gateway/index.html">⛩️ MLflow AI Gateway ⛩️</a>
    </div>


Serving as a unified interface, the MLflow AI Gateway simplifies interactions with multiple LLM providers, such as OpenAI, MosaicML, Cohere, and Anthropic. 
In addition to supporting the most popular SaaS LLM providers, the AI Gateway provides an integration to MLflow model serving, allowing you to serve your 
own LLM or a fine-tuned foundation model within your own serving infrastructure.

By offering a centralized endpoint, it negates the need for developers to juggle between different provider APIs and buidling complex integrations for each. 
Moreover, the AI Gateway brings in robustness in credential management. Instead of scattering sensitive API keys across multiple services, giving them to users 
to manage, or hardcoding them into production deployment services, the gateway centralizes these credentials, reinforcing security and simplifying key management.

.. raw:: html

    <section>
        <div class="logo-scroller">
            <div class="logo-box">
                <a href="gateway/index.html#providers">
                    <img src="../_static/images/logos/openai-logo.png" alt="OpenAI Logo" class="logo">
                </a>
            </div>
            <div class="logo-box">
                <a href="gateway/index.html#providers">
                    <img src="../_static/images/logos/mosaicml-logo.svg" alt="MosaicML Logo" class="logo">
                </a>
            </div>
            <div class="logo-box">
                <a href="gateway/index.html#providers">
                    <img src="../_static/images/logos/anthropic-logo.svg" alt="Anthropic Logo" class="logo">
                </a>
            </div>
            <div class="logo-box">
                <a href="gateway/index.html#providers">
                    <img src="../_static/images/logos/cohere-logo.png" alt="Cohere Logo" class="logo">
                </a>
            </div>
            <div class="logo-box">
                <a href="gateway/index.html#providers">
                    <img src="../_static/images/logos/mlflow-logo.svg" alt="Mlflow Logo" class="logo">
                </a>
            </div>
        </div>
    </section>

.. toctree::
    :maxdepth: 1
    :hidden:

    gateway/index

.. raw:: html

    <div class="redirect-header">
        <a href="prompt-engineering/index.html">Prompt Engineering UI</a>
    </div>

.. note::
    The Prompt Engineering UI is in an **"Experimental"** state. The features, UI, and integrations are subject to change without notice or deprecation warnings.

Effective utilization of LLMs often hinges on crafting the right prompts. 
The development of a high-quality prompt is an iterative process of trial and error, where subsequent experimentation is not guaranteed to 
result in cumulative quality improvements. With the volume and speed of iteration through prompt experimentation, it can quickly become very 
overwhelming to remember or keep a history of the state of different prompts that were tried.

In order to solve this difficult problem, MLflow has included a new UI-based feature.

The **Prompt Engineering UI** provides a comprehensive environment to prototype, iterate, and refine these prompts 
without diving deep into code. By making prompt engineering more accessible, users can quickly experiment and hone in on 
optimal model configurations for tasks like question answering or document summarization. Furthermore, each model iteration 
and its configurations are meticulously tracked, ensuring reproducibility and transparency.

.. toctree::
    :maxdepth: 1
    :hidden:

    prompt-engineering/index

.. raw:: html

    <div class="redirect-header">
        <a href="llm-tracking/index.html">LLM Tracking in MLflow</a>
    </div>

Harnessing the true potential of Large Language Models (LLMs) mandates robust tracking and management of every interaction. 
MLflow's LLM Tracking system emerges as a sophisticated solution, meticulously designed to capture, monitor, and analyze these interactions.

Building upon MLflow's foundational capabilities, the LLM Tracking system integrates unique features tailor-made for LLMs. 
From logging prompts – the foundational queries directed towards an LLM – to tracking the dynamic data these models generate, 
MLflow ensures comprehensive coverage. The inclusion of 'predictions' as a core entity, alongside the established 
artifacts, parameters, and metrics, offers an unparalleled depth of insight into the behavior and performance of text-generating models.

With this extensive tracking framework, MLflow endeavors to bring clarity, repeatability, and deep insight into the 
realm of LLMs, facilitating better decision-making and optimization in their deployment and utilization.

.. toctree::
    :maxdepth: 1
    :hidden:

    llm-tracking/index

Native MLflow Flavors for LLMs
------------------------------

To streamline the development, management, and deployment of LLMs, MLflow offers native support for popular packages like transformers, 
sentence-transformers, open-ai, and langchain. These flavors provide standardized interfaces for tasks such as saving, 
logging, managing inference configurations, and more. 

Notably, the ability to load models as PyFuncs ensures compatibility across a wide range of serving infrastructures, 
fortifying the MLOps process for LLMs.

By consolidating these tools and functionalities under one umbrella, MLflow offers a cohesive ecosystem for users to seamlessly integrate, 
manage, and deploy LLMs, ensuring that they can focus more on deriving value from the models and less on the intricacies of interfacing and optimization.

.. raw:: html

    <section>
        <div class="logo-scroller">
            <div class="logo-box">
                <a href="../models.html#transformers-transformers-experimental">
                    <img src="../_static/images/logos/huggingface-logo.svg" alt="HuggingFace Logo" class="logo">
                </a>
            </div>
            <div class="logo-box">
                <a href="../models.html#sentencetransformers-sentence-transformers-experimental">
                    <img src="../_static/images/logos/sentence-transformers-logo.png" alt="Sentence Transformers Logo" class="logo">
                </a>
            </div>
            <div class="logo-box">
                <a href="../models.html#langchain-langchain-experimental">
                    <img src="../_static/images/logos/langchain-logo.png" alt="LangChain Logo" class="logo">
                </a>
            </div>
            <div class="logo-box">
                <a href="../models.html#openai-openai-experimental">
                    <img src="../_static/images/logos/openai-logo.png" alt="OpenAI Logo" class="logo">
                </a>
            </div>
        </div>
    </section>
