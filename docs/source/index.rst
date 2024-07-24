MLflow: A Tool for Managing the Machine Learning Lifecycle
==========================================================

MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in
handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for
machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.


MLflow Getting Started Resources
--------------------------------

If this is your first time exploring MLflow, the tutorials and guides here are a great place to start. The emphasis in each of these is 
getting you up to speed as quickly as possible with the basic functionality, terms, APIs, and general best practices of using MLflow in order to 
enhance your learning in area-specific guides and tutorials. 

.. |getting-started| raw:: html

    <div class="main-container">
        <h3>Learn about the core components of MLflow</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Quickstarts</h4>
                <p>
                    Get Started with MLflow in our <a href="getting-started/intro-quickstart/index.html">5-minute tutorial</a>
                </p> 
                <h4>Guides</h4>
                <p>
                    Learn the core components of MLflow with <a href="getting-started/logging-first-model/index.html">this in-depth guide to Tracking</a>
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/learn-core-components.png" alt="Core Components">
            </div>
        </div>
    </div>

.. |starting-guides| raw:: html

    <div class="main-container">
        <h3>Learn how to perform common tasks in MLflow</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Guides</h4>
                <p>
                    <a href="tracking/autolog.html">Autologging tutorial</a> for effortless model tracking
                </p>
                <p>
                    <a href="model/signatures.html">Model Signatures</a> and type validation in MLflow
                </p>
                <p>
                    <a href="getting-started/quickstart-2/index.html">Model Deployment Quickstart</a>
                </p>
                <p>
                    <a href="traditional-ml/hyperparameter-tuning-with-child-runs/index.html">Hyperparameter tuning</a> with MLflow
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/model-dev-lifecycle.png" alt="Model Development Lifecycle">
            </div>
        </div>
    </div>

.. |model-topics| raw:: html

    <div class="main-container">
        <h3>Learn about MLflow Model-related topics</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Guides</h4>
                <p>
                    Introduction to <a href="traditional-ml/creating-custom-pyfunc/index.html">Custom Python Models</a>
                </p>
                <p>
                    <a href="model/dependencies.html">Model dependency management</a> in MLflow
                </p>
                <p>
                    <a href="model/signatures.html">Model Signatures</a> and type validation
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/model-topics.png" alt="MLflow Model Topics">
            </div>
        </div>
    </div>

.. |genai-quickstarts| raw:: html

    <div class="main-container">
        <h3>Get started with MLflow's GenAI integrations</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Quickstarts</h4>
                <p>
                    <a href="llms/transformers/tutorials/text-generation/text-generation.html">Transformers</a> Text Generation
                </p>
                <p>
                    <a href="llms/langchain/notebooks/langchain-quickstart.html">LangChain</a> Introductory Tutorial
                </p>
                <p>
                    <a href="llms/sentence-transformers/tutorials/quickstart/sentence-transformers-quickstart.html">Sentence Transformers</a> Basic Embedding Tutorial
                </p>
                <p>
                    <a href="llms/openai/notebooks/openai-quickstart.html">OpenAI</a> Quickstart Tutorial
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/genai-integrations.png" alt="GenAI with MLflow">
            </div>
        </div>
    </div>

.. |dl-quickstarts| raw:: html

    <div class="main-container">
        <h3>Get started with MLflow's Deep Learning Library integrations</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Quickstarts</h4>
                <p>
                    <a href="deep-learning/tensorflow/quickstart/quickstart_tensorflow.html">TensorFlow</a>
                </p>
                <p>
                    <a href="deep-learning/pytorch/quickstart/pytorch_quickstart.html">PyTorch</a>
                </p>
                <p>
                    <a href="deep-learning/keras/quickstart/quickstart_keras.html">Keras</a>
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/mlflow-deep-learning.png" alt="Deep Learning with MLflow">
            </div>
        </div>
    </div>
    

.. container:: intro

    .. tabs::

        .. tab:: Learn about MLflow

            |getting-started|
        
        .. tab:: MLflow Basics

            |starting-guides|
        
        .. tab:: MLflow Models Introduction

            |model-topics|
        
        .. tab:: GenAI Quickstarts

            |genai-quickstarts|
        
        .. tab:: Deep Learning Quickstarts

            |dl-quickstarts|


GenAI and MLflow
----------------

Explore the comprehensive GenAI-focused support in MLflow. From **MLflow Deployments for GenAI models** to the **Prompt Engineering UI** and native GenAI-focused MLflow flavors like 
**open-ai**, **transformers**, and **sentence-transformers**, the tutorials and guides here will help to get you started in leveraging the 
benefits of these powerful models, services, and applications.  
You'll learn how MLflow simplifies both using GenAI models and developing solutions that leverage them. Important tasks such as prompt development, evaluation of prompts, comparison of  
foundation models, fine-tuning, logging, and deploying production-grade inference servers are all covered by MLflow. 

Explore the guides and tutorials below to start your journey!

.. |genai-flavors| raw:: html

    <div class="main-container">
        <h3>Explore the Native MLflow GenAI Integrations</h3>
        <div class="icon-container">
            <div class="icon-box">
                <a href="llms/transformers/index.html">
                    <img src="_static/images/logos/huggingface-logo.svg" alt="Hugging Face Transformers"/>
                </a>
                <p>Transformers</p>
            </div>
            <div class="icon-box">
                <a href="llms/openai/index.html">
                    <img src="_static/images/logos/openai-logo.png" alt="OpenAI"/>
                </a>
                <p>OpenAI</p>
            </div>
            <div class="icon-box">
                <a href="llms/langchain/index.html">
                    <img src="_static/images/logos/langchain-logo.png" alt="LangChain"/>
                </a>
                <p>LangChain</p>
            </div>
            <div class="icon-box">
                <a href="llms/llama-index/index.html">
                    <img src="_static/images/logos/llamaindex-logo.svg" alt="LlamaIndex"/>
                </a>
                <p>LlamaIndex</p>
            </div>
            <div class="icon-box">
                <a href="llms/sentence-transformers/index.html">
                    <img src="_static/images/logos/sentence-transformers-logo.png" alt="Sentence Transformers"/>
                </a>
                <p>Sentence Transformers</p>
            </div>
        </div>
    </div>

.. |tracing| raw:: html
    
    <div class="main-container">
        <h3>Learn about how to instrument your GenAI Workloads with MLflow Tracing</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Guides</h4>
                <ul>
                    <li>
                        Learn how to leverage <a href="llms/tracing/index.html">Tracing</a> in MLflow
                    </li>
                    <li>
                        View the <a href="llms/tracing/overview.html">Tracing Guide</a> for more information on tracing
                    </li>
                    <li>
                        Learn how to use MLflow autologging with <a href="llms/openai/autologging.html">OpenAI</a> for automated trace logging
                    </li>
                    <li>
                        Discover the automated <a href="llms/langchain/autologging.html">LangChain trace logging</a> with MLflow autologging
                    </li>
                </ul>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/tracing-ui.gif" alt="MLflow Tracing">
            </div>
        </div>
    </div>

.. |prompt-engineering-ui| raw:: html

    <div class="main-container">
        <h3>Explore the Prompt Engineering UI</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Quickstarts</h4>
                <p>
                    Learn how to use the <a href="llms/prompt-engineering/index.html">Prompt Engineering UI</a>
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/prompt-engineering-ui.png" alt="Prompt Engineering UI">
            </div>
        </div>
    </div>


.. |deployments-server| raw:: html
    
    <div class="main-container">
        <h3>Learn about managed access to GenAI services with the MLflow Deployments Server</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Guides</h4>
                <p>
                    Learn how to use the <a href="llms/deployments/guides/index.html">MLflow Deployments Server</a>
                </p>
                <p>
                    View the <a href="llms/deployments/index.html">in-depth Guide for the MLflow Deployments Server</a>
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/deployments-server.png" alt="MLflow Deployments Server">
            </div>
        </div>
    </div>

.. |llm-evaluate| raw:: html

    <div class="main-container">
        <h3>Learn about GenAI Evaluation</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Guides</h4>
                <p>
                    Learn how to <a href="llms/llm-evaluate/index.html">evaluate your GenAI applications</a> with MLflow
                </p>
                <p>
                    Discover how to use <a href="llms/prompt-engineering/index.html#step-10-evaluate-the-new-prompt-template-on-previous-inputs">MLflow Evaluate</a> with the Prompt Engineering UI
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/evaluate.png" alt="MLflow GenAI Evaluation">
            </div>
        </div>
    </div>



.. |rag| raw:: html

    <div class="main-container">
        <h3>Learn about using Retrieval Augmented Generation (RAG) with MLflow</h3>
        <div class="sub-container-two-columns">
            <div class="text-box">
                <h4>Guides</h4>
                <p>
                    Learn how to <a href="llms/rag/index.html">work with RAG systems</a> in MLflow
                </p>
                <p>
                    View the hands-on <a href="llms/langchain/notebooks/langchain-retriever.html">LangChain RAG Guide</a>
                </p>
            </div>
            <div class="image-box">
                <img src="_static/images/intro/rag.png" alt="MLflow RAG">
            </div>
        </div>
    </div>




.. container:: genai

    .. tabs::

        .. tab:: GenAI Integrations

            |genai-flavors|
        
        .. tab:: Tracing

            |tracing|
        
        .. tab:: Prompt Engineering UI

            |prompt-engineering-ui|

        .. tab:: Deployments Server

            |deployments-server|
        
        .. tab:: GenAI Evaluation

            |llm-evaluate|
        
        .. tab:: RAG

            |rag|


.. toctree::
    :maxdepth: 1
    :hidden:

    introduction/index
    getting-started/index
    new-features/index
    llms/index
    model-evaluation/index
    deep-learning/index
    traditional-ml/index
    deployment/index
    tracking
    system-metrics/index
    projects
    models
    model-registry
    recipes
    plugins
    auth/index
    cli
    search-runs
    search-experiments
    python_api/index
    R-api
    java_api/index
    rest-api
    docker
    community-model-flavors
    tutorials-and-examples/index
