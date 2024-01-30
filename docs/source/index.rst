MLflow: A Tool for Managing the Machine Learning Lifecycle
==========================================================

MLflow IS an open-source platform, purpose-built to assist machine learning practitioners and teams in
handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for
machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.


In each of the sections below, you will find overviews, guides, and step-by-step tutorials to walk you through 
the features of MLflow and how they can be leveraged to solve real-world MLOps problems. 


`Getting Started with MLflow <getting-started/index.html>`_
-----------------------------------------------------------

If this is your first time exploring MLflow, the tutorials and guides here are a great place to start. The emphasis in each of these is 
getting you up to speed as quickly as possible with the basic functionality, terms, APIs, and general best practices of using MLflow in order to 
enhance your learning in area-specific guides and tutorials. 

Getting Started Guides and Quickstarts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="getting-started/intro-quickstart/index.html" >
                    <div class="header">
                        MLflow Tracking Quickstart
                    </div>
                    <p>
                    A great place to start to learn the fundamentals of MLflow Tracking! Learn in 5 minutes how to log, register, and load a model for inference. 
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="getting-started/logging-first-model/index.html" >
                    <div class="header">
                        Intro to MLflow Tutorial
                    </div>
                    <p>
                        Learn how to get started with the basics of MLflow in a step-by-step instructional tutorial that shows the critical 
                        path to logging your first model
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="tracking/autolog.html" >
                    <div class="header">
                        Autologging Quickstart
                    </div>
                    <p>
                        Short on time? This is a no-frills quickstart that shows how to leverage autologging during training and how to 
                        load a model for inference
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="getting-started/quickstart-2/index.html" >
                    <div class="header">
                        Deployment Quickstart
                    </div>
                    <p>
                        Learn the basics of registering a model, setting up local serving for validation, and the process of 
                        containerization of a model for remote serving
                    </p>
                </a>
            </div>
        </article>
    </section>

`LLMs <llms/index.html>`_
-------------------------

Explore the comprehensive LLM-focused native support in MLflow. From **MLflow Deployments for LLMs** to the **Prompt Engineering UI** and native LLM-focused MLflow flavors like 
**open-ai**, **transformers**, and **sentence-transformers**, the tutorials and guides here will help to get you started in leveraging the 
benefits of these powerful natural language deep learning models.  
You'll learn how MLflow simplifies both using LLMs and developing solutions that leverage LLMs. Important tasks such as prompt development, evaluation of prompts, comparison of  
foundation models, fine-tuning and logging LLMs, and setting up production-grade interface servers are all covered by MLflow. 

Explore the guides and tutorials below to start your journey!

LLM Guides and Tutorials
^^^^^^^^^^^^^^^^^^^^^^^^

MLflow Native flavors for GenAI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="llms/transformers/index.html" >
                    <div class="header">
                        🤗 Transformers in MLflow
                    </div>
                    <p>
                        Learn how to leverage the HuggingFace Transformers package with MLflow. Explore multiple tutorials and examples that leverage 
                        the power of state-of-the-art LLMs.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="llms/openai/index.html" >
                    <div class="header">
                        OpenAI in MLflow
                    </div>
                    <p>
                        Learn how to leverage the state-of-the-art LLMs offered by OpenAI directly as MLflow flavors to build and track useful language-based 
                        applications.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="llms/langchain/index.html" >
                    <div class="header">
                        LangChain in MLflow
                    </div>
                    <p>
                        Learn how to build both simple and complex LLM-powered applications with LangChain, using MLflow to simplify deployment, dependency 
                        management, and service integration.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="llms/sentence-transformers/index.html" >
                    <div class="header">
                        Sentence Transformers in MLflow
                    </div>
                    <p>
                        Learn how to leverage the advanced capabilities with semantic sentence embeddings within the Sentence Transformers package, using MLflow to simplify 
                        inference, create custom deployable applications, and more.
                    </p>
                </a>
            </div>
        </article>
    </section>


General GenAI Guides 
~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="llms/prompt-engineering/index.html" >
                    <div class="header">
                        Guide to the MLflow Prompt Engineering UI
                    </div>
                    <p>
                        Explore the features and functions of MLflow's Prompt Engineering UI for development, testing, evaluation, and 
                        deployment of validated prompts for your LLM use cases.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="llms/deployments/index.html" >
                    <div class="header">
                        Guide for the MLflow Deployments for LLMs
                    </div>
                    <p>
                        Learn how to configure, setup, deploy, and use the MLflow Deployments for testing and production use cases of both 
                        SaaS and custom open-source LLMs.
                    </p>
                </a>
            </div>
            
            <div class="simple-card">
                <a href="llms/llm-evaluate/index.html" >
                    <div class="header">
                        Evaluating LLMs with MLflow Guide
                    </div>
                    <p>
                        Learn how to evaluate LLMs and LLM-powered solutions with MLflow Evaluate.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="llms/custom-pyfunc-for-llms/index.html" >
                    <div class="header">
                        Using Custom PyFunc with LLMs
                    </div>
                    <p>
                        Explore the nuances of packaging and deploying advanced LLMs in MLflow using custom PyFuncs. This guide delves deep 
                        into managing intricate model behaviors, ensuring seamless and efficient LLM deployments.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="llms/rag/index.html" >
                    <div class="header">
                        Evaluation for RAG
                    </div>
                    <p>
                        Learn how to evaluate Retrieval Augmented Generation applications by leveraging LLMs to generate a evaluation dataset and evaluate it using the built-in metrics in the MLflow Evaluate API.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="llms/llm-tracking/index.html" >
                    <div class="header">
                        LLM Tracking with MLflow
                    </div>
                    <p>
                        Dive into the intricacies of MLflow's LLM Tracking system. From capturing prompts to monitoring generated outputs, 
                        discover how MLflow provides a holistic solution for managing LLM interactions.
                    </p>
                </a>
            </div>
        </article>
    </section>


`Model Evaluation <model-evaluation/index.html>`_
-------------------------------------------------

Dive into MLflow's robust framework for evaluating the performance of your ML models. 

With support for traditional ML evaluation (classification and regression tasks), as well as support for evaluating large language models (LLMs), 
this suite of APIs offers a simple but powerful automated approach to evaluating the quality of the model development work that you're doing.

In particular, for LLM evaluation, the `mlflow.evaluate()` API allows you to validate not only models, but providers and prompts. 
By leveraging your own datasets and using the provided default evaluation criteria for tasks such as text summarization and question answering, you can 
get reliable metrics that allow you to focus on improving the quality of your solution, rather than spending time writing scoring code.

Visual insights are also available through the MLflow UI, showcasing logged outputs, auto-generated plots, and model comparison artifacts. 



`Deep Learning <deep-learning/index.html>`_
-------------------------------------------

See how MLflow can help manage the full lifecycle of your Deep Learning projects. Whether you're using frameworks like 
**TensorFlow (tensorflow)**, **Keras (keras)**, **PyTorch (torch)**, **Fastai (fastai)**, or **spaCy (spacy)**, MLflow offers first-class support, 
ensuring seamless integration and deployment. Additionally, generic packaging frameworks that have native MLflow integration such as **ONNX (onnx)** 
grealy help to simplify the deployment of deep learning models to a wide variety of deployment providers and environments. 

Paired with MLflow's streamlined APIs and comparative UI, you are equipped with everything needed to manage, track, and optimize your deep learning workflows.



`Traditional ML <traditional-ml/index.html>`_
---------------------------------------------

Leverage the power of MLflow for all your Traditional Machine Learning needs. Whether you're working with supervised, unsupervised, statistical, or time series data, 
MLflow streamlines the process by providing an integrated environment that supports a large array of widely-used libraries like **Scikit-learn (sklearn)**, 
**SparkML (spark)**, **XGBoost (xgboost)**, **LightGBM (lightgbm)**, **CatBoost (catboost)**, **Statsmodels**, **Prophet**, and **Pmdarima**. 
With MLflow, you not only get APIs tailored for these libraries but also a user-friendly UI to compare various runs, ensuring that your model tuning and 
evaluation phases are both efficient and insightful.

Traditional ML Guides and Tutorials 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="traditional-ml/hyperparameter-tuning-with-child-runs/index.html" >
                    <div class="header">
                        Hyperparameter Tuning with Optuna and MLflow
                    </div>
                    <p>
                        This in-depth guide will show you how to leverage some core functionality in MLflow to keep your tuning iterations organized and 
                        searchable, all while covering a number of features within MLflow that cater to the needs of this common activity.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="traditional-ml/creating-custom-pyfunc/index.html" >
                    <div class="header">
                        Custom PyFunc Tutorials with MLflow
                    </div>
                    <p>
                        Dive into the foundational aspects of MLflow's custom `pyfunc` to encapsulate, manage, and invoke models from any framework. 
                        This guide elucidates the versatility of `pyfunc`, highlighting how it bridges the gap between supported named flavors and bespoke model requirements.
                    </p>
                </a>
            </div>
        </article>
    </section>


`Deployment <deployment/index.html>`_
-------------------------------------

In today's ML-driven landscape, the ability to deploy models seamlessly and reliably is crucial. MLflow offers a robust suite tailored for this 
very purpose, ensuring that models transition from development to production without a hitch. Whether you're aiming for real-time predictions, 
batch analyses, or interactive insights, MLflow's deployment capabilities have got you covered. From managing dependencies and packaging models 
with their associated code to offering a large ecosystem of deployment avenues like local servers, cloud platforms, or Kubernetes clusters, 
MLflow ensures your models are not just artifacts but actionable, decision-making tools. 

Dive deep into the platform's offerings, explore the tutorials, and harness the power of MLflow for efficient model serving.

Deployment Guides and Tutorials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="deployment/deploy-model-to-kubernetes/index.html" >
                    <div class="header">
                        Deploying a Model to Kubernetes with MLflow
                    </div>
                    <p>
                        This guide showcases the seamless end-to-end process of training a linear regression model, packaging it in a reproducible format, 
                        and deploying to a Kubernetes cluster using MLflow. Explore how MLflow simplifies model deployment to production environments.
                    </p>
                </a>
            </div>
        </article>
    </section>




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
