MLflow: A Tool for Managing the Machine Learning Lifecycle
==========================================================

MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in
handling the complexities of the machine learning process. Mlflow focuses on the full lifecycle for
machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.


In each of the sections below, you will find overviews, guides, and step-by-step tutorials to walk you through 
the features of MLflow and how they can be leveraged to solve real-world MLOps problems. 


`Getting Started with MLflow <getting-started/index.html>`_
-----------------------------------------------------------

If this is your first time exploring MLflow, the tutorials and guides here are a great place to start. The emphasis in each of these is 
getting you up to speed as quickly as possible with the basic functionality, terms, APIs, and general best practices of using MLflow in order to 
enhance your learning in area-specific guides and tutorials. 

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="getting-started/logging-first-model/index.html" >
                    <div class="header">
                        Intro to MLflow Tutorial
                    </div>
                </a>
                <p>
                    Learn how to get started with the basics of MLflow in a step-by-step instructional tutorial that shows the critical 
                    path to logging your first model
                </p>
            </div>
            <div class="simple-card">
                <a href="getting-started/quickstart-1/index.html" >
                    <div class="header">
                        15 minute Tracking Quickstart
                    </div>
                </a>
                <p>
                    Short on time? This is a no-frills quickstart that shows how to leverage autologging during training and how to 
                    load a model for inference
                </p>
            </div>
            <div class="simple-card">
                <a href="getting-started/quickstart-2/index.html" >
                    <div class="header">
                        15 minute Deployment Quickstart
                    </div>
                </a>
                <p>
                    Learn the basics of registering a model, setting up local serving for validation, and the process of 
                    containerization of a model for remote serving
                </p>
            </div>
        </article>
    </section>

`LLMs <llms/index.html>`_
-------------------------

Explore the comprehensive LLM-focused native support in MLflow. From **MLflow AI Gateway** to the **Prompt Engineering UI** and native LLM-focused MLflow flavors like 
**open-ai**, **transformers**, and **sentence-transformers**, the tutorials and guides here will help to get you started in leveraging the 
benefits of these powerful natural language deep learning models.  
You'll learn how MLflow simplifies both using LLMs and developing solutions that leverage LLMs. Important tasks such as prompt development, evaluation of prompts, comparison of  
foundation models, fine-tuning and logging LLMs, and setting up production-grade interface servers are all covered by MLflow. 

Explore the guides and tutorials below to start your journey!

LLM Guides and Tutorials
^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="llms/prompt-engineering/index.html" >
                    <div class="header">
                        Guide to the MLflow Prompt Engineering UI
                    </div>
                </a>
                <p>
                    Explore the features and functions of MLflow's Prompt Engineering UI for development, testing, evaluation, and 
                    deployment of validated prompts for your LLM use cases.
                </p>
            </div>
            <div class="simple-card">
                <a href="llms/gateway/index.html" >
                    <div class="header">
                        Guide for the MLflow AI Gateway
                    </div>
                </a>
                <p>
                    Learn how to configure, setup, deploy, and use the MLflow AI Gateway for testing and production use cases of both 
                    SaaS and custom open-source LLMs.
                </p>
            </div>
            <div class="simple-card">
                <a href="llms/llm-tracking/index.html" >
                    <div class="header">
                        LLM Tracking with MLflow
                    </div>
                </a>
                <p>
                    Dive into the intricacies of MLflow's LLM Tracking system. From capturing prompts to monitoring generated outputs, 
                    discover how MLflow provides a holistic solution for managing LLM interactions.
                </p>
            </div>
            <div class="simple-card">
                <a href="llms/rag/index.html" >
                    <div class="header">
                        Question Generation for RAG
                    </div>
                </a>
                <p>
                    Learn how to leverage LLMs to generate a question dataset for use in Retrieval Augmented Generation applications.
                </p>
            </div>
        </article>
    </section>


Model Evaluation
----------------

Dive into MLflow's robust framework for evaluating the performance of your ML models. With the `mlflow.evaluate()` API, you can assess models on 
your chosen datasets, including special support for Large Language Models (LLMs) encompassing tasks like text summarization and question answering. 
Additionally, discover the intricacies of prompt engineering with OpenAI, define custom metrics, and set validation thresholds for comprehensive 
model quality checks. 
Visual insights are also available through the MLflow UI, showcasing logged outputs and model comparison artifacts. 




Deep Learning
-------------

See how MLflow can help manage the full lifecycle of your Deep Learning projects. Whether you're using frameworks like **TensorFlow (tensorflow)**, **Keras (keras)**, 
**PyTorch (pytorch)**, or **MXNet Gluon (gluon)**, MLflow offers first-class support, ensuring seamless integration and deployment. Additionally, 
libraries like **Fastai (fastai)** and **ONNX (onnx)** are also natively supported. Paired with MLflow's streamlined APIs and comparative UI, 
you are equipped with everything needed to manage, track, and optimize your deep learning workflows.



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
                </a>
                <p>
                    This in-depth guide will show you how to leverage some core functionality in MLflow to keep your tuning iterations organized and 
                    searchable, all while covering a number of features within MLflow that cater to the needs of this common activity.
                </p>
            </div>
        </article>
    </section>


Deployment
----------

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="deployment/kubernetes-deployment/index.html" >
                    <div class="header">
                        Deploying a Model to Kubernetes with MLflow
                    </div>
                </a>
                <p>
                    This guide showcases the seamless end-to-end process of training a linear regression model, packaging it in a reproducible format, 
                    and deploying to a Kubernetes cluster using MLflow. Explore how MLflow simplifies model deployment to production environments.
                </p>
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
