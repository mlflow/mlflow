MLflow: A Tool for Managing the Machine Learning Lifecycle
==========================================================

MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in
handling the complexities of the machine learning process. Mlflow focuses on the full lifecycle for
machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.

Core Components of MLflow
-------------------------

* :ref:`Tracking <tracking>`: A centralized repository that allows for recording the history of model training. It captures parameters, metrics, artifacts, data and environment configurations, allowing teams to observe the evolution of their models over time.

* :ref:`Model Registry <registry>`: A systematic way to manage models. It helps in managing the state of different versions of models, understanding their current state, and facilitating smoother transitions from development stages to production deployments.

* :ref:`AI Gateway <gateway>`: A server and set of standard APIs that streamlines the access to both SaaS and OSS LLM models. It acts as a consistent interface, enhances security by handling authenticated access, and provides a simplified common set of APIs for leveraging popular LLMs.

* :ref:`Evaluate <model-evaluation>`: A set of tools designed for detailed model analysis. It provides functionalities to objectively compare different models, be they traditional ML algorithms or the most state-of-the-art LLMs.

* PromptLab: An environment dedicated to prompt engineering. It offers a space where users can experiment with, refine, and test prompts in isolation.

* :ref:`Recipes <recipes>`: A framework that guides users in structuring ML projects. While it offers recommendations, the goal is to ensure that the end results are not just functional but also optimized for real-world deployments.

Why Use MLflow?
---------------

The ML process can be intricate, with multiple stages and numerous considerations at each step.
MLflow aims to simplify this process by providing a unified platform where each phase of the ML lifecycle
is treated as a critical component. By doing so, it ensures:

- **Traceability**: With tools like the Tracking Server, every experiment is logged, ensuring that teams can trace back and understand the evolution of models.

- **Consistency**: Whether you're accessing models through the AI Gateway or structuring projects with MLflow Recipes, there's a consistent approach, reducing the learning curve and potential for errors.

- **Flexibility**: MLflow is designed to be library-agnostic, making it compatible with various machine learning libraries. Additionally, its functionalities are available across different programming languages, supported by a comprehensive :ref:`rest-api`, :ref:`CLI<cli>`, and APIs for :ref:`python-api`, :ref:`R-api`, and :ref:`java_api`.

Learn about MLflow
------------------
MLflow can seem daunting at first. While there are many features that you will probably want to end up using, starting with the core concepts
will help to reduce the complexity. By following along with the tutorials and guides,
To get started with the core components of MLflow, follow along with the starter tutorials and guides below.

.. tip:: **New to MLFlow?**

    Starting with the *Introductory Tutorials* is recommended before moving on to the guides.

.. container:: boxes-wrapper

    .. container:: left-box

        **Introductory Tutorials**

        * :doc:`tutorials/introductory/logging-first-model/index`
        * Navigating the MLflow UI
        * Serving your first model
        * Comparing runs in the UI
        * Prompt Engineering with PromptLab 🎉 **new!** 🎉



        **Expert Tutorials**

        * Using nested runs for hyperparameter optimization
        * Packaging custom code with a model
        * Batch inference with Apache Spark

    .. container:: right-box

        **Introductory Guides**

        - Using the MLflow AI Gateway
        - Creating a custom pyfunc

        **Expert Guides**

        - MLflow server deployment options
        - Creating plugins
        - Creating a custom flavor



Get started using the :ref:`quickstart` or by reading about the :ref:`key concepts<concepts>`.

.. toctree::
    :maxdepth: 1

    what-is-mlflow
    introduction/index
    tutorials/index
    quickstart
    quickstart_mlops
    tutorials-and-examples/index
    concepts
    tracking
    llm-tracking
    projects
    models
    model-registry
    recipes
    gateway/index
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
