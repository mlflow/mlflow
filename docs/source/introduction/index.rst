MLflow 101
==========

Stepping into the world of Machine Learning (ML) is an exciting journey, but it often comes with
complexities that can hinder innovation and experimentation. MLflow is a solution to many of these issues in this
dynamic landscape, offering tools and processes to streamline the ML lifecycle and foster collaboration
among ML practitioners.

Whether you're an individual researcher, a member of a large team, or somewhere in between, MLflow
provides a unified platform to navigate the intricate maze of model development, deployment, and
management. It's not just about efficiency – it's about enabling innovation and ensuring that ML
projects are robust, transparent, and ready for real-world challenges.

Read on to discover the core components of MLflow and understand the unique advantages it brings
to the table.

Core Components of MLflow
-------------------------

At its essence, MLflow is a robust framework designed to enhance and simplify your ML
journey. Despite its expansive suite of features and continuous enhancements, MLflow
remains grounded in a set of foundational components. These are meticulously crafted
to cater to a diverse range of ML projects and workflows, ensuring that you have the
right tools at every stage of your endeavor.

* :ref:`Tracking <tracking>`: A centralized repository that allows for recording the history of model training. It captures parameters, metrics, artifacts, data and environment configurations, allowing teams to observe the evolution of their models over time.

* :ref:`Model Registry <registry>`: A systematic way to manage models. It helps in managing the state of different versions of models, understanding their current state, and facilitating smoother transitions from development stages to production deployments.

* :ref:`AI Gateway <gateway>`: A server and set of standard APIs that streamlines the access to both SaaS and OSS LLM models. It acts as a consistent interface, enhances security by handling authenticated access, and provides a simplified common set of APIs for leveraging popular LLMs.

* :ref:`Evaluate <model-evaluation>`: A set of tools designed for detailed model analysis. It provides functionalities to objectively compare different models, be they traditional ML algorithms or the most state-of-the-art LLMs.

* Prompt Engineering UI: An environment dedicated to prompt engineering. It offers a space where users can experiment with, refine, evaluate, test, and deploy prompts in an interactive UI environment.

* :ref:`Recipes <recipes>`: A framework that guides users in structuring ML projects. While it offers recommendations, the goal is to ensure that the end results are not just functional but also optimized for real-world deployments.

Why Use MLflow?
---------------

The ML process can be intricate, with multiple stages and numerous considerations at each step.
MLflow aims to simplify this process by providing a unified platform where each phase of the ML lifecycle
is treated as a critical component. By doing so, it ensures:

- **Traceability**: With tools like the Tracking Server, every experiment is logged, ensuring that teams can trace back and understand the evolution of models.

- **Consistency**: Whether you're accessing models through the AI Gateway or structuring projects with MLflow Recipes, there's a consistent approach, reducing the learning curve and potential for errors.

- **Flexibility**: MLflow is designed to be library-agnostic, making it compatible with various machine learning libraries. Additionally, its functionalities are available across different programming languages, supported by a comprehensive :ref:`rest-api`, :ref:`CLI<cli>`, and APIs for :ref:`python-api`, :ref:`R-api`, and :ref:`java_api`.

Who Uses MLflow?
----------------

Throughout the lifecycle of a particular project, there are components within MLflow that are designed
to cater to different needs.

.. figure:: ../_static/images/what-is-mlflow/mlflow-overview.png
    :width: 100%
    :align: center
    :alt: MLflow overview, showing the ML lifecycle from data preparation to monitoring. Labels above show the personas associated each stage: Data engineers and scientists at the earlier stages, ML engineers and business stakeholders at the later stages. The Data Governance officer is involved at all stages.

MLflow's versatility enhances workflows across various roles, from data scientists to prompt
engineers, extending its impact beyond just the confines of a Data Science team.

.. container:: left-box

    **Data Scientists** leverage MLflow for:

    * Experiment tracking and hypothesis testing persistence.
    * Code structuring for better reproducibility.
    * Model packaging and dependency management.
    * Evaluating hyperparameter tuning selection boundaries.
    * Comparing the results of model retraining over time.
    * Reviewing and selecting optimal models for deployment.

    **MLOps Professionals** utilize MLflow to:

    * Manage the lifecycles of trained models, both pre and post deployment.
    * Deploy models securely to production environments.
    * Audit and review candidate models prior to deployment.
    * Manage deployment dependencies.

    **Data Science Managers** interact with MLflow by:

    * Reviewing the outcomes of experimentation and modeling activities.
    * Collaborating with teams to ensure that modeling objectives align with business goals.

    **Prompt Engineering Users** use MLflow for:

    * Evaluating and experimenting with large language models.
    * Crafting custom prompts and persisting their candidate creations.
    * Deciding on the best base model suitable for their specific project requirements.


Use Cases of MLflow
-------------------

MLflow is versatile, catering to diverse machine learning scenarios. Here are some typical use cases:

- **Experiment Tracking**: A data science team leverages MLflow Tracking to log parameters and metrics for experiments within a particular domain. Using the MLflow UI, they can compare results and fine-tune their solution approach. The outcomes of these experiments are preserved as MLflow models.

- **Model Selection and Deployment**: MLOps engineers employ the MLflow UI to assess and pick the top-performing models. The chosen model is registered in the MLflow Registry, allowing for monitoring its real-world performance.

- **Model Performance Monitoring**: Post deployment, MLOps engineers utilize the MLflow Registry to gauge the model's efficacy, juxtaposing it against other models in a live environment.

- **Collaborative Projects**: Data scientists embarking on new ventures organize their work as an MLflow Project. This structure facilitates easy sharing and parameter modifications, promoting collaboration.


Learn about MLflow
------------------
MLflow can seem daunting at first. While there are many features that you will probably want to end up using, starting with the core concepts
will help to reduce the complexity. By following along with the tutorials and guides,
To get started with the core components of MLflow, start with some of the selected tutorials and guides below.
You'll find more in the respective sections within the documentation.

.. tip:: **New to MLFlow?**

    Starting with the *Introductory Tutorials* is recommended before moving on to the guides.

Learning Journey
^^^^^^^^^^^^^^^^
Below is a list of a select group of tutorials and guides.

To see more of the tutorials available, visit :doc:`../tutorials/index` for the full listing.
To see the full listing of guides, visit :doc:`../guides/index`.

.. container:: boxes-wrapper

    .. container:: left-box

        **Introductory Tutorials**

        * :doc:`../tutorials/introductory/logging-first-model/index` 🎉 **new!** 🎉
        * Navigating the MLflow UI 🚧
        * Serving your first model 🚧
        * Comparing runs in the UI 🚧
        * Prompt Engineering with PromptLab 🚧

        **Expert Tutorials**

        * Packaging custom code with a model 🚧
        * Batch inference with Apache Spark 🚧

    .. container:: right-box

        **Introductory Guides**

        * :doc:`../guides/introductory/hyperparameter-tuning-with-child-runs/index` 🎉 **new!** 🎉
        * :doc:`../guides/introductory/deploy-model-to-kubernetes/index`
        * Using the MLflow AI Gateway 🚧
        * Creating a custom pyfunc 🚧

        **Expert Guides**

        * MLflow server deployment options 🚧
        * Creating plugins 🚧
        * Creating a custom flavor 🚧
