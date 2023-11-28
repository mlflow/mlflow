Getting Started with MLflow
===========================

For those new to MLflow or seeking a refresher on its core functionalities, the
quickstart tutorials here are the perfect starting point. They will guide you
step-by-step through fundamental concepts, focusing purely on a task that will maximize your understanding of 
how to use MLflow to solve a particular task. 

`5-minute Quickstart - MLflow Tracking <intro-quickstart/index.html>`_
----------------------------------------------------------------------

In this brief introductory quickstart on MLflow Tracking, you will learn how to leverage MLflow to:

* **Log** training statistics (loss, accuracy, etc.) and hyperparameters for a model
* **Log** (save) a model for later retrieval
* **Register** a model to enable deployment
* **Load** the model and use it for inference

In the process of learning these key concepts, you will be exposed to the MLflow fluent API, the MLflow Tracking UI, and learn how to add metadata associated with 
a model training event to an MLflow run.

.. toctree::
    :maxdepth: 1
    :hidden:

    intro-quickstart/index

If you would like to get started immediately by interactively running the notebook, you can:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/getting-started/intro-quickstart/notebooks/tracking_quickstart.ipynb" class="notebook-download-btn">
        <i class="fas fa-download"></i>Download the Notebook</a><br/>

Quickstart elements
^^^^^^^^^^^^^^^^^^^

You can read through the quickstart as a guide, or navigate directly to the notebook example to get started. 

.. raw:: html

     <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="intro-quickstart/index.html" >
                    <div class="header">
                        MLflow Tracking Quickstart Guide
                    </div>
                    <p>
                    Learn the basics of MLflow Tracking in a fast-paced guide with a focus on seeing your first model in the MLflow UI
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="intro-quickstart/notebooks/tracking_quickstart.html" >
                    <div class="header">
                        MLflow Tracking Quickstart Notebook
                    </div>
                    <p>
                        See an example of using the MLflow fluent API to log, load, and register a model for inference. Great for code-focused learners!
                    </p>
                </a>
            </div>
        </article>
    </section>


`Logging your first MLflow Model <logging-first-model/index.html>`_
-------------------------------------------------------------------

In this lengthy tutorial, you will walk through the basics of MLflow in a sequential and guided manner. With each subsequent step, 
you will increase your familiarity with the primary functionality around MLflow Tracking and how to navigate the MLflow UI.

If you would like to get started immediately by interactively running the notebook, you can:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/getting-started/logging-first-model/notebooks/logging-first-model.ipynb" class="notebook-download-btn">
        <i class="fas fa-download"></i>Download the Notebook</a><br/>

Guide sections
^^^^^^^^^^^^^^

Interested in navigating directly to the content that you're curious about? Select the section from each tutorial below!

.. raw:: html

     <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="logging-first-model/step1-tracking-server.html" >
                    <div class="header">
                        Setting up the MLflow Tracking Server
                    </div>
                    <p>
                        Learn how to start an MLflow Tracking Server and the MLflow UI Server locally
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step2-mlflow-client.html" >
                    <div class="header">
                        Using the MLflow Client
                    </div>
                    <p>
                        Connect to the Tracking Server with the MLflow Client and learn how to search for experiments
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step3-create-experiment.html" >
                    <div class="header">
                        Create your first Experiment
                    </div>
                    <p>
                        Explore the MLflow UI and create your first MLflow experiment with a unique name and identifying tags
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step4-experiment-search.html" >
                    <div class="header">
                        Search Experiments by tags
                    </div>
                    <p>
                        Learn how to use tags for MLflow experiments and how to leverage them for searching
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step5-synthetic-data.html" >
                    <div class="header">
                        Creating a Dataset for Testing
                    </div>
                    <p>
                        Build a synthetic dataset to use while exploring the features of MLflow
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step6-logging-a-run.html" >
                    <div class="header">
                        Logging your first MLflow run
                    </div>
                    <p>
                        Train a model using the synthetic dataset and log the trained model, metrics, and parameters
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/notebooks/logging-first-model.html" >
                    <div class="header">
                        View the full Notebook
                    </div>
                    <p>
                        See the tutorial notebook in its entirety. If you prefer just reading code, this is the best place to look.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. toctree::
    :maxdepth: 1
    :hidden:

    logging-first-model/index
    logging-first-model/step1-tracking-server
    logging-first-model/step2-mlflow-client
    logging-first-model/step3-create-experiment
    logging-first-model/step4-experiment-search
    logging-first-model/step5-synthetic-data
    logging-first-model/step6-logging-a-run
    logging-first-model/notebooks/index

`15 minute Quickstart - Autologging in MLflow <quickstart-1/index.html>`_
-------------------------------------------------------------------------

In this rapid-pace quickstart, you will be exposed to the autologging feature in MLflow to simplify the logging of models, metrics, and parameters. 
After training and viewing the logged run data, we'll load the logged model to perform inference, showing core features of MLflow Tracking in the 
most time-efficient manner possible.

.. raw:: html

     <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="quickstart-1/index.html#add-mlflow-tracking-to-your-code" >
                    <div class="header">
                        Introduction to autologging
                    </div>
                    <p>
                        Train a model and use MLflow autologging to automatically record model artifacts, metrics, and parameters
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="quickstart-1/index.html#view-mlflow-runs-and-experiments" >
                    <div class="header">
                        View autologged data in the MLflow UI
                    </div>
                    <p>
                        See what autologging will autonomously log for you during model training with only a single line of code
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="quickstart-1/index.html#load-a-model-from-a-specific-training-run-for-inference" >
                    <div class="header">
                        Loading a model for inference
                    </div>
                    <p>
                        Load the autologged model in its native format and use it to generate predictions
                    </p>
                </a>
            </div>
        </article>
    </section>

.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart-1/index

`15 minute Quickstart - Comparing Runs and Deploying your Best Model <quickstart-2/index.html>`_
------------------------------------------------------------------------------------------------

This quickstart tutorial focuses on the MLflow UI's run comparison feature, provides a brief overview of MLflow Projects, 
and shows how to register a model. After locally serving the registered model, a brief example of preparing a model for remote deployment 
via containerizing the model via Docker is covered. 

.. raw:: html

     <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="quickstart-2/index.html#set-up" >
                    <div class="header">
                        Generate runs
                    </div>
                    <p>
                        Run an MLflow Project that will perform hyperparameter tuning to generate a large volume of runs
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="quickstart-2/index.html#compare-the-results" >
                    <div class="header">
                        Run comparison
                    </div>
                    <p>
                        Use the MLflow UI Runs Compare functionality to evaluate the hyperparameter tuning run and select the best model
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="quickstart-2/index.html#register-your-best-model" >
                    <div class="header">
                        Register the best model
                    </div>
                    <p>
                        Learn to register models with the MLflow UI and perform stage transitions from within the UI
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="quickstart-2/index.html#serve-the-model-locally" >
                    <div class="header">
                        Start a local ML inference server
                    </div>
                    <p>
                        Use the integrated inference server in MLflow to serve your registered model locally
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="quickstart-2/index.html#build-a-container-image-for-your-model" >
                    <div class="header">
                        Build a deployable container for your model
                    </div>
                    <p>
                        Learn how to generate a docker container that houses your model for deployment to external services
                    </p>
                </a>
            </div>
        </article>
    </section>



.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart-2/index

`5 Minute Tracking Server Overview <tracking-server-overview/index.html>`_
---------------------------------------------------------------------------

This quickstart tutorial walks through different types of MLflow Tracking Servers and how to use them to log 
your MLflow experiments.

.. raw:: html

     <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tracking-server-overview/index.html" >
                    <div class="header">
                        5 Minute Tracking Server Overview
                    </div>
                </a>
                <p>
                    Learn how to log MLflow experiments with different tracking servers
                </p>
            </div>
        </article>
    </section>



.. toctree::
    :maxdepth: 1
    :hidden:

    tracking-server-overview/index