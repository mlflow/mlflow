Getting Started with MLflow
===========================

For those new to MLflow or seeking a refresher on its core functionalities, the
quickstart tutorials here are the perfect starting point. They will guide you
step-by-step through fundamental concepts, focusing purely on a task that will maximize your understanding of 
how to use MLflow to solve a particular task. 


`Logging your first MLflow Model <logging-first-model/index.html>`_
-------------------------------------------------------------------

In this quickstart tutorial, you will walk through the basics of MLflow in a sequential and guided manner. With each subsequent step, 
you will increase your familiarity with the primary functionality around MLflow Tracking and how to navigate the MLflow UI.

If you would like to get started immediately by interactively running the notebook, you can:

.. raw:: html

    <a href="logging-first-model/notebooks/logging-first-model.ipynb" class="download-btn">
        <i class="fas fa-download"></i>Download the Notebook</a><br>

Guide sections
^^^^^^^^^^^^^^

Interested in navigating directly to the content that you're curious about? Select the section from each tutorial below!

.. raw:: html

     <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="logging-first-model/step1-tracking-server.html" >
                    <div class="header">
                        Setting up MLflow Servers
                    </div>
                </a>
                <p>
                    Learn how to start a MLflow Tracking Server and the MLflow UI Server locally
                </p>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step2-mlflow-client.html" >
                    <div class="header">
                        Using the MLflow Client
                    </div>
                </a>
                <p>
                    Connect to the Tracking Server with the MLflow Client and learn to search for experiments
                </p>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step3-create-experiment.html" >
                    <div class="header">
                        Create your first Experiment
                    </div>
                </a>
                <p>
                    Explore the MLflow UI and create your first MLflow experiment with a unique name and identifying tags
                </p>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step4-experiment-search.html" >
                    <div class="header">
                        Search Experiments by tags
                    </div>
                </a>
                <p>
                    Learn the benefits of relevant identifying tags as you execute a search for experiments containing identifying tag values
                </p>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step5-synthetic-data.html" >
                    <div class="header">
                        Creating a Dataset for Testing
                    </div>
                </a>
                <p>
                    Build a synthetic dataset to use while exploring the features of MLflow
                </p>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/step6-logging-a-run.html" >
                    <div class="header">
                        Logging your first MLflow run
                    </div>
                </a>
                <p>
                    Train a model using the synthetic dataset and log the trained model, metrics, and parameters
                </p>
            </div>
            <div class="simple-card">
                <a href="logging-first-model/notebooks/logging-first-model.html" >
                    <div class="header">
                        View the full Notebook
                    </div>
                </a>
                <p>
                    See the tutorial notebook in its entirety. If you prefer just reading code, this is the best place to look.
                </p>
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

`15 minute Quickstart - Learn the basics of MLflow <quickstart-1/index.html>`_
------------------------------------------------------------------------------

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
                </a>
                <p>
                    Train a model and use MLflow autologging to automatically record model artifacts, metrics, and parameters
                </p>
            </div>
            <div class="simple-card">
                <a href="quickstart-1/index.html#view-mlflow-runs-and-experiments" >
                    <div class="header">
                        View autologged data in the MLflow UI
                    </div>
                </a>
                <p>
                    See what autologging will autonomously log for you during model training with only a single line of code
                </p>
            </div>
            <div class="simple-card">
                <a href="quickstart-1/index.html#load-a-model-from-a-specific-training-run-for-inference" >
                    <div class="header">
                        Loading a model for inference
                    </div>
                </a>
                <p>
                    Load the autologged model in its native format and use it to generate predictions
                </p>
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
                </a>
                <p>
                    Run an MLflow Project that will perform hyperparameter tuning to generate a large volume of runs
                </p>
            </div>
            <div class="simple-card">
                <a href="quickstart-2/index.html#compare-the-results" >
                    <div class="header">
                        Run comparison
                    </div>
                </a>
                <p>
                    Use the MLflow UI Runs Compare functionality to evaluate the hyperparameter tuning run and select the best model
                </p>
            </div>
            <div class="simple-card">
                <a href="quickstart-2/index.html#register-your-best-model" >
                    <div class="header">
                        Register the best model
                    </div>
                </a>
                <p>
                    Learn to register models with the MLflow UI and perform stage transitions from within the UI
                </p>
            </div>
            <div class="simple-card">
                <a href="quickstart-2/index.html#serve-the-model-locally" >
                    <div class="header">
                        Start a local ML inference server
                    </div>
                </a>
                <p>
                    Use the integrated inference server in MLflow to serve your registered model locally
                </p>
            </div>
            <div class="simple-card">
                <a href="quickstart-2/index.html#build-a-container-image-for-your-model" >
                    <div class="header">
                        Build a deployable container for your model
                    </div>
                </a>
                <p>
                    Learn how to generate a docker container that houses your model for deployment to external services
                </p>
            </div>
        </article>
    </section>



.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart-2/index
