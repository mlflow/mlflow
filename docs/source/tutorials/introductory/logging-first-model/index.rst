Logging your first model with MLflow
====================================

In this tutorial, we'll be starting with the basics of using MLflow to record information about a model
training event.

In the first section, we'll be starting a local MLflow Tracking Server from a command prompt (shell) to begin with.
After the tracking server is running, we'll start the MLflow UI, use the MLflow Client API to interact with the
tracking server, and finally, we'll build towards logging our first model in MLflow.

We'll be focusing on:

* Starting a local MLflow Tracking Server and the MLflow UI Server
* Exploring the MlflowClient (briefly)
* Understanding the Default Experiment
* Searching for Experiments with the MLflow client API
* Understanding the uses of tags and how to leverage them for model organization
* Creating an Experiment that will contain our run (and our model)
* Learning how to log metrics, parameters, and a model artifact to a run
* Viewing our Experiment and our first run within the MLflow UI

For those who prefer to view the complete code before diving into the step-by-step creation process,
we offer two options:

.. raw:: html

    <div class="grid-container">
        <a href="notebooks/logging-first-model.html" class="button default" style="background-image: url('../../../_static/images/tutorials/logging-first-model-notebook.gif');"><span>View the Notebook</span></a>
        <a href="notebooks/logging-first-model.ipynb" class="button default" style="background-image: url('../../../_static/icons/download-icon.svg');"><span>Download</span></a>
    </div>
    <br><br>

.. toctree::
    :maxdepth: 1
    :caption: If you would like to navigate directly to a topic within this tutorial, each link below will bring you to the relevant section:

    Setting up our environment to use MLflow <step1-tracking-server>
    Getting familiar with the MLflow Client API and the Default Experiment <step2-mlflow-client>
    Creating your first Experiment <step3-create-experiment>
    Learning to search experiments and how to use custom tags <step4-experiment-search>
    Generating a test data set for training <step5-synthetic-data>
    Logging your first model to MLflow <step6-logging-a-run>
    A full end-to-end example Notebook <notebooks/index>

To get started with the tutorial, click NEXT below.
