Logging your first model with MLflow
====================================

In this tutorial, we'll be starting with the basics of using MLflow to record information about a model
training event.

In the first section, we'll be starting a local MLflow Tracking Server from a command prompt (shell) to begin with.
After the tracking server is running, we'll start the MLflow UI, use the MLflow Client API to interact with the
tracking server, and finally, we'll build towards logging our first model in MLflow.

We'll be focusing on:

* Starting an MLflow Tracking Server
* Understanding the Default Experiment
* Searching Experiments
* Creating an Experiment that will encapsulate the runs that we will be initiating.
* Creating a series of runs that will be used to store individual iterations of training in our Experiment.
* Logging metrics, parameters, and tags to our runs.
* Storing our model artifacts to these runs.
* Viewing our Experiment and its Runs within the MLflow UI.


.. toctree::
    :hidden:

    step1-tracking-server
    step2-mlflow-client
    step3-create-experiment
    step4-experiment-search
    step5-synthetic-data
    step6-logging-a-run
    notebooks/index