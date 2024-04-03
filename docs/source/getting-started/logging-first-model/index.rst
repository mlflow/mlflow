Tutorial Overview
=================

In this entry point tutorial to MLflow, we'll be covering the essential basics of core MLflow functionality associated 
with tracking training event data. 

We'll start by learning how to start a local MLflow Tracking server, how to access and view the MLflow UI, and move on to 
our first interactions with the Tracking server through the use of the MLflow Client. 

The tutorial content builds upon itself, culminating in successfully logging your first MLflow model.

The topics in this tutorial cover:

* Starting an **MLflow Tracking Server** (Optionally) and connecting to a Tracking Server
* Exploring the **MlflowClient** API (briefly)
* Understanding the Default Experiment
* **Searching** for Experiments with the MLflow client API
* Understanding the uses of **tags** and how to leverage them for model organization
* Creating an Experiment that will contain our **run** (and our model)
* Learning how to **log** metrics, parameters, and a model artifact to a run
* Viewing our Experiment and our first run within the **MLflow UI**

To get started with the tutorial, click NEXT below or navigate to the section that you're interested in:

.. toctree::
    :maxdepth: 1

    step1-tracking-server
    step2-mlflow-client
    step3-create-experiment
    step4-experiment-search
    step5-synthetic-data
    step6-logging-a-run
    notebooks/index

If you would instead like to download a notebook-based version of this guide and follow along locally, you can download the notebook from the link below. 

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/getting-started/logging-first-model/notebooks/logging-first-model.ipynb" class="notebook-download-btn">
        <i class="fas fa-download"></i>Download the Notebook</a><br/>
