MLflow Guides
=============

Welcome to the MLflow guides section! This collection of guides provides in-depth solutions
to specific use cases, aiding you in achieving end-to-end objectives with MLflow.

**Purpose of These Guides**

Each guide is curated to tackle real-world scenarios, offering step-by-step walkthroughs
and best practices. By presenting solutions to common challenges faced by MLflow users,
we hope to empower you to optimize your workflows and applications.

Use Case Guides
---------------

For those looking to implement MLflow in specific scenarios or wanting to learn best practices
for unique use cases, these guides are invaluable. Like our tutorials, every guide comes with
supporting materials, be it code snippets, notebooks, or scripts, so you can get hands-on
experience.

.. raw:: html

    <section class="card-list">
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="introductory/hyperparameter-tuning-with-child-runs/index.html">
                    <h1>
                        Hyperparameter tuning with MLflow and Optuna
                    </h1>
                </a>
                <img src="../_static/images/intro/optuna-logo.jpeg" alt="Optuna Logo" class="card-header-image" style="max-height: 4.5rem;"></img>
                <h2>
                    Learn how to leverage parent and child runs, custom visualizations, and encapsulation to simplify model tuning
                </h2>
                <p>
                    In this in-depth guide to model tuning in MLflow, there are both notebook examples and supplementary 
                    instructional pages to explain and demonstrate the best practices around model tuning with MLflow.
                </p>
                <ul>
                    <li>
                        <span class="icon notebook"></span>
                        <a class="card-link" href="introductory/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html">
                            Hyperparameter tuning example  
                        </a>
                    </li>
                    <li>
                        <span class="icon notebook"></span>
                        <a class="card-link" href="introductory/hyperparameter-tuning-with-child-runs/notebooks/parent-child-runs.html">
                             Parent-Child run associations
                        </a>
                    </li>
                    <li>
                        <span class="icon teacher"></span>
                        <a class="card-link" href="introductory/hyperparameter-tuning-with-child-runs/part1-child-runs.html">
                            Understanding the benefits of using child runs 
                        </a>
                    </li>
                    <li>
                        <span class="icon notebook"></span>
                        <a class="card-link" href="introductory/hyperparameter-tuning-with-child-runs/notebooks/logging-plots-in-mlflow.html">
                            Logging custom plots in MLflow
                        </a>
                    </li>
                    <li>
                        <span class="icon teacher"></span>
                        <a class="card-link" href="introductory/hyperparameter-tuning-with-child-runs/part2-logging-plots.html">
                            Custom plot overview and where to view your logged plots
                        </a>
                    </li>
                </ul>
            </header>
            <div class="tags">
                <div>
                    <a class="download" href="introductory/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.ipynb">
                        Download the Hyperparameter tuning Notebook
                    </a>
                </div>
                <div>
                    <a class="download" href="introductory/hyperparameter-tuning-with-child-runs/notebooks/parent-child-runs.ipynb">
                        Download the Parent-Child notebook
                    </a>
                </div>
                <div>
                    <a class="download" href="introductory/hyperparameter-tuning-with-child-runs/notebooks/logging-plots-in-mlflow.ipynb">
                        Download the custom plot logging notebook
                    </a>
                </div>
        </article>
        <article class="card">
            <header class="card-header">
                <a class="card-link" href="introductory/deploy-model-to-kubernetes/index.html">
                    <h1>
                        Deploy a MLflow Model to Kubernetes
                    </h1>
                </a>
                <img src="../_static/images/guides/introductory/deploy-model-to-kubernetes/k8s-logo.svg" alt="Kubernetes Logo" class="card-header-image" style="max-height: 4.5rem;"></img>
                <h2>
                    Learn how to define and build a MLflow model container for deployment to Kubernetes
                </h2>
                <p>
                    In this guide, we take a look at the common deployment pattern of containerized model serving on elastically 
                    scaling Kubernetes. 

                    We will go through the processes of:
                </p>
                <ul>
                    <li>
                        <span class="icon teacher"></span>
                        <a class="card-link" href="introductory/deploy-model-to-kubernetes/index.html#training-the-model">
                            Building a model  
                        </a>
                    </li>
                    <li>
                        <span class="icon teacher"></span>
                        <a class="card-link" href="introductory/deploy-model-to-kubernetes/index.html#packaging-training-code-in-a-conda-environment">
                             Defining inference dependencies for our container
                        </a>
                    </li>
                    <li>
                        <span class="icon teacher"></span>
                        <a class="card-link" href="introductory/deploy-model-to-kubernetes/index.html#serving-the-model">
                            Locally testing model serving with MLServer 
                        </a>
                    </li>
                    <li>
                        <span class="icon teacher"></span>
                        <a class="card-link" href="introductory/deploy-model-to-kubernetes/index.html#deploy-the-model-to-seldon-core-or-kserve">
                            Deploying our model container to Kubernetes
                        </a>
                    </li>
                </ul>
            </header>
        </article>
    </section>

.. toctree::
    :maxdepth: 2
    :hidden:

    introductory/deploy-model-to-kubernetes/index
    introductory/hyperparameter-tuning-with-child-runs/index
