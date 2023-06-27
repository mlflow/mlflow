.. _recipes:

===============================
MLflow Recipes
===============================

MLflow Recipes (previously known as MLflow Pipelines) is a framework that enables data scientists
to quickly develop high-quality models and deploy them to production.
Compared to ad-hoc ML workflows, MLflow Recipes offers several major benefits:

- **Get started quickly**: :ref:`Predefined templates <recipe-templates>` for common ML tasks,
  such as :ref:`regression modeling <regression-template>`, enable data scientists to get started
  quickly and focus on building great models, eliminating the large amount of boilerplate code that
  is traditionally required to curate datasets, engineer features, train & tune models, and package
  models for production deployment.

- **Iterate faster**: The intelligent recipe execution engine accelerates model development by
  caching results from each step of the process and re-running the minimal set of steps as changes
  are made.

- **Easily ship to production**: The modular, git-integrated :ref:`recipe structure
  <recipe-templates-key-concept>` dramatically simplifies the handoff from development to
  production by ensuring that all model code, data, and configurations are easily reviewable and
  deployable by ML engineers.

Quickstarts
-----------

Prerequisites
~~~~~~~~~~~~~

.. _recipes-installation:

MLflow Recipes is available as an extension of the
`MLflow Python library <https://pypi.org/project/mlflow/>`_. You can install MLflow Recipes
as follows:

- **Local**: Install MLflow from PyPI: ``pip install mlflow``.
  Note that MLflow Recipes requires `Make <https://www.gnu.org/software/make>`_,
  which may not be preinstalled on some Windows systems.
  Windows users must install Make before using MLflow Recipes. For more information about
  installing Make on Windows, see https://gnuwin32.sourceforge.net/install.html.

- **Databricks**: Install MLflow Recipes from a Databricks Notebook by running
  ``%pip install mlflow``, or install MLflow Recipes on a Databricks Cluster by
  following the PyPI library installation instructions `here
  <https://docs.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster>`_
  and specifying the ``mlflow`` package string.

  .. note::
    `Databricks Runtime <https://docs.databricks.com/runtime/dbr.html>`_ version 11.0
    or greater is required in order to install MLflow Recipes on Databricks.

NYC taxi fare prediction example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `NYC taxi fare prediction example <https://github.com/mlflow/recipes-examples/tree/main/regression>`_
uses the |MLflow Recipes Regression Template| to develop and score models on the
`NYC Taxi (TLC) Trip Record Dataset
<https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page>`_. You can run the example locally
by :ref:`installing MLflow Recipes <recipes-installation>` and running the `Jupyter example
regression notebook <https://github.com/mlflow/recipes-examples/blob/main/regression/notebooks/jupyter.ipynb>`_.
You can run the example on Databricks by `cloning the example repository with Databricks Repos
<https://docs.databricks.com/repos/work-with-notebooks-other-files.html#clone-a-remote-git-repository>`_
and running the `Databricks example regression notebook
<https://github.com/mlflow/recipes-examples/blob/main/regression/notebooks/databricks.py>`_.

To build and score models for your own use cases, we recommend using the
|MLflow Recipes Regression Template|. For more information, see the
|Regression Template reference guide|.


Classification problem example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `Classification problem example <https://github.com/mlflow/recipes-examples/tree/main/classification>`_
uses the |MLflow Recipes Classification Template| to develop and score models on the
`Wine Quality Dataset <https://github.com/mlflow/recipes-examples/tree/main/classification/data>`_.
You can run the example locally
by :ref:`installing MLflow Recipes <recipes-installation>` and running the `Jupyter example
classification notebook <https://github.com/mlflow/recipes-examples/blob/main/classification/notebooks/jupyter.ipynb>`_.
You can run the example on Databricks by `cloning the example repository with Databricks Repos
<https://docs.databricks.com/repos/work-with-notebooks-other-files.html#clone-a-remote-git-repository>`_
and running the `Databricks example classification notebook
<https://github.com/mlflow/recipes-examples/blob/main/classification/notebooks/databricks.py>`_.

To build and score models for your own use cases, we recommend using the
|MLflow Recipes Classification Template|. For more information, see the
|Classification Template reference guide|.

Key concepts
------------

.. _steps-key-concept:

- **Steps**: A Step represents an individual ML operation, such as ingesting data, fitting an
  estimator, evaluating a model against test data, or deploying a model for real-time scoring.
  Each Step accepts a collection of well-defined inputs and produce well-defined outputs according
  to user-defined configurations and code.

.. _recipes-key-concept:

- **Recipes**: A Recipe is an ordered composition of :ref:`Steps <steps-key-concept>` used to
  solve an ML problem or perform an MLOps task, such as developing a regression model or performing
  batch model scoring on production data. MLflow Recipes provides
  :py:class:`APIs <mlflow.recipes.Recipe>` and a :ref:`CLI <cli>` for running recipes and
  inspecting their results.

.. _recipe-templates-key-concept:

- **Templates**: A Recipe Template is a git repository with a :ref:`standardized, modular layout
  <recipe-template-structure>` containing all of the customizable code and configurations for a
  :ref:`Recipe <recipes-key-concept>`. Configurations are defined in YAML format for easy
  review via the |recipe.yaml| file and :ref:`Profile YAML files <profiles-key-concept>`. Each
  template also defines its requirements, data science notebooks, and tests. MLflow Recipes
  includes :ref:`predefined templates <recipe-templates>` for a variety of model development and
  MLOps tasks.

.. _profiles-key-concept:

- **Profiles**: Profiles contain user-specific or environment-specific configurations for a
  :ref:`Recipe <recipes-key-concept>`, such as the particular set of hyperparameters being
  tuned by a data scientist in development or the MLflow Model Registry URI and credentials
  used to store production-worthy models. Each profile is represented as a YAML file
  in the :ref:`Recipe Template <recipe-templates-key-concept>` (e.g.
  `local.yaml <https://github.com/mlflow/recipes-examples/blob/main/regression/profiles/local.yaml>`_
  and `databricks.yaml
  <https://github.com/mlflow/recipes-examples/blob/main/regression/profiles/databricks.yaml>`_).

.. _step-cards-key-concept:

- **Step Cards**: Step Cards display the results produced by running a
  :ref:`Step <steps-key-concept>`, including dataset profiles, model performance & explainability
  plots, overviews of the best model parameters found during tuning, and more. Step Cards and their
  corresponding dataset and model information are also logged to MLflow Tracking.

Usage
-----
Model development workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

The general model development workflow for using MLflow Recipes is as follows:

1. Clone a :ref:`Recipe Template <recipe-templates>` git repository corresponding to the ML
   problem that you wish to solve. Follow the template's README file for template-specific
   instructions.

    - [Local] Clone the |MLflow Recipes Regression Template| into a local directory.

    .. code-block:: sh

      git clone https://github.com/mlflow/recipes-regression-template

    - [Databricks] Clone the |MLflow Recipes Regression Template| git repository using |Databricks Repos|.

      .. image:: _static/images/recipes_databricks_repo_ui.png
        :width: 60%

2. Edit required fields marked by ``FIXME::REQUIRED`` comments in ``recipe.yaml`` and
   ``profiles/*.yaml``. The recipe is runnable once all required fields are filled with
   proper values. You may proceed to step 3 if this is the first time going through this step.
   Otherwise, continue to edit the YAML config files as well as ``steps/*.py`` files,
   filling out areas marked by ``FIXME::OPTIONAL`` as you see fit to
   customize the recipe steps to your ML problem for better model performance.

      .. image:: _static/images/recipes_databricks_fixme.png
        :width: 60%

3. Run the recipe by selecting a desired profile. Profiles are used to quickly switch environment
   specific recipe settings, such as ingest data location.
   When a recipe run completes, you may inspect the run results. MLflow Recipes
   creates and displays an interactive **Step Card** with the results of the last executed
   :ref:`step <steps-key-concept>`.
   Each :ref:`Recipe Template <recipe-templates>` also includes a |Databricks Notebook|
   and a |Jupyter Notebook| for running the recipe and inspecting its results.

    .. code-section::

        .. code-block:: python
            :caption: Example API and CLI workflows for running the |MLflow Recipes Regression Template| and inspecting results. Note that recipes must be run from within their corresponding git repositories.

            import os
            from mlflow.recipes import Recipe
            from mlflow.pyfunc import PyFuncModel

            os.chdir("~/recipes-regression-template")
            regression_recipe = Recipe(profile="local")
            # Run the full recipe
            regression_recipe.run()
            # Inspect the model training results
            regression_recipe.inspect(step="train")
            # Load the trained model
            regression_model_recipe: PyFuncModel = regression_recipe.get_artifact("model")

        .. code-block:: sh

          git clone https://github.com/mlflow/recipes-regression-template
          cd recipes-regression-template
          # Run the full recipe
          mlflow recipes run --profile local
          # Inspect the model training results
          mlflow recipes inspect --step train --profile local
          # Inspect the resulting model performance evaluations
          mlflow recipes inspect --step evaluate --profile local


    .. figure:: _static/images/recipes_evaluate_step_card.png
      :width: 60%

      An example step card produced by running the **evaluate** step of the
      |MLflow Recipes Regression Template|. The step card results
      indicate that the trained model passed all performance validations and is ready for
      registration with the :ref:`MLflow Model Registry <registry>`.


    .. figure:: _static/images/recipes_databricks_logged_artifacts.png
      :width: 60%

      An example MLflow run view page, showing artifacts logged from the Recipe's steps.

    .. figure:: _static/images/recipes_databricks_notebook_ui.png
      :scale: 25

      Example recipe run from the |Databricks Notebook| included in the
      |MLflow Recipes Regression Template|.

    .. note::
      Data profiling is often best viewed with "quantiles" mode. To switch it on, on the Facet
      data profile, find ``Chart to show``, click the selector below, and choose ``Quantiles``.

4. Iterate over step 2 and 3: make changes to an individual step, and test them by running
   the step and observing the results it produces.
   Use ``Recipe.inspect()`` to visualize the overall Recipe dependency graph and artifacts
   each step produces.
   Use ``Recipe.get_artifact()`` to further inspect individual step outputs in a notebook.

   MLflow Recipes intelligently caches results from each :ref:`Recipe Step <steps-key-concept>`,
   ensuring that steps are only executed if their inputs, code, or configurations have changed,
   or if such changes have occurred in dependent steps. Once you are satisfied with the results of
   your changes, commit them to a branch of the :ref:`Recipe Repository
   <recipe-templates-key-concept>` in order to ensure reproducibility, and share or review the
   changes with your team.

      .. figure:: _static/images/recipes_databricks_dag.png
        :width: 60%

        Example Recipe.inspect() output, showing the dependency graph of recipe steps and
        artifacts each step produces.

    .. note::
      Before testing changes in a staging or production environment, it is recommended that you
      commit the changes to a branch of the
      :ref:`Recipe Repository <recipe-templates-key-concept>` to ensure reproducibility.

    .. note::
      By default, MLflow Recipes caches results from each :ref:`Recipe Step
      <steps-key-concept>` within the ``.mlflow`` subdirectory of the home folder on the
      local filesystem. The ``MLFLOW_RECIPES_EXECUTION_DIRECTORY`` environment variable can
      be used to specify an alternative location for caching results.

Development environments
~~~~~~~~~~~~~~~~~~~~~~~~
We recommend using one of the following environment configurations to develop models with MLflow Recipes:

[**Databricks**]
  - Edit YAML config and Python files in Databricks Repos. Open separate browser tabs for each
    file module that you wish to modify. For example,
    one for the recipe config file ``recipe.yaml``,
    one for the profile config file ``profile/databricks.yaml``,
    one for the driver notebook ``notebooks/databricks.py``,
    and one for the current step (e.g. train) under development ``steps/train.py``.
  - Use ``notebooks/databricks.py`` as the driver to run recipe steps and inspect its output.
  - Pin the workspace browser for easy file navigation.

  .. image:: _static/images/recipes_databricks_ui.png
    :width: 60%

[**Local with Jupyter Notebook**]
  - Use ``notebooks/jupyter.ipynb`` as the driver to run recipe steps and inspect its output.
  - Edit ``recipe.yaml``, ``steps/*.py`` and ``profiles/*.yaml`` accordingly with an editor of your
    choice.
  - To run the entire recipe, either run ``notebooks/jupyter.ipynb`` or on commandline, invoke
    ``mlflow recipes run --profile local`` (change the current working directory to the project root first).

[**Edit locally with IDE (VSCode) and run on Databricks**]
  - Edit files on your local machine with VSCode and Jupyter plugin.
  - Use |dbx| to sync them to |Databricks Repos| as demonstrated below.
  - On Databricks, use the ``notebooks/databricks.py`` notebook as the driver to run recipe steps and inspect their outputs.

  .. code-block:: sh
   :caption: Example workflow for efficiently editing a recipe on a local machine
             and synchronizing changes to |Databricks Repos|

   # Install the Databricks CLI, which is used to remotely access your Databricks Workspace
   pip install databricks-cli
   # Configure remote access to your Databricks Workspace
   databricks configure
   # Install dbx, which is used to automatically sync changes to and from Databricks Repos
   pip install dbx
   # Clone the MLflow Recipes Regression Template
   git clone https://github.com/mlflow/recipes-regression-template
   # Enter the MLflow Recipes Regression Template directory and configure dbx within it
   cd recipes-regression-template
   dbx configure
   # Use dbx to enable syncing from the repository directory to Databricks Repos
   dbx sync repo -d recipes-regression-template
   # Iteratively make changes to files in the repository directory and observe that they
   # are automatically synced to Databricks Repos


.. _recipe-templates:

Recipe Templates
------------------

MLflow Recipes currently offers the following predefined templates that can be easily customized
to develop and deploy high-quality, production-ready models for your use cases:

.. _regression-template:

- **MLflow Recipes Regression Template**: The MLflow Recipes Regression Template is designed for developing and scoring
  regression models. For more information, see the |Regression Template reference guide|.

- **MLflow Recipes Classification Template**: The MLflow Recipes Classification Template is designed for developing and scoring
  classification models. For more information, see the |Classification Template reference guide|.

Additional recipes for a variety of ML problems and MLOps tasks are under active development.


Detailed reference guide
------------------------

.. _recipe-template-structure:

Template structure
~~~~~~~~~~~~~~~~~~

Recipe Templates are git repositories with a standardized, modular layout. The following
example provides an overview of the recipe repository structure. It is adapted from the
|MLflow Recipes Regression Template|.

::

  ├── recipe.yaml
  ├── requirements.txt
  ├── steps
  │   ├── ingest.py
  │   ├── split.py
  │   ├── transform.py
  │   ├── train.py
  │   ├── custom_metrics.py
  ├── profiles
  │   ├── local.yaml
  │   ├── databricks.yaml
  ├── tests
  │   ├── ingest_test.py
  │   ├── ...
  │   ├── train_test.py
  │   ├── ...

The main components of the Recipe Template layout, which are common across all recipes, are:

    - ``recipe.yaml``: The main recipe configuration file that declaratively defines the
      attributes and behavior of each recipe step, such as the input dataset to use for training
      a model or the performance criteria for promoting a model to production. For reference,
      see the |recipe.yaml| configuration file from the |MLflow Recipes Regression Template|.

    - ``requirements.txt``: A `pip requirements file
      <https://pip.pypa.io/en/stable/reference/requirements-file-format>`_ specifying packages
      that must be installed in order to run the recipe.

    - ``steps``: A directory containing Python code modules used by the recipe steps. For example,
      the |MLflow Recipes Regression Template| defines the estimator type and parameters to use
      when training a model in |steps/train.py| and defines custom metric computations in
      |steps/custom_metrics.py|.

    .. _profiles-directory:

    - ``profiles``: A directory containing :ref:`Profile <profiles-key-concept>` customizations for
      the configurations defined in ``recipe.yaml``. For example, the
      |MLflow Recipes Regression Template| defines a |local profile| that
      |customizes the dataset used for local model development| and |specifies a local MLflow
      Tracking store for logging model content|. The |MLflow Recipes Regression Template| also
      defines a |databricks profile| for development on Databricks.

    - ``tests``: A directory containing Python test code for recipe steps. For example, the
      |MLflow Recipes Regression Template| implements tests for the transformer and the estimator
      defined in the respective ``steps/transform.py`` and ``steps/train.py`` modules.

.. code-block:: yaml
    :caption: Shown below is an example |recipe.yaml| configuration file adapted from the
              |MLflow Recipes Regression Template|. ``recipe.yaml`` is the main
              configuration file for a recipe containing aggregated configurations for
              all recipe steps; :ref:`Profile <profiles-key-concept>`-based substitutions and
              overrides are supported using |Jinja2| templating syntax.

    recipe: "regression/v1"
    target_col: "fare_amount"
    primary_metrics: "root_mean_squared_error"
    steps:
      ingest: {{INGEST_CONFIG}}
      split:
        split_ratios: {{SPLIT_RATIOS|default([0.75, 0.125, 0.125])}}
      transform:
        using: custom
        transformer_method: transformer_fn
      train:
        using: custom
        estimator_method: estimator_fn
      evaluate:
        validation_criteria:
          - metric: root_mean_squared_error
            threshold: 10
          - metric: weighted_mean_squared_error
            threshold: 20
      register:
        allow_non_validated_model: false
    custom_metrics:
      - name: weighted_mean_squared_error
        function: weighted_mean_squared_error
        greater_is_better: False



Working with profiles
~~~~~~~~~~~~~~~~~~~~~

A profile is a collection of customizations for the configurations defined in the recipe's main
:ref:`recipe.yaml <recipe-template-structure>` file. Profiles are defined as YAML files
within the recipe repository's :ref:`profiles directory <profiles-directory>`. When running a
recipe or inspecting its results, the desired profile is specified as an API or CLI argument.

.. code-section::

    .. code-block:: python
      :caption: Example API and CLI workflows for running recipes with different profile customizations

      import os
      from mlflow.recipes import Recipe

      os.chdir("~/recipes-regression-template")
      # Run the regression recipe to train and evaluate the performance of an ElasticNet regressor
      regression_recipe_local_elasticnet = Recipe(profile="local-elasticnet")
      regression_recipe_local_elasticnet.run()
      # Run the recipe again to train and evaluate the performance of an SGD regressor
      regression_recipe_local_sgd = Recipe(profile="local-sgd")
      regression_recipe_local_sgd.run()
      # After finding the best model type and updating the 'shared-workspace' profile accordingly,
      # run the recipe again to retrain the best model in a workspace where teammates can view it
      regression_recipe_shared = Recipe(profile="shared-workspace")
      regression_recipe_shared.run()

    .. code-block:: sh

      git clone https://github.com/mlflow/recipes-regression-template
      cd recipes-regression-template
      # Run the regression recipe to train and evaluate the performance of an ElasticNet regressor
      mlflow recipes run --profile local-elasticnet
      # Run the recipe again to train and evaluate the performance of an SGD regressor
      mlflow recipes run --profile local-sgd
      # After finding the best model type and updating the 'shared-workspace' profile accordingly,
      # run the recipe again to retrain the best model in a workspace where teammates can view it
      mlflow recipes run --profile shared-workspace

The following profile customizations are supported:

    - overrides
        - If the ``recipe.yaml`` configuration file defines a |Jinja2|-templated attribute with
          a default value, a profile can override the value by mapping the attribute to a different
          value using YAML dictionary syntax. Note that override values may have arbitrarily nested
          types (e.g. lists, dictionaries, lists of dictionaries, ...).

          .. code-block:: yaml
            :caption: Example ``recipe.yaml`` configuration file defining an overrideable
                      ``RMSE_THRESHOLD`` attribute for validating model performance with a
                      default value of ``10``

            steps:
              evaluate:
                validation_criteria:
                  - metric: root_mean_squared_error
                    # The maximum RMSE value on the test dataset that a model can have
                    # to be eligible for production deployment
                    threshold: {{RMSE_THRESHOLD|default(10)}}

          .. code-block:: yaml
            :caption: Example ``prod.yaml`` profile that overrides ``RMSE_THRESHOLD`` with
                      a custom value to more aggressively validate model quality for production

            RMSE_THRESHOLD: 5.2

    - substitutions
        - If the ``recipe.yaml`` configuration file defines a |Jinja2|-templated attribute
          without a default value, a profile *must* map the attribute to a specific value using
          YAML dictionary syntax. Note that substitute values may have arbitrarily nested types
          (e.g. lists, dictionaries, lists of dictionaries, ...).

          .. code-block:: yaml
            :caption: Example ``recipe.yaml`` configuration file defining a ``DATASET_INFO``
                      variable whose value must be specified by the selected recipe profile

            # Specifies the dataset to use for model training
            ingest: {{INGEST_CONFIG}}

          .. code-block:: yaml
            :caption: Example ``dev.yaml`` profile that provides a value for ``DATASET_INFO``
                      corresponding to a small dataset for development purposes

            INGEST_CONFIG:
                location: ./data/taxi-small.parquet
                format: parquet

    - additions
        - If the ``recipe.yaml`` configuration file does not define a particular attribute, a
          profile may define it instead. This capability is helpful for providing values of
          optional configurations that, if unspecified, a recipe would otherwise ignore.

          .. code-block:: yaml
            :caption: Example ``local.yaml`` profile that specifies a
                      `sqlite <https://www.sqlite.org/index.html>`_-based
                      :ref:`MLflow Tracking <tracking>` store for local testing on a laptop

            experiment:
              tracking_uri: "sqlite:///metadata/mlflow/mlruns.db"
              name: "sklearn_regression_experiment"
              artifact_location: "./metadata/mlflow/mlartifacts"


    .. warning::
        If the ``recipe.yaml`` configuration file defines an attribute that cannot be overridden
        or substituted (i.e. because its value is not specified using |Jinja2| templating syntax),
        a profile must not define it. Defining such an attribute in a profile produces an error.


.. |MLflow Recipes Regression Template| replace:: :ref:`MLflow Recipes Regression Template <regression-template>`
.. |MLflow Recipes Classification Template| replace:: :ref:`MLflow Recipes Classification Template <regression-template>`
.. |Regression Template reference guide| replace:: `Regression Template reference guide <https://github.com/mlflow/recipes-regression-template/blob/main/README.md>`__
.. |Classification Template reference guide| replace:: `Classification Template reference guide <https://github.com/mlflow/recipes-classification-template/blob/main/README.md>`__
.. |recipe.yaml| replace:: `recipe.yaml <https://github.com/mlflow/recipes-regression-template/blob/main/recipe.yaml>`__
.. |train step| replace:: `train step <https://github.com/mlflow/recipes-regression-template#train-step>`__
.. |split step| replace:: `split step <https://github.com/mlflow/recipes-regression-template#split-step>`__
.. |Jinja2| replace:: `Jinja2 <https://jinja.palletsprojects.com>`__
.. |local profile| replace:: `profiles/local.yaml profile <https://github.com/mlflow/recipes-regression-template/blob/main/profiles/local.yaml>`__
.. |databricks profile| replace:: `profiles/databricks.yaml profile <https://github.com/mlflow/recipes-regression-template/blob/main/profiles/databricks.yaml>`__
.. |customizes the dataset used for local model development| replace:: `customizes the dataset used for local model development <https://github.com/mlflow/recipes-regression-template/blob/main/profiles/local.yaml#L17>`__
.. |specifies a local MLflow Tracking store for logging model content| replace:: `specifies a local MLflow Tracking store for logging model content <https://github.com/mlflow/recipes-regression-template/blob/main/profiles/local.yaml#L4-L7>`__
.. |Databricks Repos| replace:: `Databricks Repos <https://docs.databricks.com/repos/index.html>`__
.. |Databricks Notebook| replace:: `Databricks Notebook <https://github.com/mlflow/recipes-regression-template/blob/main/notebooks/databricks.py>`__
.. |Jupyter Notebook| replace:: `Jupyter Notebook <https://github.com/mlflow/recipes-regression-template/blob/main/notebooks/jupyter.ipynb>`__
.. |dbx| replace:: `dbx <https://docs.databricks.com/dev-tools/dbx.html>`__
.. |edit files in Databricks Repos| replace:: `edit files in Databricks Repos <https://docs.databricks.com/repos/work-with-notebooks-other-files.html#edit-a-file>`__
.. |steps/train.py| replace:: `steps/train.py <https://github.com/mlflow/recipes-regression-template/blob/main/steps/train.py>`__
.. |steps/custom_metrics.py| replace:: `steps/custom_metrics.py <https://github.com/mlflow/recipes-regression-template/blob/main/steps/custom_metrics.py>`__
