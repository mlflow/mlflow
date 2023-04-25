"""
.. _mlflow-classification-recipe:

The MLflow Classification Recipe is an MLflow Recipe for developing binary classification
models. Multiclass classifiers are currently not supported.
The classification recipe is designed for developing models using scikit-learn and
frameworks that integrate with scikit-learn, such as the ``XGBClassifier`` API from XGBoost.
The `ClassificationRecipe API Documentation <https://github.com/mlflow/recipes-classification-template/blob/main/README.md>`
provides instructions for executing the recipe and inspecting its results.

The training recipe contains the following sequential steps:

**ingest** -> **split** -> **transform** -> **train** -> **evaluate** -> **register**

The batch scoring recipe contains the following sequential steps:

**ingest_scoring** -> **predict**

The recipe steps are defined as follows:

   - **ingest**
      - The **ingest** step resolves the dataset specified by
        |'ingest' step definition in recipe.yaml| and converts it to parquet format, leveraging
        the custom dataset parsing code defined in |steps/ingest.py| if necessary. Subsequent steps
        convert this dataset into training, validation, & test sets and use them to develop a model.

        .. note::
            If you make changes to the dataset referenced by the **ingest** step (e.g. by adding
            new records or columns), you must manually re-run the **ingest** step in order to
            use the updated dataset in the recipe. The **ingest** step does *not* automatically
            detect changes in the dataset.

        .. note::
            `target_col` must have a cardinality of two and `positive_class` must be specified.

   .. _mlflow-classification-recipe-split-step:

   - **split**
      - The **split** step splits the ingested dataset produced by the **ingest** step into
        a training dataset for model training, a validation dataset for model performance
        evaluation  & tuning, and a test dataset for model performance evaluation. The fraction
        of records allocated to each dataset is defined by the ``split_ratios`` attribute of the
        |'split' step definition in recipe.yaml|. The **split** step also preprocesses the
        datasets using logic defined in |steps/split.py|. Subsequent steps use these datasets
        to develop a model and measure its performance.

   - **transform**
      - The **transform** step uses the training dataset created by **split** to fit
        a transformer that performs the transformations defined in |steps/transform.py|. The
        transformer is then applied to the training dataset and the validation dataset, creating
        transformed datasets that are used by subsequent steps for estimator training and model
        performance evaluation.

   .. _mlflow-classification-recipe-train-step:

   - **train**
      - The **train** step uses the transformed training dataset output from the **transform**
        step to fit an estimator with the type and parameters defined in |steps/train.py|. The
        estimator is then joined with the fitted transformer output from the **transform** step
        to create a model recipe. Finally, this model recipe is evaluated against the
        transformed training and validation datasets to compute performance metrics; custom
        metrics are computed according to definitions in |steps/custom_metrics.py| and the
        |'custom_metrics' section of recipe.yaml|. The model recipe and its associated parameters,
        performance metrics, and lineage information are logged to MLflow Tracking, producing
        an MLflow Run.

            .. note::
                The **train** step supports hyperparameter tuning with hyperopt by adding
                configurations in the
                |'tuning' section of the 'train' step definition in recipe.yaml|.

   - **evaluate**
      - The **evaluate** step evaluates the model recipe created by the **train** step on
        the test dataset output from the **split** step, computing performance metrics and
        model explanations. Performance metrics are compared against configured thresholds to
        compute a ``model_validation_status``, which indicates whether or not a model is good
        enough to be registered to the MLflow Model Registry by the subsequent **register**
        step. Custom performance metrics are computed according to definitions in
        |steps/custom_metrics.py| and the |'custom_metrics' section of recipe.yaml|. Model
        performance thresholds are defined in the
        |'validation_criteria' section of the 'evaluate' step definition in recipe.yaml|. Model
        performance metrics and explanations are logged to the same MLflow Tracking Run used by
        the **train** step.

   - **register**
      - The **register** step checks the ``model_validation_status`` output of the preceding
        **evaluate** step and, if model validation was successful
        (as indicated by the ``'VALIDATED'`` status), registers the model recipe created by
        the **train** step to the MLflow Model Registry. If the ``model_validation_status`` does
        not indicate that the model passed validation checks (i.e. its value is ``'REJECTED'``),
        the model recipe is not registered to the MLflow Model Registry.
        If the model recipe is registered to the MLflow Model Registry, a
        ``registered_model_version`` is produced containing the model name and the model version.

            .. note::
                The model validation status check can be disabled by specifying
                ``allow_non_validated_model: true`` in the
                |'register' step definition of recipe.yaml|, in which case the model recipe is
                always registered with the MLflow Model Registry when the **register** step is
                executed.

   - **ingest_scoring**
      - The **ingest_scoring** step resolves the dataset specified by the
        |'ingest_scoring' section in recipe.yaml| and converts it to parquet format, leveraging
        the custom dataset parsing code defined in |steps/ingest.py| if necessary.

            .. note::
                If you make changes to the dataset referenced by the **ingest_scoring** step
                (e.g. by adding new records or columns), you must manually re-run the
                **ingest_scoring** step in order to use the updated dataset in the recipe.
                The **ingest_scoring** step does *not* automatically detect changes in the dataset.

   - **predict**
      - The **predict** step uses the ingested dataset for scoring created by the
        **ingest_scoring** step and applies the specified model to the dataset.

            .. note::
                In Databricks, the **predict** step writes the output parquet/delta files to
                DBFS.
"""

import logging
from typing import Any, Optional

from mlflow.recipes.recipe import BaseRecipe
from mlflow.recipes.steps.ingest import IngestStep, IngestScoringStep
from mlflow.recipes.steps.split import SplitStep
from mlflow.recipes.steps.transform import TransformStep
from mlflow.recipes.steps.train import TrainStep
from mlflow.recipes.steps.evaluate import EvaluateStep
from mlflow.recipes.steps.predict import PredictStep
from mlflow.recipes.steps.register import RegisterStep
from mlflow.recipes.step import BaseStep

_logger = logging.getLogger(__name__)


class ClassificationRecipe(BaseRecipe):
    """
    A recipe for developing high-quality classification models. The recipe is designed for
    developing models using scikit-learn and frameworks that integrate with scikit-learn,
    such as the ``XGBClassifier`` API from XGBoost.
    The training recipe contains the following sequential steps:

    **ingest** -> **split** -> **transform** -> **train** -> **evaluate** -> **register**

    while the batch scoring recipe contains this set of sequential steps:

    **ingest_scoring** -> **predict**

    .. code-block:: python
        :caption: Example

        import os
        from mlflow.recipes import Recipe

        os.chdir("~/mlp-classification-template")
        classification_recipe = Recipe(profile="local")
        # Display a visual overview of the recipe graph
        classification_recipe.inspect()
        # Run the full recipe
        classification_recipe.run()
        # Display a summary of results from the 'train' step, including the trained model
        # and associated performance metrics computed from the training & validation datasets
        classification_recipe.inspect(step="train")
        # Display a summary of results from the 'evaluate' step, including model explanations
        # computed from the validation dataset and metrics computed from the test dataset
        classification_recipe.inspect(step="evaluate")
    """

    _RECIPE_STEPS = (
        # Training data ingestion DAG
        IngestStep,
        # Model training DAG
        SplitStep,
        TransformStep,
        TrainStep,
        EvaluateStep,
        RegisterStep,
        # Batch scoring DAG
        IngestScoringStep,
        PredictStep,
    )

    _DEFAULT_STEP_INDEX = _RECIPE_STEPS.index(RegisterStep)

    def _get_step_classes(self):
        return self._RECIPE_STEPS

    def _get_default_step(self) -> BaseStep:
        return self._steps[self._DEFAULT_STEP_INDEX]

    def run(self, step: str = None) -> None:
        """
        Runs the full recipe or a particular recipe step, producing outputs and displaying a
        summary of results upon completion. Step outputs are cached from previous executions, and
        steps are only re-executed if configuration or code changes have been made to the step or
        to any of its dependent steps (e.g. changes to the recipe's ``recipe.yaml`` file or
        ``steps/ingest.py`` file) since the previous execution.

        :param step: String name of the step to run within the classification recipe. The step and
                     its dependencies are executed sequentially. If a step is not specified, the
                     entire recipe is executed. Supported steps, in their order of execution, are:

                     - ``"ingest"``: resolves the dataset specified by the ``data/training`` section
                       in the recipe's configuration file (``recipe.yaml``) and converts it to
                       parquet format.

                     - ``"ingest_scoring"``: resolves the dataset specified by the
                       ``ingest_scoring`` section in the recipe's configuration file
                       (``recipe.yaml``) and converts it to parquet format.

                     - ``"split"``: splits the ingested dataset produced by the **ingest** step into
                       a training dataset for model training, a validation dataset for model
                       performance evaluation & tuning, and a test dataset for model performance
                       evaluation.

                     - ``"transform"``: uses the training dataset created by the **split** step to
                       fit a transformer that performs the transformations defined in the
                       recipe's ``steps/transform.py`` file. Then, applies the transformer to the
                       training dataset and the validation dataset, creating transformed datasets
                       that are used by subsequent steps for estimator training and model
                       performance evaluation.

                     - ``"train"``: uses the transformed training dataset output from the
                       **transform** step to fit an estimator with the type and parameters defined
                       in in the recipe's ``steps/train.py`` file. Then, joins the estimator with
                       the fitted transformer output from the **transform** step to create a model
                       recipe. Finally, evaluates the model recipe against the transformed
                       training and validation datasets to compute performance metrics.

                     - ``"evaluate"``: evaluates the model recipe created by the **train** step
                       on the validation and test dataset outputs from the **split** step, computing
                       performance metrics and model explanations. Then, compares performance
                       metrics against thresholds configured in the recipe's ``recipe.yaml``
                       configuration file to compute a ``model_validation_status``, which indicates
                       whether or not the model is good enough to be registered to the MLflow Model
                       Registry by the subsequent **register** step.

                     - ``"register"``: checks the ``model_validation_status`` output of the
                       preceding **evaluate** step and, if model validation was successful (as
                       indicated by the ``'VALIDATED'`` status), registers the model recipe
                       created by the **train** step to the MLflow Model Registry.

                     - ``"predict"``: uses the ingested dataset for scoring created by the
                       **ingest_scoring** step and applies the specified model to the dataset.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.recipes import Recipe

            os.chdir("~/mlp-classification-template")
            classification_recipe = Recipe(profile="local")
            # Run the 'train' step and preceding steps
            classification_recipe.run(step="train")
            # Run the 'register' step and preceding steps; the 'train' step and all steps
            # prior to 'train' are not re-executed because their outputs are already cached
            classification_recipe.run(step="register")
            # Run all recipe steps; equivalent to running 'register'; no steps are re-executed
            # because the outputs of all steps are already cached
            classification_recipe.run()
        """
        return super().run(step=step)

    def get_artifact(self, artifact_name: str) -> Optional[Any]:
        """
        Reads an artifact from the recipe's outputs. Supported artifact names can be obtained by
        examining the recipe graph visualization displayed by
        :py:func:`ClassificationRecipe.inspect()`.

        :param artifact_name: The string name of the artifact. Supported artifact values are:

                         - ``"ingested_data"``: returns the ingested dataset created in the
                           **ingest** step as a pandas DataFrame.

                         - ``"training_data"``: returns the training dataset created in the
                           **split** step as a pandas DataFrame.

                         - ``"validation_data"``: returns the validation dataset created in the
                           **split** step as a pandas DataFrame.

                         - ``"test_data"``: returns the test dataset created in the **split** step
                           as a pandas DataFrame.

                         - ``"ingested_scoring_data"``: returns the scoring dataset created in the
                           **ingest_scoring** step as a pandas DataFrame.

                         - ``"transformed_training_data"``: returns the transformed training dataset
                           created in the **transform** step as a pandas DataFrame.

                         - ``"transformed_validation_data"``: returns the transformed validation
                           dataset created in the **transform** step as a pandas DataFrame.

                         - ``"model"``: returns the MLflow Model recipe created in the **train**
                           step as a :py:class:`PyFuncModel <mlflow.pyfunc.PyFuncModel>` instance.

                         - ``"transformer"``: returns the scikit-learn transformer created in the
                           **transform** step.

                         - ``"run"``: returns the
                           :py:class:`MLflow Tracking Run <mlflow.entities.Run>` containing the
                           model recipe created in the **train** step and its associated
                           parameters, as well as performance metrics and model explanations created
                           during the **train** and **evaluate** steps.

                         - ``"registered_model_version``": returns the MLflow Model Registry
                           :py:class:`ModelVersion <mlflow.entities.model_registry.ModelVersion>`
                           created by the **register** step.

                         - ``"scored_data"``: returns the scored dataset created in the
                           **predict** step as a pandas DataFrame.

        :return: An object representation of the artifact corresponding to the specified name,
                 as described in the ``artifact_name`` parameter docstring. If the artifact is
                 not present because its corresponding step has not been executed or its output
                 cache has been cleaned, ``None`` is returned.

        .. code-block:: python
            :caption: Example

            import os
            import pandas as pd
            from mlflow.recipes import Recipe
            from mlflow.pyfunc import PyFuncModel

            os.chdir("~/mlp-classification-template")
            classification_recipe = Recipe(profile="local")
            classification_recipe.run()
            train_df: pd.DataFrame = classification_recipe.get_artifact("training_data")
            trained_model: PyFuncModel = classification_recipe.get_artifact("model")
        """
        return super().get_artifact(artifact_name=artifact_name)

    def clean(self, step: str = None) -> None:
        """
        Removes all recipe outputs from the cache, or removes the cached outputs of a particular
        recipe step if specified. After cached outputs are cleaned for a particular step, the
        step will be re-executed in its entirety the next time it is run.

        :param step: String name of the step to clean within the recipe. If not specified,
                     cached outputs are removed for all recipe steps.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.recipes import Recipe

            os.chdir("~/mlp-classification-template")
            classification_recipe = Recipe(profile="local")
            # Run the 'train' step and preceding steps
            classification_recipe.run(step="train")
            # Clean the cache of the 'transform' step
            classification_recipe.clean(step="transform")
            # Run the 'split' step; outputs are still cached because 'split' precedes
            # 'transform' & 'train'
            classification_recipe.run(step="split")
            # Run the 'train' step again; the 'transform' and 'train' steps are re-executed because:
            # 1. the cache of the preceding 'transform' step was cleaned and 2. 'train' occurs after
            # 'transform'. The 'ingest' and 'split' steps are not re-executed because their outputs
            # are still cached
            classification_recipe.run(step="train")
        """
        super().clean(step=step)

    def inspect(self, step: str = None) -> None:
        """
        Displays a visual overview of the recipe graph, or displays a summary of results from
        a particular recipe step if specified. If the specified step has not been executed,
        nothing is displayed.

        :param step: String name of the recipe step for which to display a results summary. If
                     unspecified, a visual overview of the recipe graph is displayed.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.recipes import Recipe

            os.chdir("~/mlp-classification-template")
            classification_recipe = Recipe(profile="local")
            # Display a visual overview of the recipe graph.
            classification_recipe.inspect()
            # Run the 'train' recipe step
            classification_recipe.run(step="train")
            # Display a summary of results from the preceding 'transform' step
            classification_recipe.inspect(step="transform")
        """
        super().inspect(step=step)
