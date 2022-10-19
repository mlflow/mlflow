"""
.. _mlflow-classification-pipeline:

The MLflow Classification Pipeline is an MLflow Pipeline for developing classification models.
It is designed for developing models using scikit-learn and frameworks that integrate with
scikit-learn, such as the ``XGBClassifier`` API from XGBoost. The :py:class:`ClassificationPipeline
API Documentation <ClassificationPipeline>` provides instructions for executing the pipeline and
inspecting its results.

The training pipeline contains the following sequential steps:

**ingest** -> **split** -> **transform** -> **train** -> **evaluate** -> **register**

The batch scoring pipeline contains the following sequential steps:

**ingest_scoring** -> **predict**

The pipeline steps are defined as follows:

   - **ingest**
      - The **ingest** step resolves the dataset specified by the |'data' section in pipeline.yaml|
        and converts it to parquet format, leveraging the custom dataset parsing code defined in
        |steps/ingest.py| if necessary. Subsequent steps convert this dataset into training,
        validation, & test sets and use them to develop a model.

        .. note::
            If you make changes to the dataset referenced by the **ingest** step (e.g. by adding
            new records or columns), you must manually re-run the **ingest** step in order to
            use the updated dataset in the pipeline. The **ingest** step does *not* automatically
            detect changes in the dataset.

   .. _mlflow-classification-pipeline-split-step:

   - **split**
      - The **split** step splits the ingested dataset produced by the **ingest** step into
        a training dataset for model training, a validation dataset for model performance
        evaluation  & tuning, and a test dataset for model performance evaluation. The fraction
        of records allocated to each dataset is defined by the ``split_ratios`` attribute of the
        |'split' step definition in pipeline.yaml|. The **split** step also preprocesses the
        datasets using logic defined in |steps/split.py|. Subsequent steps use these datasets
        to develop a model and measure its performance.

   - **transform**
      - The **transform** step uses the training dataset created by **split** to fit
        a transformer that performs the transformations defined in |steps/transform.py|. The
        transformer is then applied to the training dataset and the validation dataset, creating
        transformed datasets that are used by subsequent steps for estimator training and model
        performance evaluation.

   .. _mlflow-classification-pipeline-train-step:

   - **train**
      - The **train** step uses the transformed training dataset output from the **transform**
        step to fit an estimator with the type and parameters defined in |steps/train.py|. The
        estimator is then joined with the fitted transformer output from the **transform** step
        to create a model pipeline. Finally, this model pipeline is evaluated against the
        transformed training and validation datasets to compute performance metrics; custom
        metrics are computed according to definitions in |steps/custom_metrics.py| and the
        |'metrics' section of pipeline.yaml|. The model pipeline and its associated parameters,
        performance metrics, and lineage information are logged to MLflow Tracking, producing
        an MLflow Run.

            .. note::
                The **train** step supports hyperparameter tuning with hyperopt by adding 
                configurations in the 
                |'tuning' section of the 'train' step definition in pipeline.yaml|. 

   - **evaluate**
      - The **evaluate** step evaluates the model pipeline created by the **train** step on
        the test dataset output from the **split** step, computing performance metrics and
        model explanations. Performance metrics are compared against configured thresholds to
        compute a ``model_validation_status``, which indicates whether or not a model is good
        enough to be registered to the MLflow Model Registry by the subsequent **register**
        step. Custom performance metrics are computed according to definitions in
        |steps/custom_metrics.py| and the |'metrics' section of pipeline.yaml|. Model
        performance thresholds are defined in the
        |'validation_criteria' section of the 'evaluate' step definition in pipeline.yaml|. Model
        performance metrics and explanations are logged to the same MLflow Tracking Run used by
        the **train** step.

   - **register**
      - The **register** step checks the ``model_validation_status`` output of the preceding
        **evaluate** step and, if model validation was successful
        (as indicated by the ``'VALIDATED'`` status), registers the model pipeline created by
        the **train** step to the MLflow Model Registry. If the ``model_validation_status`` does
        not indicate that the model passed validation checks (i.e. its value is ``'REJECTED'``),
        the model pipeline is not registered to the MLflow Model Registry.
        If the model pipeline is registered to the MLflow Model Registry, a
        ``registered_model_version`` is produced containing the model name and the model version.

            .. note::
                The model validation status check can be disabled by specifying
                ``allow_non_validated_model: true`` in the
                |'register' step definition of pipeline.yaml|, in which case the model pipeline is
                always registered with the MLflow Model Registry when the **register** step is
                executed.

   - **ingest_scoring**
      - The **ingest_scoring** step resolves the dataset specified by the 
        |'data/scoring' section in pipeline.yaml| and converts it to parquet format, leveraging 
        the custom dataset parsing code defined in |steps/ingest.py| if necessary. 
    
            .. note::
                If you make changes to the dataset referenced by the **ingest_scoring** step 
                (e.g. by adding new records or columns), you must manually re-run the 
                **ingest_scoring** step in order to use the updated dataset in the pipeline. 
                The **ingest_scoring** step does *not* automatically detect changes in the dataset.
    
   - **predict**
      - The **predict** step uses the ingested dataset for scoring created by the
        **ingest_scoring** step and applies the specified model to the dataset.

            .. note::
                In Databricks, the **predict** step writes the output parquet/delta files to
                DBFS.
"""

import logging

from mlflow.pipelines.pipeline import _BasePipeline
from mlflow.pipelines.steps.ingest import IngestStep, IngestScoringStep
from mlflow.pipelines.steps.split import SplitStep
from mlflow.pipelines.steps.transform import TransformStep
from mlflow.pipelines.steps.train import TrainStep
from mlflow.pipelines.steps.evaluate import EvaluateStep
from mlflow.pipelines.steps.predict import PredictStep
from mlflow.pipelines.steps.register import RegisterStep
from mlflow.pipelines.step import BaseStep
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental
class ClassificationPipeline(_BasePipeline):
    """
    A pipeline for developing high-quality classification models. The pipeline is designed for
    developing models using scikit-learn and frameworks that integrate with scikit-learn,
    such as the ``XGBClassifier`` API from XGBoost.
    The training pipeline contains the following sequential steps:

    **ingest** -> **split** -> **transform** -> **train** -> **evaluate** -> **register**

    while the batch scoring pipeline contains this set of sequential steps:

    **ingest_scoring** -> **predict**

    .. code-block:: python
        :caption: Example

        import os
        from mlflow.pipelines import Pipeline

        os.chdir("~/mlp-classification-template")
        classification_pipeline = Pipeline(profile="local")
        # Display a visual overview of the pipeline graph
        classification_pipeline.inspect()
        # Run the full pipeline
        classification_pipeline.run()
        # Display a summary of results from the 'train' step, including the trained model
        # and associated performance metrics computed from the training & validation datasets
        classification_pipeline.inspect(step="train")
        # Display a summary of results from the 'evaluate' step, including model explanations
        # computed from the validation dataset and metrics computed from the test dataset
        classification_pipeline.inspect(step="evaluate")
    """

    _PIPELINE_STEPS = (
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

    _DEFAULT_STEP_INDEX = _PIPELINE_STEPS.index(RegisterStep)

    def _get_step_classes(self):
        return self._PIPELINE_STEPS

    def _get_default_step(self) -> BaseStep:
        return self._steps[self._DEFAULT_STEP_INDEX]
