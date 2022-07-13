"""
.. _mlflow-batch-scoring-pipeline:

The MLflow Batch Scoring Pipeline is an MLflow Pipeline for developing high-quality regression models.
It is designed for developing models using scikit-learn and frameworks that integrate with
scikit-learn, such as the ``XGBRegressor`` API from XGBoost. The corresponding pipeline
template repository is available at https://github.com/mlflow/mlp-batch-scoring-template, and the
:py:class:` BatchScoringPipeline API Documentation <BatchScoringPipeline>` provides instructions for
executing the pipeline and inspecting its results.

The pipeline contains the following sequential steps:

**ingest** -> **data_clean** -> **predict**

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

   .. _mlflow-batch-scoring-pipeline-predict-step:

   - **predict**
      - The **predict** step ...

.. |steps/ingest.py| replace:: `steps/ingest.py <https://github.com/mlflow/mlp-regression-template/blob/main/steps/ingest.py>`__
"""

import os
import logging

import mlflow.pipelines.batch_scoring.v1.dag_help_strings as dag_help_strings
from mlflow.tracking.client import MlflowClient
from mlflow.pipelines.pipeline import _BasePipeline
from mlflow.pipelines.steps.ingest import IngestStep
from mlflow.pipelines.steps.data_clean import (
    _CLEANED_OUTPUT_FILE_NAME,
    DataCleanStep
)
from mlflow.pipelines.steps.predict import (
    _SCORED_OUTPUT_FILE_NAME,
    PredictStep
)
from mlflow.pipelines.step import BaseStep
from typing import List, Any, Optional
from mlflow.pipelines.utils import get_pipeline_root_path
from mlflow.pipelines.utils.execution import get_or_create_base_execution_directory
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental
class BatchScoringPipeline(_BasePipeline):
    """
    A pipeline for developing high-quality regression models. The pipeline is designed for
    developing models using scikit-learn and frameworks that integrate with scikit-learn,
    such as the ``XGBRegressor`` API from XGBoost. The corresponding pipeline
    template repository is available at https://github.com/mlflow/mlp-regression-template.
    The pipeline contains the following sequential steps:

    **ingest** -> **split** -> **transform** -> **train** -> **evaluate** -> **register**

    .. code-block:: python
        :caption: Example

        import os
        from mlflow.pipelines import Pipeline

        os.chdir("~/mlp-regression-template")
        regression_pipeline = Pipeline(profile="local")
        # Display a visual overview of the pipeline graph
        regression_pipeline.inspect()
        # Run the full pipeline
        regression_pipeline.run()
        # Display a summary of results from the 'train' step, including the trained model
        # and associated performance metrics computed from the training & validation datasets
        regression_pipeline.inspect(step="train")
        # Display a summary of results from the 'evaluate' step, including model explanations
        # computed from the validation dataset and metrics computed from the test dataset
        regression_pipeline.inspect(step="evaluate")
    """

    _PIPELINE_STEPS = (IngestStep, DataCleanStep, PredictStep)

    def _get_step_classes(self) -> List[BaseStep]:
        return self._PIPELINE_STEPS

    def _get_pipeline_dag_file(self) -> str:
        import jinja2

        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))
        pipeline_dag_template = j2_env.get_template("resources/pipeline_dag_template.html").render(
            {
                "pipeline_yaml_help": {
                    "help_string_type": "yaml",
                    "help_string": dag_help_strings.PIPELINE_YAML,
                },
                "ingest_step_help": {
                    "help_string": dag_help_strings.INGEST_STEP,
                    "help_string_type": "text",
                },
                "ingest_user_code_help": {
                    "help_string": dag_help_strings.INGEST_USER_CODE,
                    "help_string_type": "python",
                },
                "ingested_data_help": {
                    "help_string": dag_help_strings.INGESTED_DATA,
                    "help_string_type": "text",
                },
                "data_clean_step_help": {
                    "help_string": dag_help_strings.DATA_CLEAN_STEP,
                    "help_string_type": "text",
                },
                "data_clean_user_code_help": {
                    "help_string": dag_help_strings.DATA_CLEAN_USER_CODE,
                    "help_string_type": "python",
                },
                "predict_step_help": {
                    "help_string": dag_help_strings.PREDICT_STEP,
                    "help_string_type": "text",
                },
            }
        )

        pipeline_dag_file = os.path.join(
            get_or_create_base_execution_directory(self._pipeline_root_path), "pipeline_dag.html"
        )
        with open(pipeline_dag_file, "w") as f:
            f.write(pipeline_dag_template)

        return pipeline_dag_file

    def run(self, step: str = None) -> None:
        """
        Runs the full pipeline or a particular pipeline step, producing outputs and displaying a
        summary of results upon completion. Step outputs are cached from previous executions, and
        steps are only re-executed if configuration or code changes have been made to the step or
        to any of its dependent steps (e.g. changes to the pipeline's ``pipeline.yaml`` file or
        ``steps/ingest.py`` file) since the previous execution.

        :param step: String name of the step to run within the regression pipeline. The step and
                     its dependencies are executed sequentially. If a step is not specified, the
                     entire pipeline is executed. Supported steps, in their order of execution, are:

                     - ``"ingest"``: resolves the dataset specified by the ``data`` section in the
                       pipeline's configuration file (``pipeline.yaml``) and converts it to parquet
                       format.

                     - ``"data_clean"``: cleans the ingested dataset produced by the **ingest** step into
                       a cleaned dataset for batch scoring.

                     - ``"predict"``: uses the cleaned dataset created by the **data_clean** step and
                       applies the specified model to the dataset.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.pipelines import Pipeline

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline(profile="local")
            # Run the 'train' step and preceding steps
            regression_pipeline.run(step="train")
            # Run the 'register' step and preceding steps; the 'train' step and all steps
            # prior to 'train' are not re-executed because their outputs are already cached
            regression_pipeline.run(step="register")
            # Run all pipeline steps; equivalent to running 'register'; no steps are re-executed
            # because the outputs of all steps are already cached
            regression_pipeline.run()
        """
        return super().run(step=step)

    @experimental
    def get_artifact(self, artifact_name: str) -> Optional[Any]:
        """
        Reads an artifact from the pipeline's outputs. Supported artifact names can be obtained by
        examining the pipeline graph visualization displayed by
        :py:func:`RegressionPipeline.inspect()`.

        :param artifact_name: The string name of the artifact. Supported artifact values are:

                         - ``"ingested_data"``: returns the ingested dataset created in the
                           **ingest** step as a pandas DataFrame.

                         - ``"cleaned_data"``: returns the cleaned dataset created in the
                           **data_clean** step as a pandas DataFrame.

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
            from mlflow.pipelines import Pipeline
            from mlflow.pyfunc import PyFuncModel

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline(profile="local")
            regression_pipeline.run()
            train_df: pd.DataFrame = regression_pipeline.get_artifact("training_data")
            trained_model: PyFuncModel = regression_pipeline.get_artifact("model")
        """
        import mlflow.pyfunc

        ingest_step, data_clean_step, predict_step = self._steps

        ingest_output_dir = get_step_output_path(self._pipeline_root_path, ingest_step.name, "")
        data_clean_output_dir = get_step_output_path(self._pipeline_root_path, data_clean_step.name, "")
        predict_output_dir = get_step_output_path(
            self._pipeline_root_path, predict_step.name, ""
        )

        def log_artifact_not_found_warning(artifact_name, step_name):
            _logger.warning(
                f"The artifact with name '{artifact_name}' was not found."
                f" Re-run the '{step_name}' step to generate it."
            )

        pipeline_root_path = get_pipeline_root_path()

        def read_dataframe(artifact_name, output_dir, file_name, step_name):
            import pandas as pd

            data_path = os.path.join(output_dir, file_name)
            if os.path.exists(data_path):
                return pd.read_parquet(data_path)
            else:
                log_artifact_not_found_warning(artifact_name, step_name)
                return None

        if artifact_name == "ingested_data":
            return read_dataframe(
                "ingested_data",
                ingest_output_dir,
                IngestStep._DATASET_OUTPUT_NAME,
                ingest_step.name,
            )

        elif artifact_name == "cleaned_data":
            return read_dataframe(
                "cleaned_data", data_clean_output_dir, _CLEANED_OUTPUT_FILE_NAME, data_clean_step.name
            )

        elif artifact_name == "scored_data":
            return read_dataframe(
                "scored_data", predict_output_dir, _SCORED_OUTPUT_FILE_NAME, predict_step.name
            )

        else:
            raise MlflowException(
                f"The artifact with name '{artifact_name}' is not supported.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def clean(self, step: str = None) -> None:
        """
        Removes all pipeline outputs from the cache, or removes the cached outputs of a particular
        pipeline step if specified. After cached outputs are cleaned for a particular step, the
        step will be re-executed in its entirety the next time it is run.

        :param step: String name of the step to clean within the pipeline. If not specified,
                     cached outputs are removed for all pipeline steps.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.pipelines import Pipeline

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline(profile="local")
            # Run the 'train' step and preceding steps
            regression_pipeline.run(step="train")
            # Clean the cache of the 'transform' step
            regression_pipeline.clean(step="transform")
            # Run the 'split' step; outputs are still cached because 'split' precedes
            # 'transform' & 'train'
            regression_pipeline.run(step="split")
            # Run the 'train' step again; the 'transform' and 'train' steps are re-executed because:
            # 1. the cache of the preceding 'transform' step was cleaned and 2. 'train' occurs after
            # 'transform'. The 'ingest' and 'split' steps are not re-executed because their outputs
            # are still cached
            regression_pipeline.run(step="train")
        """
        super().clean(step=step)

    def inspect(self, step: str = None) -> None:
        """
        Displays a visual overview of the pipeline graph, or displays a summary of results from
        a particular pipeline step if specified. If the specified step has not been executed,
        nothing is displayed.

        :param step: String name of the pipeline step for which to display a results summary. If
                     unspecified, a visual overview of the pipeline graph is displayed.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.pipelines import Pipeline

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline(profile="local")
            # Display a visual overview of the pipeline graph.
            regression_pipeline.inspect()
            # Run the 'train' pipeline step
            regression_pipeline.run(step="train")
            # Display a summary of results from the preceding 'transform' step
            regression_pipeline.inspect(step="transform")
        """
        super().inspect(step=step)
