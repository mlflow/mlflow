import abc
import logging

from mlflow.exceptions import MlflowException
from mlflow.pipelines.step import BaseStep, StepStatus
from mlflow.pipelines.utils import (
    get_pipeline_config,
    get_pipeline_name,
    get_pipeline_root_path,
)
from mlflow.pipelines.utils.execution import (
    clean_execution_state,
    run_pipeline_step,
    get_step_output_path,
)
from mlflow.pipelines.utils.step import display_html
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR, BAD_REQUEST
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string
from typing import List

_logger = logging.getLogger(__name__)


@experimental
class _BasePipeline:
    """
    Base Pipeline
    """

    @experimental
    def __init__(self, pipeline_root_path: str, profile: str) -> None:
        """
        Pipeline base class.

        :param pipeline_root_path: String path to the directory under which the pipeline template
                                   such as pipeline.yaml, profiles/{profile}.yaml and
                                   steps/{step_name}.py are defined.
        :param profile: String specifying the profile name, with which
                        {pipeline_root_path}/profiles/{profile}.yaml is read and merged with
                        pipeline.yaml to generate the configuration to run the pipeline.
        """
        self._pipeline_root_path = pipeline_root_path
        self._profile = profile
        self._name = get_pipeline_name(pipeline_root_path)
        self._steps = self._resolve_pipeline_steps()

    @experimental
    @property
    def name(self) -> str:
        """Returns the name of the pipeline."""
        return self._name

    @experimental
    @property
    def profile(self) -> str:
        """
        Returns the profile under which the pipeline and its steps will execute.
        """
        return self._profile

    @experimental
    def run(self, step: str = None) -> None:
        """
        Runs a step in the pipeline, or the entire pipeline if a step is not specified.

        :param step: String name to run a step within the pipeline. The step and its dependencies
                     will be run sequentially. If a step is not specified, the entire pipeline is
                     executed.
        :return: None
        """
        # TODO Record performance here.
        # Always resolve the steps to load latest step modules before execution.
        self._steps = self._resolve_pipeline_steps()
        last_executed_step = run_pipeline_step(
            self._pipeline_root_path,
            self._steps,
            # Runs the last step of the pipeline if no step is specified.
            self._get_step(step) if step else self._steps[-1],
        )

        self.inspect(last_executed_step.name)

        # Verify that the step execution succeeded and throw if it didn't.
        last_executed_step_output_directory = get_step_output_path(
            self._pipeline_root_path, last_executed_step.name, ""
        )
        last_executed_step_status = last_executed_step.get_execution_state(
            last_executed_step_output_directory
        ).status
        if last_executed_step_status != StepStatus.SUCCEEDED:
            if step is not None:
                raise MlflowException(
                    f"Failed to run step '{step}' of pipeline '{self.name}'."
                    f" An error was encountered while running step '{last_executed_step.name}'.",
                    error_code=BAD_REQUEST,
                )
            else:
                raise MlflowException(
                    f"Failed to run pipeline '{self.name}'."
                    f" An error was encountered while running step '{last_executed_step.name}'.",
                    error_code=BAD_REQUEST,
                )

    @experimental
    def inspect(self, step: str = None) -> None:
        """
        Displays main output from a step, or a pipeline DAG if no step is specified.

        :param step: String name to display a step output within the pipeline. If a step is not
                     specified, the DAG of the pipeline is shown instead.
        :return: None
        """
        if not step:
            display_html(html_file_path=self._get_pipeline_dag_file())
        else:
            output_directory = get_step_output_path(self._pipeline_root_path, step, "")
            self._get_step(step).inspect(output_directory)

    @experimental
    def clean(self, step: str = None) -> None:
        """
        Removes the outputs of the specified step from the cache, or removes the cached outputs
        of all steps if no particular step is specified. After cached outputs are cleaned
        for a particular step, the step will be re-executed in its entirety the next time it is
        invoked via ``_BasePipeline.run()``.

        :param step: String name of the step to clean within the pipeline. If not specified,
                     cached outputs are removed for all pipeline steps.
        """
        to_clean = self._steps if not step else [self._get_step(step)]
        clean_execution_state(self._pipeline_root_path, to_clean)

    def _get_step(self, step_name) -> BaseStep:
        """Returns a step class object from the pipeline."""
        steps = self._steps or self._resolve_pipeline_steps()
        step_names = [s.name for s in steps]
        if step_name not in step_names:
            raise MlflowException(
                f"Step {step_name} not found in pipeline. Available steps are {step_names}"
            )
        return self._steps[step_names.index(step_name)]

    @experimental
    @abc.abstractmethod
    def _get_step_classes(self) -> List[BaseStep]:
        """
        Returns a list of step classes defined in the pipeline.

        Concrete pipeline class should implement this method.
        """
        pass

    @experimental
    @abc.abstractmethod
    def _get_pipeline_dag_file(self) -> str:
        """
        Returns absolute path to the pipeline DAG representation HTML file.

        Concrete pipeline class should implement this method.
        """
        pass

    def _resolve_pipeline_steps(self) -> List[BaseStep]:
        """
        Constructs and returns all pipeline step objects from the pipeline configuration.
        """
        pipeline_config = get_pipeline_config(self._pipeline_root_path, self._profile)
        return [
            s.from_pipeline_config(pipeline_config, self._pipeline_root_path)
            for s in self._get_step_classes()
        ]

    @experimental
    @abc.abstractmethod
    def get_artifact(self, artifact_name: str):
        """
        Read an artifact from pipeline output. artifact names can be obtained from
        `Pipeline.inspect()` or `Pipeline.run()` output.

        Returns None if the specified artifact is not found.
        Raise an error if the artifact is not supported.
        """
        pass


from mlflow.pipelines.regression.v1.pipeline import RegressionPipeline


@experimental
class Pipeline:
    """
    A factory class that creates an instance of a pipeline for a particular ML problem
    (e.g. regression, classification) or MLOps task (e.g. batch scoring) based on the current
    working directory and supplied configuration.

    .. code-block:: python
        :caption: Example

        import os
        from mlflow.pipelines import Pipeline

        os.chdir("~/mlp-regression-template")
        regression_pipeline = Pipeline(profile="local")
        regression_pipeline.run(step="train")
    """

    @experimental
    def __new__(cls, profile: str) -> RegressionPipeline:
        """
        Creates an instance of an MLflow Pipeline for a particular ML problem or MLOps task based
        on the current working directory and supplied configuration. The current working directory
        must be the root directory of an MLflow Pipeline repository or a subdirectory of an
        MLflow Pipeline repository.

        :param profile: The name of the profile to use for configuring the problem-specific or
                        task-specific pipeline. Profiles customize the configuration of
                        one or more pipeline steps, and pipeline executions with different profiles
                        often produce different results.
        :return: A pipeline for a particular ML problem or MLOps task. For example, an instance of
                 :py:class:`RegressionPipeline
                 <mlflow.pipelines.regression.v1.pipeline.RegressionPipeline>`
                 for regression problems.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.pipelines import Pipeline

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline(profile="local")
            regression_pipeline.run(step="train")
        """
        if not profile:
            raise MlflowException(
                "A profile name must be provided to construct a valid Pipeline object.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None

        pipeline_root_path = get_pipeline_root_path()
        if " " in pipeline_root_path:
            raise MlflowException(
                message=(
                    "Pipeline directory path cannot contain spaces. Please move or rename your"
                    f" pipeline directory. Current path: {pipeline_root_path}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            ) from None

        pipeline_config = get_pipeline_config(
            pipeline_root_path=pipeline_root_path, profile=profile
        )
        template = pipeline_config.get("template")
        if template is None:
            raise MlflowException(
                "The `template` property needs to be defined in the `pipeline.yaml` file."
                "For example: `template: regression/v1`",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None
        template_path = template.replace("/", ".").replace("@", ".")
        class_name = f"mlflow.pipelines.{template_path}.PipelineImpl"

        try:
            pipeline_class_module = _get_class_from_string(class_name)
        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                raise MlflowException(
                    f"Failed to find Pipeline {class_name}."
                    f"Please check the correctness of the pipeline template setting: {template}",
                    error_code=INVALID_PARAMETER_VALUE,
                ) from None
            else:
                raise MlflowException(
                    f"Failed to construct Pipeline {class_name}. Error: {repr(e)}",
                    error_code=INTERNAL_ERROR,
                ) from None

        pipeline_name = get_pipeline_name(pipeline_root_path)
        _logger.info(f"Creating MLflow Pipeline '{pipeline_name}' with profile: '{profile}'")
        return pipeline_class_module(pipeline_root_path, profile)
