import abc
import logging
import os
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.recipes import dag_help_strings
from mlflow.recipes.artifacts import Artifact
from mlflow.recipes.step import BaseStep, StepClass, StepStatus
from mlflow.recipes.utils import (
    get_recipe_config,
    get_recipe_name,
    get_recipe_root_path,
)
from mlflow.recipes.utils.execution import (
    clean_execution_state,
    get_or_create_base_execution_directory,
    get_step_output_path,
    run_recipe_step,
)
from mlflow.recipes.utils.step import display_html
from mlflow.utils.class_utils import _get_class_from_string

_logger = logging.getLogger(__name__)


class BaseRecipe:
    """
    Base Recipe
    """

    def __init__(self, recipe_root_path: str, profile: str) -> None:
        """
        Recipe base class.

        Args:
            recipe_root_path: String path to the directory under which the recipe template
                such as recipe.yaml, profiles/{profile}.yaml and steps/{step_name}.py are defined.
            profile: String specifying the profile name, with which
                {recipe_root_path}/profiles/{profile}.yaml is read and merged with
                recipe.yaml to generate the configuration to run the recipe.
        """
        self._recipe_root_path = recipe_root_path
        self._run_args = {}
        self._profile = profile
        self._name = get_recipe_name(recipe_root_path)
        # self._steps contains concatenated ordered lists of step objects representing multiple
        # disjoint DAGs. To keep it in sync with the underlying config file, it should be reloaded
        # from config files using self._resolve_recipe_steps() at the beginning of __init__(),
        # run(), and inspect(), and should not reload it elsewhere.
        self._steps = self._resolve_recipe_steps()
        self._recipe = get_recipe_config(self._recipe_root_path, self._profile).get("recipe")

    @property
    def name(self) -> str:
        """Returns the name of the recipe."""
        return self._name

    @property
    def profile(self) -> str:
        """
        Returns the profile under which the recipe and its steps will execute.
        """
        return self._profile

    def run(self, step: Optional[str] = None) -> None:
        """
        Runs a step in the recipe, or the entire recipe if a step is not specified.

        Args:
            step: String name to run a step within the recipe. The step and its dependencies
                will be run sequentially. If a step is not specified, the entire recipe is
                executed.

        Returns:
            None
        """

        # TODO Record performance here.
        self._steps = self._resolve_recipe_steps()
        target_step = self._get_step(step) if step else self._get_default_step()
        last_executed_step = run_recipe_step(
            self._recipe_root_path,
            self._get_subgraph_for_target_step(target_step),
            target_step,
            self._recipe,
        )

        self.inspect(last_executed_step.name)

        # Verify that the step execution succeeded and throw if it didn't.
        last_executed_step_output_directory = get_step_output_path(
            self._recipe_root_path, last_executed_step.name, ""
        )
        last_executed_step_state = last_executed_step.get_execution_state(
            last_executed_step_output_directory
        )
        if last_executed_step_state.status != StepStatus.SUCCEEDED:
            last_step_error_mesg = (
                f"The following error occurred while running step '{last_executed_step}':\n"
                f"{last_executed_step_state.stack_trace}\n"
                f"Last step status: '{last_executed_step_state.status}'\n"
            )
            if step is not None:
                raise MlflowException(
                    f"Failed to run step '{step}' of recipe '{self.name}':\n{last_step_error_mesg}",
                    error_code=BAD_REQUEST,
                )
            else:
                raise MlflowException(
                    f"Failed to run recipe '{self.name}':\n{last_step_error_mesg}",
                    error_code=BAD_REQUEST,
                )

    def inspect(self, step: Optional[str] = None) -> None:
        """
        Displays main output from a step, or a recipe DAG if no step is specified.

        Args:
            step: String name to display a step output within the recipe. If a step is not
                specified, the DAG of the recipe is shown instead.

        Returns:
            None
        """
        self._steps = self._resolve_recipe_steps()
        if not step:
            display_html(html_file_path=self._get_recipe_dag_file())
        else:
            output_directory = get_step_output_path(self._recipe_root_path, step, "")
            self._get_step(step).inspect(output_directory)

    def clean(self, step: Optional[str] = None) -> None:
        """
        Removes the outputs of the specified step from the cache, or removes the cached outputs
        of all steps if no particular step is specified. After cached outputs are cleaned
        for a particular step, the step will be re-executed in its entirety the next time it is
        invoked via ``BaseRecipe.run()``.

        Args:
            step: String name of the step to clean within the recipe. If not specified,
                cached outputs are removed for all recipe steps.
        """
        to_clean = self._steps if not step else [self._get_step(step)]
        clean_execution_state(self._recipe_root_path, to_clean)

    def _get_step(self, step_name) -> BaseStep:
        """Returns a step class object from the recipe."""
        steps = self._steps
        step_names = [s.name for s in steps]
        if step_name not in step_names:
            raise MlflowException(
                f"Step {step_name} not found in recipe. Available steps are {step_names}"
            )
        return self._steps[step_names.index(step_name)]

    def _get_subgraph_for_target_step(self, target_step: BaseStep) -> list[BaseStep]:
        """
        Return a list of step objects representing a connected DAG containing the target_step.
        The returned list should be a sublist of self._steps.
        """
        subgraph = []
        if target_step.step_class == StepClass.UNKNOWN:
            return subgraph
        for step in self._steps:
            if target_step.step_class() == step.step_class():
                subgraph.append(step)
        return subgraph

    @abc.abstractmethod
    def _get_default_step(self) -> BaseStep:
        """
        Defines which step to run if no step is specified.

        Concrete recipe class should implement this method.
        """

    @abc.abstractmethod
    def _get_step_classes(self):
        """
        Returns a list of step classes defined in the recipe.

        Concrete recipe class should implement this method.
        """

    def _get_recipe_dag_file(self) -> str:
        """
        Returns absolute path to the recipe DAG representation HTML file.
        """
        import jinja2

        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))
        recipe_dag_template = j2_env.get_template("resources/recipe_dag_template.html").render(
            {
                "recipe_yaml_help": {
                    "help_string_type": "yaml",
                    "help_string": dag_help_strings.RECIPE_YAML,
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
                "split_step_help": {
                    "help_string": dag_help_strings.SPLIT_STEP,
                    "help_string_type": "text",
                },
                "split_user_code_help": {
                    "help_string": dag_help_strings.SPLIT_USER_CODE,
                    "help_string_type": "python",
                },
                "training_data_help": {
                    "help_string": dag_help_strings.TRAINING_DATA,
                    "help_string_type": "text",
                },
                "validation_data_help": {
                    "help_string": dag_help_strings.VALIDATION_DATA,
                    "help_string_type": "text",
                },
                "test_data_help": {
                    "help_string": dag_help_strings.TEST_DATA,
                    "help_string_type": "text",
                },
                "transform_step_help": {
                    "help_string": dag_help_strings.TRANSFORM_STEP,
                    "help_string_type": "text",
                },
                "transform_user_code_help": {
                    "help_string": dag_help_strings.TRANSFORM_USER_CODE,
                    "help_string_type": "python",
                },
                "fitted_transformer_help": {
                    "help_string": dag_help_strings.FITTED_TRANSFORMER,
                    "help_string_type": "text",
                },
                "transformed_training_and_validation_data_help": {
                    "help_string": dag_help_strings.TRANSFORMED_TRAINING_AND_VALIDATION_DATA,
                    "help_string_type": "text",
                },
                "train_step_help": {
                    "help_string": dag_help_strings.TRAIN_STEP,
                    "help_string_type": "text",
                },
                "train_user_code_help": {
                    "help_string": dag_help_strings.TRAIN_USER_CODE,
                    "help_string_type": "python",
                },
                "fitted_model_help": {
                    "help_string": dag_help_strings.FITTED_MODEL,
                    "help_string_type": "text",
                },
                "mlflow_run_help": {
                    "help_string": dag_help_strings.MLFLOW_RUN,
                    "help_string_type": "text",
                },
                "predicted_training_data_help": {
                    "help_string": dag_help_strings.PREDICTED_TRAINING_DATA,
                    "help_string_type": "text",
                },
                "custom_metrics_user_code_help": {
                    "help_string": dag_help_strings.CUSTOM_METRICS_USER_CODE,
                    "help_string_type": "python",
                },
                "evaluate_step_help": {
                    "help_string": dag_help_strings.EVALUATE_STEP,
                    "help_string_type": "text",
                },
                "model_validation_status_help": {
                    "help_string": dag_help_strings.MODEL_VALIDATION_STATUS,
                    "help_string_type": "text",
                },
                "register_step_help": {
                    "help_string": dag_help_strings.REGISTER_STEP,
                    "help_string_type": "text",
                },
                "registered_model_version_help": {
                    "help_string": dag_help_strings.REGISTERED_MODEL_VERSION,
                    "help_string_type": "text",
                },
                "ingest_scoring_step_help": {
                    "help_string": dag_help_strings.INGEST_SCORING_STEP,
                    "help_string_type": "text",
                },
                "ingested_scoring_data_help": {
                    "help_string": dag_help_strings.INGESTED_SCORING_DATA,
                    "help_string_type": "text",
                },
                "predict_step_help": {
                    "help_string": dag_help_strings.PREDICT_STEP,
                    "help_string_type": "text",
                },
                "scored_data_help": {
                    "help_string": dag_help_strings.SCORED_DATA,
                    "help_string_type": "text",
                },
            }
        )

        recipe_dag_file = os.path.join(
            get_or_create_base_execution_directory(self._recipe_root_path), "recipe_dag.html"
        )
        with open(recipe_dag_file, "w") as f:
            f.write(recipe_dag_template)

        return recipe_dag_file

    def _resolve_recipe_steps(self) -> list[BaseStep]:
        """
        Constructs and returns all recipe step objects from the recipe configuration.
        """
        recipe_config = get_recipe_config(self._recipe_root_path, self._profile)
        recipe_config["profile"] = self.profile
        return [
            s.from_recipe_config(recipe_config, self._recipe_root_path)
            for s in self._get_step_classes()
        ]

    def get_artifact(self, artifact_name: str):
        """
        Read an artifact from recipe output. artifact names can be obtained from
        `Recipe.inspect()` or `Recipe.run()` output.

        Returns None if the specified artifact is not found.
        Raise an error if the artifact is not supported.
        """
        return self._get_artifact(artifact_name).load()

    def _get_artifact(self, artifact_name: str) -> Artifact:
        """
        Read an Artifact object from recipe output. artifact names can be obtained
        from `Recipe.inspect()` or `Recipe.run()` output.

        Returns None if the specified artifact is not found.
        Raise an error if the artifact is not supported.
        """
        for step in self._steps:
            for artifact in step.get_artifacts():
                if artifact.name() == artifact_name:
                    return artifact
        raise MlflowException(
            f"The artifact with name '{artifact_name}' is not supported.",
            error_code=INVALID_PARAMETER_VALUE,
        )


class Recipe:
    """
    A factory class that creates an instance of a recipe for a particular ML problem
    (e.g. regression, classification) or MLOps task (e.g. batch scoring) based on the current
    working directory and supplied configuration.

    .. code-block:: python
        :caption: Example

        import os
        from mlflow.recipes import Recipe

        os.chdir("~/recipes-regression-template")
        regression_recipe = Recipe(profile="local")
        regression_recipe.run(step="train")
    """

    def __new__(cls, profile: str):
        """
        Creates an instance of an MLflow Recipe for a particular ML problem or MLOps task based
        on the current working directory and supplied configuration. The current working directory
        must be the root directory of an MLflow Recipe repository or a subdirectory of an
        MLflow Recipe repository.

        Args:
            profile: The name of the profile to use for configuring the problem-specific or
                task-specific recipe. Profiles customize the configuration of
                one or more recipe steps, and recipe executions with different profiles
                often produce different results.

        Returns:
            A recipe for a particular ML problem or MLOps task. For example, an instance of
            `RegressionRecipe <https://github.com/mlflow/recipes-regression-template>`_
            for regression problems.

        .. code-block:: python

            import os
            from mlflow.recipes import Recipe

            os.chdir("~/recipes-regression-template")
            regression_recipe = Recipe(profile="local")
            regression_recipe.run(step="train")
        """
        if not profile:
            raise MlflowException(
                "A profile name must be provided to construct a valid Recipe object.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None

        recipe_root_path = get_recipe_root_path()
        if " " in recipe_root_path:
            raise MlflowException(
                message=(
                    "Recipe directory path cannot contain spaces. Please move or rename your "
                    f"recipe directory. Current path: {recipe_root_path}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            ) from None

        recipe_config = get_recipe_config(recipe_root_path=recipe_root_path, profile=profile)
        recipe = recipe_config.get("recipe")
        if recipe is None:
            raise MlflowException(
                "The `recipe` property needs to be defined in the `recipe.yaml` file. "
                "For example: `recipe: regression/v1`",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None
        recipe_path = recipe.replace("/", ".").replace("@", ".")
        class_name = f"mlflow.recipes.{recipe_path}.RecipeImpl"

        try:
            recipe_class_module = _get_class_from_string(class_name)
        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                raise MlflowException(
                    f"Failed to find Recipe {class_name}."
                    f"Please check the correctness of the recipe template setting: {recipe}",
                    error_code=INVALID_PARAMETER_VALUE,
                ) from None
            else:
                raise MlflowException(
                    f"Failed to construct Recipe {class_name}",
                    error_code=INTERNAL_ERROR,
                ) from e

        recipe_name = get_recipe_name(recipe_root_path)
        _logger.info(f"Creating MLflow Recipe '{recipe_name}' with profile: '{profile}'")
        return recipe_class_module(recipe_root_path, profile)
