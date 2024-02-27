import hashlib
import logging
import os
import pathlib
import re
import shutil
from typing import Dict, List

from mlflow.environment_variables import (
    MLFLOW_RECIPES_EXECUTION_DIRECTORY,
    MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME,
)
from mlflow.recipes.step import BaseStep, StepStatus
from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd

_logger = logging.getLogger(__name__)

_STEPS_SUBDIRECTORY_NAME = "steps"
_STEP_OUTPUTS_SUBDIRECTORY_NAME = "outputs"
_STEP_CONF_YAML_NAME = "conf.yaml"


def run_recipe_step(
    recipe_root_path: str,
    recipe_steps: List[BaseStep],
    target_step: BaseStep,
    template: str,
) -> BaseStep:
    """
    Runs the specified step in the specified recipe, as well as all dependent steps.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        recipe_steps: A list of all the steps contained in the subgraph of the specified
            recipe that contains the target_step. Recipe steps must be provided in the order
            that they are intended to be executed.
        target_step: The step to run.
        template: The template to use when selecting a Makefile to load. If the template is
            invalid, an exception is thrown.

    Returns:
        The last step that successfully completed during the recipe execution. If execution
        was successful, this always corresponds to the supplied target step. If execution was
        unsuccessful, this corresponds to the step that failed.
    """
    target_step_index = recipe_steps.index(target_step)
    execution_dir_path = _get_or_create_execution_directory(
        recipe_root_path, recipe_steps, template
    )

    def get_execution_state(step):
        return step.get_execution_state(
            output_directory=_get_step_output_directory_path(
                execution_directory_path=execution_dir_path,
                step_name=step.name,
            )
        )

    # Check the previous execution state of the target step and all of its
    # dependencies. If any of these steps previously failed, clear its execution
    # state to ensure that the step is run again during the upcoming execution
    clean_execution_state(
        recipe_root_path=recipe_root_path,
        recipe_steps=[
            step
            for step in recipe_steps[: target_step_index + 1]
            if get_execution_state(step).status != StepStatus.SUCCEEDED
        ],
    )

    _write_updated_step_confs(
        recipe_steps=recipe_steps,
        execution_directory_path=execution_dir_path,
    )

    # Aggregate step-specific environment variables into a single environment dictionary
    # that is passed to the Make subprocess. In the future, steps with different environments
    # should be isolated in different subprocesses
    make_env = {
        # Include target step name in the environment variable set
        MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME.name: target_step.name,
    }
    for step in recipe_steps:
        make_env.update(step.environment)
    # Use Make to run the target step and all of its dependencies
    _run_make(
        execution_directory_path=execution_dir_path,
        rule_name=target_step.name,
        extra_env=make_env,
        recipe_steps=recipe_steps,
    )

    # Identify the last step that was executed, excluding steps that are downstream of the
    # specified target step
    last_executed_step = recipe_steps[0]
    last_executed_step_state = get_execution_state(last_executed_step)
    for step in recipe_steps[1 : target_step_index + 1]:
        step_state = get_execution_state(step)
        if step_state.last_updated_timestamp >= last_executed_step_state.last_updated_timestamp:
            last_executed_step = step
            last_executed_step_state = step_state

    # Check the previous execution state of all recipe steps downstream of the last executed step.
    # If any of these steps was last executed before the target step or another step upstream of the
    # target step, this indicates that downstream steps are out of date and need to be cleared
    clean_execution_state(
        recipe_root_path=recipe_root_path,
        recipe_steps=[
            step
            for step in recipe_steps[recipe_steps.index(last_executed_step) :]
            if get_execution_state(step).last_updated_timestamp
            < last_executed_step_state.last_updated_timestamp
        ],
    )

    return last_executed_step


def clean_execution_state(recipe_root_path: str, recipe_steps: List[BaseStep]) -> None:
    """
    Removes all execution state for the specified recipe steps from the associated execution
    directory on the local filesystem. This method does *not* remove other execution results, such
    as content logged to MLflow Tracking.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        recipe_steps: The recipe steps for which to remove execution state.
    """
    execution_dir_path = get_or_create_base_execution_directory(recipe_root_path=recipe_root_path)
    for step in recipe_steps:
        step_outputs_path = _get_step_output_directory_path(
            execution_directory_path=execution_dir_path,
            step_name=step.name,
        )
        if os.path.exists(step_outputs_path):
            shutil.rmtree(step_outputs_path)
        os.makedirs(step_outputs_path)


def get_step_output_path(recipe_root_path: str, step_name: str, relative_path: str) -> str:
    """
    Obtains the absolute path of the specified step output on the local filesystem. Does
    not check the existence of the output.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        step_name: The name of the recipe step containing the specified output.
        relative_path: The relative path of the output within the output directory
            of the specified recipe step.

    Returns:
        The absolute path of the step output on the local filesystem, which may or may
        not exist.
    """
    execution_dir_path = get_or_create_base_execution_directory(recipe_root_path=recipe_root_path)
    step_outputs_path = _get_step_output_directory_path(
        execution_directory_path=execution_dir_path,
        step_name=step_name,
    )
    return os.path.abspath(os.path.join(step_outputs_path, relative_path))


def _get_or_create_execution_directory(
    recipe_root_path: str, recipe_steps: List[BaseStep], template: str
) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified recipe, creating the execution directory and its required contents if they do
    not already exist.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        recipe_steps: A list of all the steps contained in the specified recipe.
        template: The template to use to generate the makefile.

    Returns:
        The absolute path of the execution directory on the local filesystem for the specified
        recipe.
    """
    execution_dir_path = get_or_create_base_execution_directory(recipe_root_path=recipe_root_path)

    _create_makefile(recipe_root_path, execution_dir_path, template)
    for step in recipe_steps:
        step_output_subdir_path = _get_step_output_directory_path(execution_dir_path, step.name)
        os.makedirs(step_output_subdir_path, exist_ok=True)

    return execution_dir_path


def _write_updated_step_confs(recipe_steps: List[BaseStep], execution_directory_path: str) -> None:
    """
    Compares the in-memory configuration state of the specified recipe steps with step-specific
    internal configuration files written by prior executions. If updates are found, writes updated
    state to the corresponding files. If no updates are found, configuration state is not
    rewritten.

    Args:
        recipe_steps: A list of all the steps contained in the specified recipe.
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the specified recipe. Configuration files are written to step-specific
            subdirectories of this execution directory.
    """
    for step in recipe_steps:
        step_subdir_path = os.path.join(
            execution_directory_path, _STEPS_SUBDIRECTORY_NAME, step.name
        )
        step_conf_path = os.path.join(step_subdir_path, _STEP_CONF_YAML_NAME)
        if os.path.exists(step_conf_path):
            prev_step_conf = read_yaml(root=step_subdir_path, file_name=_STEP_CONF_YAML_NAME)
        else:
            prev_step_conf = None

        if prev_step_conf != step.step_config:
            write_yaml(
                root=step_subdir_path,
                file_name=_STEP_CONF_YAML_NAME,
                data=step.step_config,
                overwrite=True,
                sort_keys=True,
            )


def get_or_create_base_execution_directory(recipe_root_path: str) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified recipe. The directory is created if it does not exist.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.

    Returns:
        The path of the execution directory on the local filesystem corresponding to the
        specified recipe.
    """
    execution_directory_basename = _get_execution_directory_basename(
        recipe_root_path=recipe_root_path
    )

    execution_dir_path = os.path.abspath(
        MLFLOW_RECIPES_EXECUTION_DIRECTORY.get()
        or os.path.join(os.path.expanduser("~"), ".mlflow", "recipes", execution_directory_basename)
    )
    os.makedirs(execution_dir_path, exist_ok=True)
    return execution_dir_path


def _get_execution_directory_basename(recipe_root_path):
    """
    Obtains the basename of the execution directory corresponding to the specified recipe.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.

    Returns:
        The basename of the execution directory corresponding to the specified recipe.
    """
    return hashlib.sha256(os.path.abspath(recipe_root_path).encode("utf-8")).hexdigest()


def _get_step_output_directory_path(execution_directory_path: str, step_name: str) -> str:
    """
    Obtains the path of the local filesystem directory containing outputs for the specified step,
    which may or may not exist.

    Args:
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the relevant recipe. The Makefile is created in this directory.
        step_name: The name of the recipe step for which to obtain the output directory path.

    Returns:
        The absolute path of the local filesystem directory containing outputs for the specified
        step.
    """
    return os.path.abspath(
        os.path.join(
            execution_directory_path,
            _STEPS_SUBDIRECTORY_NAME,
            step_name,
            _STEP_OUTPUTS_SUBDIRECTORY_NAME,
        )
    )


class _ExecutionPlan:
    _MSG_REGEX = r'^echo "Run MLflow Recipe step: (\w+)"\n$'
    _FORMAT_STEPS_CACHED = "%s: No changes. Skipping."

    def __init__(self, rule_name, output_lines_of_make: List[str], recipe_step_names: List[str]):
        steps_to_run = self._parse_output_lines(output_lines_of_make)
        self.steps_cached = self._infer_cached_steps(rule_name, steps_to_run, recipe_step_names)

    @staticmethod
    def _parse_output_lines(output_lines_of_make: List[str]) -> List[str]:
        """
        Parse the output lines of Make to get steps to run.
        """

        def get_step_to_run(output_line: str):
            m = re.search(_ExecutionPlan._MSG_REGEX, output_line)
            return m.group(1) if m else None

        def steps_to_run():
            for output_line in output_lines_of_make:
                step = get_step_to_run(output_line)
                if step is not None:
                    yield step

        return list(steps_to_run())

    @staticmethod
    def _infer_cached_steps(rule_name, steps_to_run, recipe_step_names) -> List[str]:
        """
        Infer cached steps.

        Args:
            rule_name: The name of the Make rule to run.
            steps_to_run: The step names obtained by parsing the Make output showing
                which steps will be executed.
            recipe_step_names: A list of all the step names contained in the specified
                recipe sorted by the execution order.

        """
        index = recipe_step_names.index(rule_name)
        if index == 0:
            # If the rule_name is ingest, it should always be executed
            return []

        if len(steps_to_run) == 0:
            # All steps are cached
            return recipe_step_names[: index + 1]

        first_step_index = min([recipe_step_names.index(step) for step in steps_to_run])
        return recipe_step_names[:first_step_index]

    def print(self) -> None:
        if len(self.steps_cached) > 0:
            steps_cached_str = ", ".join(self.steps_cached)
            _logger.info(self._FORMAT_STEPS_CACHED, steps_cached_str)


def _run_make(
    execution_directory_path,
    rule_name: str,
    extra_env: Dict[str, str],
    recipe_steps: List[BaseStep],
) -> None:
    """
    Runs the specified recipe rule with Make. This method assumes that a Makefile named `Makefile`
    exists in the specified execution directory.

    Args:
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the relevant recipe. The Makefile is created in this directory.
        extra_env: Extra environment variables to be defined when running the Make child process.
        rule_name: The name of the Make rule to run.
        recipe_steps: A list of step instances that is a subgraph containing the step specified
            by `rule_name`.
    """
    # Dry-run Make and collect the outputs
    process = _exec_cmd(
        ["make", "-n", "-f", "Makefile", rule_name],
        capture_output=False,
        stream_output=True,
        synchronous=False,
        throw_on_error=False,
        cwd=execution_directory_path,
        extra_env=extra_env,
    )
    output_lines = list(iter(process.stdout.readline, ""))
    process.communicate()
    return_code = process.poll()
    if return_code == 0:
        # Only try to print cached steps message when `make -n` completes with no error.
        # Note that runtime errors from shell cannot be detected by Make dry-run, so the
        # return code will be 0 in this case. As long as `make -n` has no error, cached
        # steps inference logic can work correctly even when shell runtime error occurs.
        recipe_step_names = [step.name for step in recipe_steps]
        _ExecutionPlan(rule_name, output_lines, recipe_step_names).print()

    _exec_cmd(
        ["make", "-s", "-f", "Makefile", rule_name],
        capture_output=False,
        stream_output=True,
        synchronous=True,
        throw_on_error=False,
        cwd=execution_directory_path,
        extra_env=extra_env,
    )


def _create_makefile(recipe_root_path, execution_directory_path, template) -> None:
    """
    Creates a Makefile with a set of relevant MLflow Recipes targets for the specified recipe,
    overwriting the preexisting Makefile if one exists. The Makefile is created in the specified
    execution directory.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        execution_directory_path: The absolute path of the execution directory on the local
            filesystem for the specified recipe. The Makefile is created in this directory.
        template: The template to use to generate the makefile.
    """
    makefile_path = os.path.join(execution_directory_path, "Makefile")

    if template == "regression/v1" or template == "classification/v1":
        makefile_to_use = _MAKEFILE_FORMAT_STRING
        steps_folder_path = os.path.join(recipe_root_path, "steps")
        if not os.path.exists(steps_folder_path):
            os.mkdir(steps_folder_path)
        for required_file in [
            "ingest.py",
            "split.py",
            "train.py",
            "transform.py",
            "custom_metrics.py",
        ]:
            required_file_path = os.path.join(steps_folder_path, required_file)
            if not os.path.exists(required_file_path):
                try:
                    with open(required_file_path, "w") as f:
                        f.write("# Created by MLflow Pipeliens\n")
                except OSError:
                    pass
            if not os.path.exists(required_file_path):
                raise ValueError(
                    f"Can not find required file {required_file_path} from steps folder. "
                    "Please create empty python file if the step is not used."
                )
    else:
        raise ValueError(f"Invalid template: {template}")

    makefile_contents = makefile_to_use.format(
        path=_MakefilePathFormat(
            os.path.abspath(recipe_root_path),
            execution_directory_path=os.path.abspath(execution_directory_path),
        ),
    )
    with open(makefile_path, "w") as f:
        f.write(makefile_contents)


class _MakefilePathFormat:
    r"""
    Provides platform-agnostic path substitution for execution Makefiles, ensuring that POSIX-style
    relative paths are joined correctly with POSIX-style or Windows-style recipe root paths.

    For example, given a format string `s = "{path:prp/my/subpath.txt}"`, invoking
    `s.format(path=_MakefilePathFormat(recipe_root_path="/my/recipe/root/path", ...))` on
    Unix systems or
    `s.format(path=_MakefilePathFormat(recipe_root_path="C:\my\recipe\root\path", ...))`` on
    Windows systems will yield "/my/recipe/root/path/my/subpath.txt" or
    "C:/my/recipe/root/path/my/subpath.txt", respectively.

    Additionally, given a format string `s = "{path:exe/my/subpath.txt}"`, invoking
    `s.format(path=_MakefilePathFormat(execution_directory_path="/my/exe/dir/path", ...))` on
    Unix systems or
    `s.format(path=_MakefilePathFormat(execution_directory_path="/my/exe/dir/path", ...))`` on
    Windows systems will yield "/my/exe/dir/path/my/subpath.txt" or
    "C:/my/exe/dir/path/my/subpath.txt", respectively.
    """

    _RECIPE_ROOT_PATH_PREFIX_PLACEHOLDER = "prp/"
    _EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER = "exe/"

    def __init__(self, recipe_root_path: str, execution_directory_path: str):
        """
        Args:
            recipe_root_path: The absolute path of the recipe root directory on the local
                filesystem.
            execution_directory_path: The absolute path of the execution directory on the local
                filesystem for the recipe.
        """
        self.recipe_root_path = recipe_root_path
        self.execution_directory_path = execution_directory_path

    def _get_formatted_path(
        self, path_spec: str, prefix_placeholder: str, replacement_path: str
    ) -> str:
        """
        Args:
            path_spec: A substitution path spec of the form `<placeholder>/<subpath>`. This
                method substitutes `<placeholder>` with `<recipe_root_path>`, if
                `<placeholder>` is `prp`, or `<execution_directory_path>`, if
                `<placeholder>` is `exe`.
            prefix_placeholder: The prefix placeholder, which is present at the beginning of
                `path_spec`. Either `prp` or `exe`.
            replacement_path: The path to use to replace the specified `prefix_placeholder`
                in the specified `path_spec`.

        Returns:
            The formatted path obtained by replacing the ``prefix placeholder`` in the
            specified ``path_spec`` with the specified ``replacement_path``.
        """
        subpath = pathlib.PurePosixPath(path_spec.split(prefix_placeholder)[1])
        recipe_root_posix_path = pathlib.PurePosixPath(pathlib.Path(replacement_path).as_posix())
        full_formatted_path = recipe_root_posix_path / subpath
        return str(full_formatted_path)

    def __format__(self, path_spec: str) -> str:
        """
        Args:
            path_spec: A substitution path spec of the form `<placeholder>/<subpath>`. This
                method substitutes `<placeholder>` with `<recipe_root_path>`, if
                `<placeholder>` is `prp`, or `<execution_directory_path>`, if
                `<placeholder>` is `exe`.
        """
        if path_spec.startswith(_MakefilePathFormat._RECIPE_ROOT_PATH_PREFIX_PLACEHOLDER):
            return self._get_formatted_path(
                path_spec=path_spec,
                prefix_placeholder=_MakefilePathFormat._RECIPE_ROOT_PATH_PREFIX_PLACEHOLDER,
                replacement_path=self.recipe_root_path,
            )
        elif path_spec.startswith(_MakefilePathFormat._EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER):
            return self._get_formatted_path(
                path_spec=path_spec,
                prefix_placeholder=_MakefilePathFormat._EXECUTION_DIRECTORY_PATH_PREFIX_PLACEHOLDER,
                replacement_path=self.execution_directory_path,
            )
        else:
            raise ValueError(f"Invalid Makefile string format path spec: {path_spec}")


# Makefile contents for cache-aware recipe execution. These contents include variable placeholders
# that need to be formatted (substituted) with the recipe root directory in order to produce a
# valid Makefile
_MAKEFILE_FORMAT_STRING = r"""
# Define `ingest` as a target with no dependencies to ensure that it runs whenever a user explicitly
# invokes the MLflow Recipes ingest step, allowing them to reingest data on-demand
ingest:
	cd {path:prp/} && \
        python -c "from mlflow.recipes.steps.ingest import IngestStep; IngestStep.from_step_config_path(step_config_path='{path:exe/steps/ingest/conf.yaml}', recipe_root='{path:prp/}').run(output_directory='{path:exe/steps/ingest/outputs}')"

# Define a separate target for the ingested dataset that recursively invokes make with the `ingest`
# target. Downstream steps depend on the ingested dataset target, rather than the `ingest` target,
# ensuring that data is only ingested for downstream steps if it is not already present on the
# local filesystem
steps/ingest/outputs/dataset.parquet: steps/ingest/conf.yaml {path:prp/steps/ingest.py}
	echo "Run MLflow Recipe step: ingest"
	$(MAKE) ingest

split_objects = steps/split/outputs/train.parquet steps/split/outputs/validation.parquet steps/split/outputs/test.parquet

split: $(split_objects)

steps/%/outputs/train.parquet steps/%/outputs/validation.parquet steps/%/outputs/test.parquet: {path:prp/steps/split.py} steps/ingest/outputs/dataset.parquet steps/split/conf.yaml
	echo "Run MLflow Recipe step: split"
	cd {path:prp/} && \
        python -c "from mlflow.recipes.steps.split import SplitStep; SplitStep.from_step_config_path(step_config_path='{path:exe/steps/split/conf.yaml}', recipe_root='{path:prp/}').run(output_directory='{path:exe/steps/split/outputs}')"

transform_objects = steps/transform/outputs/transformer.pkl steps/transform/outputs/transformed_training_data.parquet steps/transform/outputs/transformed_validation_data.parquet

transform: $(transform_objects)

steps/%/outputs/transformer.pkl steps/%/outputs/transformed_training_data.parquet steps/%/outputs/transformed_validation_data.parquet: {path:prp/steps/transform.py} steps/split/outputs/train.parquet steps/split/outputs/validation.parquet steps/transform/conf.yaml
	echo "Run MLflow Recipe step: transform"
	cd {path:prp/} && \
        python -c "from mlflow.recipes.steps.transform import TransformStep; TransformStep.from_step_config_path(step_config_path='{path:exe/steps/transform/conf.yaml}', recipe_root='{path:prp/}').run(output_directory='{path:exe/steps/transform/outputs}')"

train_objects = steps/train/outputs/model steps/train/outputs/run_id

train: $(train_objects)

steps/%/outputs/model steps/%/outputs/run_id: {path:prp/steps/train.py} {path:prp/steps/custom_metrics.py} steps/transform/outputs/transformed_training_data.parquet steps/transform/outputs/transformed_validation_data.parquet steps/split/outputs/train.parquet steps/split/outputs/validation.parquet steps/transform/outputs/transformer.pkl steps/train/conf.yaml
	echo "Run MLflow Recipe step: train"
	cd {path:prp/} && \
        python -c "from mlflow.recipes.steps.train import TrainStep; TrainStep.from_step_config_path(step_config_path='{path:exe/steps/train/conf.yaml}', recipe_root='{path:prp/}').run(output_directory='{path:exe/steps/train/outputs}')"

evaluate_objects = steps/evaluate/outputs/model_validation_status

evaluate: $(evaluate_objects)

steps/%/outputs/model_validation_status: {path:prp/steps/custom_metrics.py} steps/train/outputs/model steps/split/outputs/validation.parquet steps/split/outputs/test.parquet steps/train/outputs/run_id steps/evaluate/conf.yaml
	echo "Run MLflow Recipe step: evaluate"
	cd {path:prp/} && \
        python -c "from mlflow.recipes.steps.evaluate import EvaluateStep; EvaluateStep.from_step_config_path(step_config_path='{path:exe/steps/evaluate/conf.yaml}', recipe_root='{path:prp/}').run(output_directory='{path:exe/steps/evaluate/outputs}')"

register_objects = steps/register/outputs/registered_model_version.json

register: $(register_objects)

steps/%/outputs/registered_model_version.json: steps/train/outputs/run_id steps/register/conf.yaml steps/evaluate/outputs/model_validation_status
	echo "Run MLflow Recipe step: register"
	cd {path:prp/} && \
        python -c "from mlflow.recipes.steps.register import RegisterStep; RegisterStep.from_step_config_path(step_config_path='{path:exe/steps/register/conf.yaml}', recipe_root='{path:prp/}').run(output_directory='{path:exe/steps/register/outputs}')"

# Define `ingest_scoring` as a target with no dependencies to ensure that it runs whenever a user explicitly
# invokes the MLflow Recipes ingest_scoring step, allowing them to reingest data on-demand
ingest_scoring:
	cd {path:prp/} && \
        python -c "from mlflow.recipes.steps.ingest import IngestScoringStep; IngestScoringStep.from_step_config_path(step_config_path='{path:exe/steps/ingest_scoring/conf.yaml}', recipe_root='{path:prp/}').run(output_directory='{path:exe/steps/ingest_scoring/outputs}')"

# Define a separate target for the ingested dataset that recursively invokes make with the
# `ingest_scoring` target. Downstream steps depend on the ingested dataset target, rather than the
# `ingest_scoring` target, ensuring that data is only ingested for downstream steps if it is not
# already present on the local filesystem
steps/ingest_scoring/outputs/scoring-dataset.parquet: steps/ingest_scoring/conf.yaml {path:prp/steps/ingest.py}
	echo "Run MLflow Recipe step: ingest_scoring"
	$(MAKE) ingest_scoring

predict_objects = steps/predict/outputs/scored.parquet

predict: $(predict_objects)

steps/predict/outputs/scored.parquet: steps/ingest_scoring/outputs/scoring-dataset.parquet steps/predict/conf.yaml
	echo "Run MLflow Recipe step: predict"
	cd {path:prp/} && \
        python -c "from mlflow.recipes.steps.predict import PredictStep; PredictStep.from_step_config_path(step_config_path='{path:exe/steps/predict/conf.yaml}', recipe_root='{path:prp/}').run(output_directory='{path:exe/steps/predict/outputs}')"

clean:
	rm -rf $(split_objects) $(transform_objects) $(train_objects) $(evaluate_objects) $(predict_objects)
"""  # noqa: E501
