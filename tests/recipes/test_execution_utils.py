import os
import pathlib
import shutil
import time
from unittest import mock

import pandas as pd
import pytest

from mlflow.environment_variables import MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME
from mlflow.recipes import Recipe
from mlflow.recipes.step import StepStatus
from mlflow.recipes.steps.ingest import IngestStep
from mlflow.recipes.steps.split import SplitStep
from mlflow.recipes.steps.transform import TransformStep
from mlflow.recipes.utils.execution import (
    _ExecutionPlan,
    _get_or_create_execution_directory,
    get_step_output_path,
    run_recipe_step,
)

from tests.recipes.helper_functions import BaseStepImplemented


@pytest.fixture
def pandas_df():
    df = pd.DataFrame.from_dict(
        {
            "A": ["x", "y", "z"],
            "B": [1, 2, 3],
            "C": [-9.2, 82.5, 3.40],
        }
    )
    # duplicate the df so that each partition is above min size
    df2 = pd.concat([df] * 10, ignore_index=True)
    df2.index.rename("index", inplace=True)
    return df2


@pytest.fixture
def test_recipe(enter_test_recipe_directory, pandas_df, tmp_path):
    dataset_path = tmp_path / "df.parquet"
    pandas_df.to_parquet(dataset_path)
    ingest_step = IngestStep.from_recipe_config(
        recipe_config={
            "target_col": "C",
            "steps": {
                "ingest": {
                    "using": "parquet",
                    "location": str(dataset_path),
                }
            },
        },
        recipe_root=os.getcwd(),
    )
    split_step = SplitStep.from_recipe_config(
        recipe_config={
            "target_col": "C",
            "steps": {
                "split": {
                    "split_ratios": [0.34, 0.33, 0.33],
                },
            },
        },
        recipe_root=os.getcwd(),
    )
    with open(pathlib.Path().cwd() / "steps" / "transform.py", "w") as f:
        f.write(
            "\n".join(
                [
                    "from sklearn.pipeline import Pipeline",
                    "from sklearn.preprocessing import FunctionTransformer",
                    "",
                    "def add_column(df):",
                    "    df['useless'] = 'useless'",
                    "    return df",
                    "",
                    "def transform_fn():",
                    "    return Pipeline(steps=[('add_column', FunctionTransformer(add_column))])",
                ]
            )
        )
    transform_step = TransformStep.from_recipe_config(
        recipe_config={
            "target_col": "C",
            "steps": {
                "transform": {
                    "using": "custom",
                    "transformer_method": "transform_fn",
                },
            },
        },
        recipe_root=os.getcwd(),
    )
    return [ingest_step, split_step, transform_step]


@pytest.fixture(autouse=True)
def clean_test_recipe(enter_test_recipe_directory):
    Recipe(profile="local").clean()
    try:
        yield
    finally:
        Recipe(profile="local").clean()


def test_create_required_step_files(tmp_path):
    class TestStep(BaseStepImplemented):
        def __init__(self):
            pass

        @property
        def name(self):
            return "test_step"

    def check_required_files(check_exist):
        for required_file in [
            "steps",
            "steps/ingest.py",
            "steps/split.py",
            "steps/train.py",
            "steps/transform.py",
            "steps/custom_metrics.py",
        ]:
            assert (tmp_path / required_file).exists() is check_exist

    test_step = TestStep()
    check_required_files(False)
    _get_or_create_execution_directory(
        recipe_root_path=tmp_path, recipe_steps=[test_step], template="regression/v1"
    )
    check_required_files(True)


def test_get_or_create_execution_directory_is_idempotent(tmp_path):
    class TestStep(BaseStepImplemented):
        def __init__(self):
            pass

        @property
        def name(self):
            return "test_step"

    test_step = TestStep()

    def assert_expected_execution_directory_contents_exist(execution_dir_path):
        assert (execution_dir_path / "Makefile").exists()
        assert (execution_dir_path / "steps").exists()
        assert (execution_dir_path / "steps" / test_step.name / "outputs").exists()

    execution_dir_path_1 = pathlib.Path(
        _get_or_create_execution_directory(
            recipe_root_path=tmp_path, recipe_steps=[test_step], template="regression/v1"
        )
    )
    execution_dir_path_2 = pathlib.Path(
        _get_or_create_execution_directory(
            recipe_root_path=tmp_path, recipe_steps=[test_step], template="regression/v1"
        )
    )
    assert execution_dir_path_1 == execution_dir_path_2
    assert_expected_execution_directory_contents_exist(execution_dir_path_1)

    shutil.rmtree(execution_dir_path_1)

    # Simulate a failure with Makefile creation
    with (
        mock.patch(
            "mlflow.recipes.utils.execution._create_makefile",
            side_effect=Exception("Makefile creation failed"),
        ),
        pytest.raises(Exception, match="Makefile creation failed"),
    ):
        _get_or_create_execution_directory(
            recipe_root_path=tmp_path, recipe_steps=[test_step], template="regression/v1"
        )

    # Verify that the directory exists but is empty due to short circuiting after
    # failed Makefile creation
    assert execution_dir_path_1.exists()
    assert next(execution_dir_path_1.iterdir(), None) is None

    # Re-create the execution directory and verify that all expected contents are present
    execution_dir_path_3 = pathlib.Path(
        _get_or_create_execution_directory(
            recipe_root_path=tmp_path, recipe_steps=[test_step], template="regression/v1"
        )
    )
    assert execution_dir_path_3 == execution_dir_path_1
    assert_expected_execution_directory_contents_exist(execution_dir_path_3)

    shutil.rmtree(execution_dir_path_1)

    # Simulate a failure with step-specific directory creation
    with (
        mock.patch(
            "mlflow.recipes.utils.execution._get_step_output_directory_path",
            side_effect=Exception("Step directory creation failed"),
        ),
        pytest.raises(Exception, match="Step directory creation failed"),
    ):
        _get_or_create_execution_directory(
            recipe_root_path=tmp_path, recipe_steps=[test_step], template="regression/v1"
        )

    # Verify that the directory exists & that a Makefile is present but step-specific directories
    # were not created due to failures
    assert execution_dir_path_1.exists()
    assert [path.name for path in execution_dir_path_1.iterdir()] == ["Makefile"]

    # Re-create the execution directory and verify that all expected contents are present
    execution_dir_path_4 = pathlib.Path(
        _get_or_create_execution_directory(
            recipe_root_path=tmp_path, recipe_steps=[test_step], template="regression/v1"
        )
    )
    assert execution_dir_path_4 == execution_dir_path_1
    assert_expected_execution_directory_contents_exist(execution_dir_path_4)


def test_run_recipe_step_sets_environment_as_expected(tmp_path):
    class TestStep1(BaseStepImplemented):
        def __init__(self):
            self.step_config = {}

        @property
        def name(self):
            return "test_step_1"

        @property
        def environment(self):
            return {"A": "B"}

    class TestStep2(BaseStepImplemented):
        def __init__(self):
            self.step_config = {}

        @property
        def name(self):
            return "test_step_2"

        @property
        def environment(self):
            return {"C": "D"}

    with (
        mock.patch("mlflow.recipes.utils.execution._exec_cmd") as mock_run_in_subprocess,
        mock.patch("mlflow.recipes.utils.execution._ExecutionPlan"),
    ):
        process = mock.Mock()
        process.stdout.readline = mock.Mock(side_effect="")
        mock_run_in_subprocess.return_value = process

        recipe_steps = [TestStep1(), TestStep2()]
        run_recipe_step(
            recipe_root_path=tmp_path,
            recipe_steps=recipe_steps,
            target_step=recipe_steps[0],
            template="regression/v1",
        )

    _, subprocess_call_kwargs = mock_run_in_subprocess.call_args
    assert subprocess_call_kwargs.get("extra_env") == {
        "A": "B",
        "C": "D",
        MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME.name: "test_step_1",
    }
    assert mock_run_in_subprocess.call_count == 2


def test_run_recipe_step_calls_execution_plan(tmp_path):
    class TestStep(BaseStepImplemented):
        def __init__(self):
            self.step_config = {}

        @property
        def name(self):
            return "test_step"

    with (
        mock.patch("mlflow.recipes.utils.execution._exec_cmd") as mock_run_in_subprocess,
        mock.patch("mlflow.recipes.utils.execution._ExecutionPlan") as mock_execution_plan,
    ):
        process = mock.Mock()
        process.poll.return_value = 0
        process.stdout.readline = mock.Mock(side_effect="")
        mock_run_in_subprocess.return_value = process

        recipe_steps = [TestStep()]
        run_recipe_step(
            recipe_root_path=tmp_path,
            recipe_steps=recipe_steps,
            target_step=recipe_steps[0],
            template="regression/v1",
        )

    execution_plan_args, _ = mock_execution_plan.call_args
    assert execution_plan_args[0] == "test_step"
    assert execution_plan_args[2] == ["test_step"]


def run_test_recipe_step(recipe_steps, target_step):
    return run_recipe_step(
        recipe_root_path=os.getcwd(),
        recipe_steps=recipe_steps,
        target_step=target_step,
        template="regression/v1",
    )


def get_test_recipe_step_output_directory(step):
    return get_step_output_path(os.getcwd(), step.name, "")


def get_test_recipe_step_execution_state(step):
    return step.get_execution_state(get_test_recipe_step_output_directory(step))


@pytest.mark.usefixtures("enter_test_recipe_directory")
def test_run_recipe_step_maintains_execution_status_correctly(pandas_df, tmp_path):
    dataset_path = tmp_path / "df.parquet"
    pandas_df.to_parquet(dataset_path)

    ingest_step_good = IngestStep.from_recipe_config(
        recipe_config={
            "target_col": "C",
            "steps": {
                "ingest": {
                    "using": "parquet",
                    "location": str(dataset_path),
                }
            },
        },
        recipe_root=os.getcwd(),
    )

    assert get_test_recipe_step_execution_state(ingest_step_good).status == StepStatus.UNKNOWN
    assert get_test_recipe_step_execution_state(ingest_step_good).last_updated_timestamp == 0
    assert get_test_recipe_step_execution_state(ingest_step_good).stack_trace is None
    curr_time = time.time()
    run_test_recipe_step([ingest_step_good], ingest_step_good)
    assert get_test_recipe_step_execution_state(ingest_step_good).status == StepStatus.SUCCEEDED
    assert (
        get_test_recipe_step_execution_state(ingest_step_good).last_updated_timestamp >= curr_time
    )
    assert get_test_recipe_step_execution_state(ingest_step_good).stack_trace is None

    ingest_step_bad = IngestStep.from_recipe_config(
        recipe_config={
            "target_col": "C",
            "steps": {
                "ingest": {
                    "using": "parquet",
                    "location": "badlocation",
                }
            },
        },
        recipe_root=os.getcwd(),
    )
    curr_time = time.time()
    run_test_recipe_step([ingest_step_bad], ingest_step_bad)
    assert get_test_recipe_step_execution_state(ingest_step_bad).status == StepStatus.FAILED
    assert get_test_recipe_step_execution_state(ingest_step_bad).last_updated_timestamp >= curr_time
    assert "Traceback" in get_test_recipe_step_execution_state(ingest_step_bad).stack_trace


def test_run_recipe_step_returns_expected_result(test_recipe):
    ingest_step, split_step, _ = test_recipe

    assert run_test_recipe_step(test_recipe, ingest_step) == ingest_step
    assert run_test_recipe_step(test_recipe, split_step) == split_step
    assert run_test_recipe_step(test_recipe, ingest_step) == ingest_step

    ingest_step_bad = IngestStep.from_recipe_config(
        recipe_config={
            "target_col": "C",
            "steps": {
                "ingest": {
                    "using": "parquet",
                    "location": "badlocation",
                }
            },
        },
        recipe_root=os.getcwd(),
    )

    assert run_test_recipe_step([ingest_step_bad, split_step], ingest_step_bad) == ingest_step_bad
    assert run_test_recipe_step([ingest_step_bad, split_step], split_step) == ingest_step_bad


def test_run_recipe_with_ingest_step_as_target_never_caches(test_recipe):
    ingest_step, _, _ = test_recipe

    def get_step_outputs_with_timestamps(step):
        output_directory = get_test_recipe_step_output_directory(step)
        return {
            path: os.path.getmtime(os.path.join(output_directory, path))
            for path in os.listdir(output_directory)
        }

    curr_time = time.time()
    run_test_recipe_step(test_recipe, ingest_step)
    step_outputs_with_timestamps_1 = get_step_outputs_with_timestamps(ingest_step)
    assert step_outputs_with_timestamps_1
    assert get_test_recipe_step_execution_state(ingest_step).last_updated_timestamp >= curr_time

    curr_time = time.time()
    run_test_recipe_step(test_recipe, ingest_step)
    step_outputs_with_timestamps_2 = get_step_outputs_with_timestamps(ingest_step)
    assert step_outputs_with_timestamps_2
    assert get_test_recipe_step_execution_state(ingest_step).last_updated_timestamp >= curr_time

    assert step_outputs_with_timestamps_2 != step_outputs_with_timestamps_1


@pytest.mark.parametrize("target_step", ["split", "transform"])
def test_run_recipe_step_caches(test_recipe, target_step):
    _, split_step, transform_step = test_recipe
    target_step = split_step if target_step == "split" else transform_step

    def get_step_outputs_with_timestamps(step):
        output_directory = get_test_recipe_step_output_directory(step)
        return {
            path: os.path.getmtime(os.path.join(output_directory, path))
            for path in os.listdir(output_directory)
        }

    curr_time = time.time()
    run_test_recipe_step(test_recipe, target_step)
    step_outputs_with_timestamps_1 = get_step_outputs_with_timestamps(target_step)
    assert step_outputs_with_timestamps_1
    step_execution_state_1 = get_test_recipe_step_execution_state(target_step)
    assert step_execution_state_1.status == StepStatus.SUCCEEDED
    assert step_execution_state_1.last_updated_timestamp >= curr_time

    curr_time = time.time()
    run_test_recipe_step(test_recipe, target_step)
    step_outputs_with_timestamps_2 = get_step_outputs_with_timestamps(target_step)
    assert step_outputs_with_timestamps_2
    step_execution_state_2 = get_test_recipe_step_execution_state(target_step)
    assert (
        step_execution_state_2.last_updated_timestamp
        == step_execution_state_1.last_updated_timestamp
    )

    assert step_outputs_with_timestamps_2 == step_outputs_with_timestamps_1


def test_run_recipe_with_ingest_step_as_target_clears_downstream_step_state(test_recipe):
    ingest_step, split_step, transform_step = test_recipe

    curr_time = time.time()
    run_test_recipe_step(test_recipe, transform_step)
    for step in test_recipe:
        assert get_test_recipe_step_execution_state(step).status == StepStatus.SUCCEEDED
        assert get_test_recipe_step_execution_state(step).last_updated_timestamp >= curr_time
        assert os.listdir(get_test_recipe_step_output_directory(step))

    curr_time = time.time()
    run_test_recipe_step(test_recipe, ingest_step)
    assert get_test_recipe_step_execution_state(ingest_step).status == StepStatus.SUCCEEDED
    assert get_test_recipe_step_execution_state(ingest_step).last_updated_timestamp >= curr_time
    assert os.listdir(get_test_recipe_step_output_directory(ingest_step))
    for step in [split_step, transform_step]:
        assert get_test_recipe_step_execution_state(step).status == StepStatus.UNKNOWN
        assert get_test_recipe_step_execution_state(step).last_updated_timestamp == 0
        assert not os.listdir(get_test_recipe_step_output_directory(step))


def test_run_recipe_step_after_change_clears_downstream_step_state(test_recipe):
    ingest_step, _, transform_step = test_recipe
    curr_time = time.time()

    run_test_recipe_step(test_recipe, transform_step)
    for step in test_recipe:
        assert get_test_recipe_step_execution_state(step).status == StepStatus.SUCCEEDED
        assert get_test_recipe_step_execution_state(step).last_updated_timestamp >= curr_time
        assert os.listdir(get_test_recipe_step_output_directory(step))

    updated_split_step = SplitStep.from_recipe_config(
        recipe_config={
            "target_col": "C",
            "steps": {
                "split": {
                    "split_ratios": [0.5, 0.25, 0.25],
                },
            },
        },
        recipe_root=os.getcwd(),
    )
    run_test_recipe_step([ingest_step, updated_split_step, transform_step], updated_split_step)
    for step in [ingest_step, updated_split_step]:
        assert get_test_recipe_step_execution_state(step).status == StepStatus.SUCCEEDED
        assert get_test_recipe_step_execution_state(step).last_updated_timestamp >= curr_time
        assert os.listdir(get_test_recipe_step_output_directory(step))

    assert get_test_recipe_step_execution_state(transform_step).status == StepStatus.UNKNOWN
    assert get_test_recipe_step_execution_state(transform_step).last_updated_timestamp == 0
    assert not os.listdir(get_test_recipe_step_output_directory(transform_step))


def test_run_recipe_step_without_change_preserves_state_of_all_recipe_steps(test_recipe):
    _, split_step, transform_step = test_recipe
    curr_time = time.time()

    run_test_recipe_step(test_recipe, transform_step)
    step_to_execution_state = {}
    for step in test_recipe:
        step_execution_state = get_test_recipe_step_execution_state(step)
        step_to_execution_state[step] = step_execution_state
        assert step_execution_state.status == StepStatus.SUCCEEDED
        assert step_execution_state.last_updated_timestamp >= curr_time

    run_test_recipe_step(test_recipe, split_step)
    for step in test_recipe:
        step_execution_state = get_test_recipe_step_execution_state(step)
        prev_execution_state = step_to_execution_state[step]
        assert step_execution_state.status == prev_execution_state.status
        assert (
            step_execution_state.last_updated_timestamp
            == prev_execution_state.last_updated_timestamp
        )


def test_run_recipe_step_failure_clears_downstream_step_state(test_recipe):
    ingest_step, split_step, _ = test_recipe

    curr_time = time.time()
    run_test_recipe_step(test_recipe, split_step)
    for step in [ingest_step, split_step]:
        assert get_test_recipe_step_execution_state(step).status == StepStatus.SUCCEEDED
        assert get_test_recipe_step_execution_state(step).last_updated_timestamp >= curr_time
        assert os.listdir(get_test_recipe_step_output_directory(step))

    ingest_step_bad = IngestStep.from_recipe_config(
        recipe_config={
            "target_col": "C",
            "steps": {
                "ingest": {
                    "using": "parquet",
                    "location": "badlocation",
                }
            },
        },
        recipe_root=os.getcwd(),
    )

    curr_time = time.time()
    run_test_recipe_step([ingest_step_bad, split_step], ingest_step_bad)
    assert get_test_recipe_step_execution_state(ingest_step_bad).status == StepStatus.FAILED
    assert get_test_recipe_step_execution_state(ingest_step_bad).last_updated_timestamp >= curr_time
    assert get_test_recipe_step_execution_state(split_step).status == StepStatus.UNKNOWN
    assert get_test_recipe_step_execution_state(split_step).last_updated_timestamp == 0
    assert not os.listdir(get_test_recipe_step_output_directory(split_step))


def test_execution_plan():
    train_subgraph = ["ingest", "split", "transform", "train", "evaluate", "register"]

    # all steps are cached
    plan = _ExecutionPlan("register", ["make: `register' is up to date."], train_subgraph)
    assert plan.steps_cached == train_subgraph

    # all steps will be executed
    plan = _ExecutionPlan(
        "transform",
        [
            'echo "Run MLflow Recipe step: ingest"\n',
            'echo "Run MLflow Recipe step: split"\n',
            'echo "Run MLflow Recipe step: transform"\n',
        ],
        train_subgraph,
    )
    assert plan.steps_cached == []

    plan = _ExecutionPlan(
        "transform", ['echo "Run MLflow Recipe step: transform"\n'], train_subgraph
    )
    assert plan.steps_cached == ["ingest", "split"]
