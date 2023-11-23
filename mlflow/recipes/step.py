import abc
import json
import logging
import os
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml

from mlflow.recipes.cards import CARD_HTML_NAME, CARD_PICKLE_NAME, BaseCard, FailureCard
from mlflow.recipes.utils import get_recipe_name
from mlflow.recipes.utils.step import display_html
from mlflow.tracking import MlflowClient
from mlflow.utils.databricks_utils import is_in_databricks_runtime

_logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """
    Represents the execution status of a step.
    """

    # Indicates that no execution status information is available for the step,
    # which may occur if the step has never been run or its outputs have been cleared
    UNKNOWN = "UNKNOWN"
    # Indicates that the step is currently running
    RUNNING = "RUNNING"
    # Indicates that the step completed successfully
    SUCCEEDED = "SUCCEEDED"
    # Indicates that the step completed with one or more failures
    FAILED = "FAILED"


class StepClass(Enum):
    """
    Represents the class of a step.
    """

    # Indicates that the step class is unknown.
    UNKNOWN = "UNKNOWN"
    # Indicates that the step runs at training time.
    TRAINING = "TRAINING"
    # Indicates that the step runs at inference time.
    PREDICTION = "PREDICTION"


class StepExecutionState:
    """
    Represents execution state for a step, including the current status and
    the time of the last status update.
    """

    _KEY_STATUS = "recipe_step_execution_status"
    _KEY_LAST_UPDATED_TIMESTAMP = "recipe_step_execution_last_updated_timestamp"
    _KEY_STACK_TRACE = "recipe_step_stack_trace"

    def __init__(self, status: StepStatus, last_updated_timestamp: int, stack_trace: str):
        """
        :param status: The execution status of the step.
        :param last_updated_timestamp: The timestamp of the last execution status update, measured
                                       in seconds since the UNIX epoch.
        :param stack_trace: The stack trace of the last execution. None if the step execution
                            succeeds.
        """
        self.status = status
        self.last_updated_timestamp = last_updated_timestamp
        self.stack_trace = stack_trace

    def to_dict(self) -> Dict[str, Any]:
        """
        Creates a dictionary representation of the step execution state.
        """
        return {
            StepExecutionState._KEY_STATUS: self.status.value,
            StepExecutionState._KEY_LAST_UPDATED_TIMESTAMP: self.last_updated_timestamp,
            StepExecutionState._KEY_STACK_TRACE: self.stack_trace,
        }

    @classmethod
    def from_dict(cls, state_dict) -> "StepExecutionState":
        """
        Creates a ``StepExecutionState`` instance from the specified execution state dictionary.
        """
        return cls(
            status=StepStatus[state_dict[StepExecutionState._KEY_STATUS]],
            last_updated_timestamp=state_dict[StepExecutionState._KEY_LAST_UPDATED_TIMESTAMP],
            stack_trace=state_dict[StepExecutionState._KEY_STACK_TRACE],
        )


class BaseStep(metaclass=abc.ABCMeta):
    """
    Base class representing a step in an MLflow Recipe
    """

    _EXECUTION_STATE_FILE_NAME = "execution_state.json"

    def __init__(self, step_config: Dict[str, Any], recipe_root: str):
        """
        :param step_config: dictionary of the config needed to
                            run/implement the step.
        :param recipe_root: String file path to the directory where step
                              are defined.
        """
        self.step_config = step_config
        self.recipe_root = recipe_root
        self.recipe_name = get_recipe_name(recipe_root_path=recipe_root)
        self.task = self.step_config.get("recipe", "regression/v1").rsplit("/", 1)[0]
        self.step_card = None

    def __str__(self):
        return f"Step:{self.name}"

    def run(self, output_directory: str):
        """
        Executes the step by running common setup operations and invoking
        step-specific code (as defined in ``_run()``).

        :param output_directory: String file path to the directory where step
                                 outputs should be stored.
        :return: None
        """
        _logger.info(f"Running step {self.name}...")
        start_timestamp = time.time()
        self._initialize_databricks_spark_connection_and_hooks_if_applicable()
        try:
            self._update_status(status=StepStatus.RUNNING, output_directory=output_directory)
            self._validate_and_apply_step_config()
            self.step_card = self._run(output_directory=output_directory)
            self._update_status(status=StepStatus.SUCCEEDED, output_directory=output_directory)
        except Exception:
            stack_trace = traceback.format_exc()
            self._update_status(
                status=StepStatus.FAILED, output_directory=output_directory, stack_trace=stack_trace
            )
            self.step_card = FailureCard(
                recipe_name=self.recipe_name,
                step_name=self.name,
                failure_traceback=stack_trace,
                output_directory=output_directory,
            )
            raise
        finally:
            self._serialize_card(start_timestamp, output_directory)

    def inspect(self, output_directory: str):
        """
        Inspect the step output state by running the generic inspect information here and
        running the step specific inspection code in the step's _inspect() method.

        :param output_directory: String file path where to the directory where step
                                 outputs are located.
        :return: None
        """
        card_path = os.path.join(output_directory, CARD_PICKLE_NAME)
        if not os.path.exists(card_path):
            _logger.info(
                "Unable to locate runtime info for step '%s'. Re-run the step before inspect.",
                self.name,
            )
            return None

        card = BaseCard.load(card_path)
        card_html_path = os.path.join(output_directory, CARD_HTML_NAME)
        display_html(html_data=card.to_html(), html_file_path=card_html_path)

    @abc.abstractmethod
    def _run(self, output_directory: str) -> BaseCard:
        """
        This function is responsible for executing the step, writing outputs
        to the specified directory, and returning results to the user. It
        is invoked by the internal step runner.

        :param output_directory: String file path to the directory where step outputs
                                 should be stored.
        :return: A BaseCard containing step execution information.
        """
        pass

    @abc.abstractmethod
    def _validate_and_apply_step_config(self) -> None:
        """
        This function is responsible for validating and loading the step config for
        a particular step. It is invoked by the internal step runner.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_recipe_config(cls, recipe_config: Dict[str, Any], recipe_root: str) -> "BaseStep":
        """
        Constructs a step class instance by creating a step config using the recipe
        config.
        Subclasses must implement this method to produce the config required to correctly
        run the corresponding step.

        :param recipe_config: Dictionary representation of the full recipe config.
        :param recipe_root: String file path to the recipe root directory.
        :return: class instance of the step.
        """
        pass

    @classmethod
    def from_step_config_path(cls, step_config_path: str, recipe_root: str) -> "BaseStep":
        """
        Constructs a step class instance using the config specified in the
        configuration file.

        :param step_config_path: String path to the step-specific configuration
                                 on the local filesystem.
        :param recipe_root: String path to the recipe root directory on
                              the local filesystem.
        :return: class instance of the step.
        """
        with open(step_config_path) as f:
            step_config = yaml.safe_load(f)
        return cls(step_config, recipe_root)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns back the name of the step for the current class instance. This is used
        downstream by the execution engine to create step-specific directory structures.
        """
        pass

    @property
    def environment(self) -> Dict[str, str]:
        """
        Returns environment variables associated with step that should be set when the
        step is executed.
        """
        return {}

    def get_artifacts(self) -> List[Any]:
        """
        Returns the named artifacts produced by the step for the current class instance.
        """
        return {}

    @abc.abstractmethod
    def step_class(self) -> StepClass:
        """
        Returns the step class.
        """
        pass

    def get_execution_state(self, output_directory: str) -> StepExecutionState:
        """
        Returns the execution state of the step, which provides information about its
        status (succeeded, failed, unknown), last update time, and, if applicable, encountered
        stacktraces.

        :param output_directory: String file path to the directory where step
                                 outputs are stored.
        :return: A ``StepExecutionState`` instance containing the step execution state.
        """
        execution_state_file_path = os.path.join(
            output_directory, BaseStep._EXECUTION_STATE_FILE_NAME
        )
        if os.path.exists(execution_state_file_path):
            with open(execution_state_file_path) as f:
                return StepExecutionState.from_dict(json.load(f))
        else:
            return StepExecutionState(StepStatus.UNKNOWN, 0, None)

    def _serialize_card(self, start_timestamp: float, output_directory: str) -> None:
        if self.step_card is None:
            return
        execution_duration = time.time() - start_timestamp
        tab = self.step_card.get_tab("Run Summary")
        if tab is not None:
            tab.add_markdown("EXE_DURATION", f"**Run duration (s)**: {execution_duration:.3g}")
            tab.add_markdown(
                "LAST_UPDATE_TIME",
                f"**Last updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
            )
        self.step_card.save(path=output_directory)
        self.step_card.save_as_html(path=output_directory)

    def _update_status(
        self, status: StepStatus, output_directory: str, stack_trace: Optional[str] = None
    ) -> None:
        execution_state = StepExecutionState(
            status=status, last_updated_timestamp=time.time(), stack_trace=stack_trace
        )
        with open(os.path.join(output_directory, BaseStep._EXECUTION_STATE_FILE_NAME), "w") as f:
            json.dump(execution_state.to_dict(), f)

    def _initialize_databricks_spark_connection_and_hooks_if_applicable(self) -> None:
        """
        Initializes a connection to the Databricks Spark Gateway and sets up associated hooks
        (e.g. MLflow Run creation notification hooks) if MLflow Recipes is running in the
        Databricks Runtime.
        """
        if is_in_databricks_runtime():
            try:
                from dbruntime.spark_connection import (
                    initialize_spark_connection,
                    is_pinn_mode_enabled,
                )
                from IPython.utils.io import capture_output

                with capture_output():
                    spark_handles, entry_point = initialize_spark_connection(is_pinn_mode_enabled())
            except Exception as e:
                _logger.warning(
                    "Encountered unexpected failure while initializing Spark connection. Spark"
                    " operations may not succeed. Exception: %s",
                    e,
                )
            else:
                try:
                    from dbruntime.MlflowCreateRunHook import get_mlflow_create_run_hook

                    # `get_mlflow_create_run_hook` sets up a patch to trigger a Databricks command
                    # notification every time an MLflow Run is created. This notification is
                    # visible to users in notebook environments
                    get_mlflow_create_run_hook(spark_handles["sc"], entry_point)
                except Exception as e:
                    _logger.warning(
                        "Encountered unexpected failure while setting up Databricks MLflow Run"
                        " creation hooks. Exception: %s",
                        e,
                    )

    def _log_step_card(self, run_id: str, step_name: str) -> None:
        """
        Logs a step card as an artifact (destination: <step_name>/card.html) in a specified run.
        If the step card does not exist, logging is skipped.

        :param run_id: Run ID to which the step card is logged.
        :param step_name: Step name.
        """
        from mlflow.recipes.utils.execution import get_step_output_path

        local_card_path = get_step_output_path(
            recipe_root_path=self.recipe_root,
            step_name=step_name,
            relative_path=CARD_HTML_NAME,
        )
        if os.path.exists(local_card_path):
            MlflowClient().log_artifact(run_id, local_card_path, artifact_path=step_name)
        else:
            _logger.warning(
                "Failed to log step card for step %s. Run ID: %s. Card local path: %s",
                step_name,
                run_id,
                local_card_path,
            )

    @staticmethod
    def _generate_worst_examples_dataframe(
        dataframe,
        predictions,
        error,
        target_col,
        worst_k=10,
    ):
        """
        Generate dataframe containing worst k examples with largest prediction error.
        Dataframe contains columns of all features, prediction, error, and target_col column.
        The prediction error is defined as absolute error between target value and
        prediction value.
        """
        import numpy as np

        predictions = np.array(predictions)
        abs_error = np.absolute(error)
        worst_k_indexes = np.argsort(abs_error)[::-1][:worst_k]
        result_df = dataframe.iloc[worst_k_indexes].assign(
            prediction=predictions[worst_k_indexes],
            absolute_error=abs_error[worst_k_indexes],
        )
        front_columns = ["absolute_error", "prediction", target_col]
        reordered_columns = front_columns + result_df.columns.drop(front_columns).tolist()
        return result_df[reordered_columns].reset_index(drop=True)
