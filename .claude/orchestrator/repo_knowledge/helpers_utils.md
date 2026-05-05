---
name: helpers_utils
description: Auto-generated public-symbol reference for `mlflow/utils/`. Use this before suggesting a new helper.
applies_to: any PR that touches mlflow/utils/; introduces a new utility helper anywhere in the repo; or could plausibly reuse an existing utility (lazy_load, annotations, rest_utils, file_utils, etc.).
last_verified: 2026-05-05
citation_policy: each `path:line` is the `def` / `class` line. If the snippet drifts, search by symbol name.
generated_by: .claude/orchestrator/scripts/generate_helpers_md.py (refreshed weekly by .github/workflows/refresh-helpers.yml).
---

# Helpers: `mlflow/utils/`

Auto-generated. Walks `mlflow/utils/` and lists every public symbol with its signature and first docstring sentence.

## How to use this file

- **Before suggesting a new utility function in a review**, grep this file for the area you're touching. If a helper already exists, point at its `path:line` instead of asking for a new one.
- **Class entries** list public methods in the same row group (`ClassName.method` form).
- **Search by symbol name**, not by line number: line numbers drift after reformats.

## `mlflow/utils/annotations.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `experimental` | function | `(f: Callable[P, R], version: str \| None, *, skip: bool) -> Callable[P, R]` |  | 34 |
| `experimental` | function | `(f: None, version: str \| None, *, skip: bool) -> Callable[[Callable[P, R]], Callable[P, R]]` |  | 43 |
| `experimental` | function | `(f: Callable[P, R] \| None, version: str \| None, *, skip: bool) -> Callable[[Callable[P, R]], Callable[P, R]]` | Decorator / decorator creator for marking APIs experimental in the docstring. | 51 |
| `developer_stable` | function | `(func)` | The API marked here as `@developer_stable` has certain protections associated with future development work. | 104 |
| `mark_deprecated` | function | `(func)` | Mark a function as deprecated by setting a private attribute on it. | 134 |
| `is_marked_deprecated` | function | `(func)` | Is the function marked as deprecated. | 141 |
| `deprecated` | function | `(alternative: str \| None, since: str \| None, impact: str \| None)` | Annotation decorator for marking APIs as deprecated in docstrings and raising a warning if called. | 148 |
| `deprecated_parameter` | function | `(old_param: str, new_param: str, version: str \| None)` | Decorator to handle deprecated parameter renaming with automatic warning and forwarding. | 213 |
| `keyword_only` | function | `(func)` | A decorator that forces keyword arguments in the wrapped method. | 290 |
| `filter_user_warnings_once` | function | `(func)` | A decorator that filter user warnings to only show once in the wrapped method. | 306 |
| `requires_sql_backend` | function | `(func)` | Decorator for marking APIs that require a SQL-based tracking backend. | 318 |

## `mlflow/utils/async_logging/async_artifacts_logging_queue.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `AsyncArtifactsLoggingQueue` | class | `` | This is a queue based run data processor that queue incoming data and process it using a single worker thread. | 22 |
| `AsyncArtifactsLoggingQueue.flush` | method | `(self) -> None` | Flush the async logging queue. | 60 |
| `AsyncArtifactsLoggingQueue.log_artifacts_async` | method | `(self, filename, artifact_path, artifact) -> RunOperations` | Asynchronously logs runs artifacts. | 184 |
| `AsyncArtifactsLoggingQueue.is_active` | method | `(self) -> bool` |  | 216 |
| `AsyncArtifactsLoggingQueue.activate` | method | `(self) -> None` | Activates the async logging queue  1. | 241 |

## `mlflow/utils/async_logging/async_logging_queue.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `QueueStatus` | class | `(enum.Enum)` | Status of the async queue | 31 |
| `AsyncLoggingQueue` | class | `` | This is a queue based run data processor that queues incoming batches and processes them using single worker thread. | 47 |
| `AsyncLoggingQueue.end_async_logging` | method | `(self) -> None` |  | 86 |
| `AsyncLoggingQueue.shut_down_async_logging` | method | `(self) -> None` | Shut down the async logging queue and wait for the queue to be drained. | 98 |
| `AsyncLoggingQueue.flush` | method | `(self) -> None` | Flush the async logging queue and restart thread to listen to incoming data after flushing. | 108 |
| `AsyncLoggingQueue.log_batch_async` | method | `(self, run_id: str, params: list[Param], tags: list[RunTag], metrics: list[Metric]) -> RunOperations` | Asynchronously logs a batch of run data (parameters, tags, and metrics). | 284 |
| `AsyncLoggingQueue.is_active` | method | `(self) -> bool` |  | 317 |
| `AsyncLoggingQueue.is_idle` | method | `(self) -> bool` |  | 320 |
| `AsyncLoggingQueue.activate` | method | `(self) -> None` | Activates the async logging queue  1. | 349 |

## `mlflow/utils/async_logging/run_artifact.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `RunArtifact` | class | `` |  | 8 |
| `RunArtifact.exception` | method | `(self, exception)` |  | 37 |

## `mlflow/utils/async_logging/run_batch.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `RunBatch` | class | `` |  | 8 |
| `RunBatch.exception` | method | `(self, exception)` |  | 40 |
| `RunBatch.add_child_batch` | method | `(self, child_batch)` | Add a child batch to the current batch. | 43 |
| `RunBatch.complete` | method | `(self)` | Mark the batch as completed. | 51 |

## `mlflow/utils/async_logging/run_operations.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `RunOperations` | class | `` | Class that helps manage the futures of MLflow async logging. | 1 |
| `RunOperations.wait` | method | `(self)` | Blocks on completion of all futures. | 7 |
| `get_combined_run_operations` | function | `(run_operations_list: list[RunOperations]) -> RunOperations` | Combine a list of RunOperations objects into a single RunOperations object. | 25 |

## `mlflow/utils/autologging_utils/__init__.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `autologging_conf_lock` | function | `(fn)` | Apply a global lock on functions that enable / disable autologging. | 71 |
| `get_mlflow_run_params_for_fn_args` | function | `(fn, args, kwargs, unlogged)` | Given arguments explicitly passed to a function, generate a dictionary of MLflow Run parameter key / value pairs. | 83 |
| `log_fn_args_as_params` | function | `(fn, args, kwargs, unlogged)` | Log arguments explicitly passed to a function as MLflow Run parameters to the current active MLflow Run. | 123 |
| `InputExampleInfo` | class | `` | Stores info about the input example collection before it is needed. | 143 |
| `resolve_input_example_and_signature` | function | `(get_input_example, infer_model_signature, log_input_example, log_model_signature, logger)` | Handles the logic of calling functions to gather the input example and infer the model signature. | 158 |
| `BatchMetricsLogger` | class | `` | The BatchMetricsLogger will log metrics in batch against an mlflow run. | 225 |
| `BatchMetricsLogger.flush` | method | `(self)` | The metrics accumulated by BatchMetricsLogger will be batch logged to an MLflow run. | 249 |
| `BatchMetricsLogger.record_metrics` | method | `(self, metrics, step)` | Submit a set of metrics to be logged. | 280 |
| `batch_metrics_logger` | function | `(run_id: str \| None, model_id: str \| None)` | Context manager that yields a BatchMetricsLogger object, which metrics can be logged against. | 314 |
| `gen_autologging_package_version_requirements_doc` | function | `(integration_name)` | Returns:     A document note string saying the compatibility for the specified autologging     integration's associated package... | 337 |
| `autologging_integration` | function | `(name)` | **All autologging integrations should be decorated with this wrapper.**  Wraps an autologging function in order to store its co... | 385 |
| `get_autologging_config` | function | `(flavor_name, config_key, default_value)` | Returns a desired config value for a specified autologging integration. | 482 |
| `autologging_is_disabled` | function | `(integration_name)` | Returns a boolean flag of whether the autologging integration is disabled. | 501 |
| `is_autolog_supported` | function | `(integration_name: str) -> bool` | Whether the specified autologging integration is supported by the current environment. | 522 |
| `get_autolog_function` | function | `(integration_name: str) -> Callable[..., Any] \| None` | Get the autolog() function for the specified integration. | 534 |
| `disable_autologging` | function | `()` | Context manager that temporarily disables autologging globally for all integrations upon entry and restores the previous autolo... | 544 |
| `disable_discrete_autologging` | function | `(flavors_to_disable: list[str]) -> None` | Context manager for disabling specific autologging integrations temporarily while another flavor's autologging is activated. | 558 |
| `get_instance_method_first_arg_value` | function | `(method, call_pos_args, call_kwargs)` | Get instance method first argument value (exclude the `self` argument). | 689 |
| `get_method_call_arg_value` | function | `(arg_index, arg_name, default_value, call_pos_args, call_kwargs)` | Get argument value for a method call. | 709 |

## `mlflow/utils/autologging_utils/client.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `PendingRunId` | class | `` | Serves as a placeholder for the ID of a run that does not yet exist, enabling additional metadata (e.g. | 50 |
| `RunOperations` | class | `` | Represents a collection of operations on one or more MLflow Runs, such as run creation or metric logging. | 57 |
| `RunOperations.await_completion` | method | `(self)` | Blocks on completion of the MLflow Run operations. | 66 |
| `MlflowAutologgingQueueingClient` | class | `` | Efficiently implements a subset of MLflow Tracking's  `MlflowClient` and fluent APIs to provide automatic batching and async ex... | 100 |
| `MlflowAutologgingQueueingClient.create_run` | method | `(self, experiment_id: str, start_time: int \| None, tags: dict[str, Any] \| None, run_name: str \| None) ->...` | Enqueues a CreateRun operation with the specified attributes, returning a `PendingRunId` instance that can be used as input to ... | 150 |
| `MlflowAutologgingQueueingClient.set_terminated` | method | `(self, run_id: str \| PendingRunId, status: str \| None, end_time: int \| None) -> None` | Enqueues an UpdateRun operation with the specified `status` and `end_time` attributes for the specified `run_id`. | 181 |
| `MlflowAutologgingQueueingClient.log_params` | method | `(self, run_id: str \| PendingRunId, params: dict[str, Any]) -> None` | Enqueues a collection of Parameters to be logged to the run specified by `run_id`. | 195 |
| `MlflowAutologgingQueueingClient.log_inputs` | method | `(self, run_id: str \| PendingRunId, datasets: list[DatasetInput] \| None) -> None` | Enqueues a collection of Dataset to be logged to the run specified by `run_id`. | 205 |
| `MlflowAutologgingQueueingClient.log_metrics` | method | `(self, run_id: str \| PendingRunId, metrics: dict[str, float], step: int \| None, dataset: Optional['Datase...` | Enqueues a collection of Metrics to be logged to the run specified by `run_id` at the step specified by `step`. | 213 |
| `MlflowAutologgingQueueingClient.set_tags` | method | `(self, run_id: str \| PendingRunId, tags: dict[str, Any]) -> None` | Enqueues a collection of Tags to be logged to the run specified by `run_id`. | 241 |
| `MlflowAutologgingQueueingClient.flush` | method | `(self, synchronous)` | Flushes all queued run operations, resulting in the creation or mutation of runs and run data. | 251 |

## `mlflow/utils/autologging_utils/config.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `AutoLoggingConfig` | class | `` | A dataclass to hold common autologging configuration options. | 11 |
| `AutoLoggingConfig.init` | classmethod | `(cls, flavor_name: str)` |  | 23 |

## `mlflow/utils/autologging_utils/events.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `AutologgingEventLoggerWrapper` | class | `` | A wrapper around AutologgingEventLogger for DRY:   - Store common arguments to avoid passing them to each logger method   - Cat... | 19 |
| `AutologgingEventLoggerWrapper.log_patch_function_start` | method | `(self, args, kwargs)` |  | 36 |
| `AutologgingEventLoggerWrapper.log_patch_function_success` | method | `(self, args, kwargs)` |  | 42 |
| `AutologgingEventLoggerWrapper.log_patch_function_error` | method | `(self, args, kwargs, exception)` |  | 48 |
| `AutologgingEventLoggerWrapper.log_original_function_start` | method | `(self, args, kwargs)` |  | 54 |
| `AutologgingEventLoggerWrapper.log_original_function_success` | method | `(self, args, kwargs)` |  | 60 |
| `AutologgingEventLoggerWrapper.log_original_function_error` | method | `(self, args, kwargs, exception)` |  | 66 |
| `AutologgingEventLogger` | class | `` | Provides instrumentation hooks for important autologging lifecycle events, including:      - Calls to `mlflow.autolog()` APIs  ... | 72 |
| `AutologgingEventLogger.get_logger` | staticmethod | `()` | Fetches the configured `AutologgingEventLogger` instance for logging. | 94 |
| `AutologgingEventLogger.set_logger` | staticmethod | `(logger)` | Configures the `AutologgingEventLogger` instance for logging. | 106 |
| `AutologgingEventLogger.log_autolog_called` | method | `(self, integration, call_args, call_kwargs)` | Called when the `autolog()` method for an autologging integration is invoked (e.g., when a user invokes `mlflow.sklearn.autolog... | 117 |
| `AutologgingEventLogger.log_patch_function_start` | method | `(self, session, patch_obj, function_name, call_args, call_kwargs)` | Called upon invocation of a patched API associated with an autologging integration (e.g., `sklearn.linear_model.LogisticRegress... | 145 |
| `AutologgingEventLogger.log_patch_function_success` | method | `(self, session, patch_obj, function_name, call_args, call_kwargs)` | Called upon successful termination of a patched API associated with an autologging integration (e.g., `sklearn.linear_model.Log... | 166 |
| `AutologgingEventLogger.log_patch_function_error` | method | `(self, session, patch_obj, function_name, call_args, call_kwargs, exception)` | Called when execution of a patched API associated with an autologging integration (e.g., `sklearn.linear_model.LogisticRegressi... | 188 |
| `AutologgingEventLogger.log_original_function_start` | method | `(self, session, patch_obj, function_name, call_args, call_kwargs)` | Called during the execution of a patched API associated with an autologging integration when the original / underlying API is i... | 213 |
| `AutologgingEventLogger.log_original_function_success` | method | `(self, session, patch_obj, function_name, call_args, call_kwargs)` | Called during the execution of a patched API associated with an autologging integration when the original / underlying API invo... | 239 |
| `AutologgingEventLogger.log_original_function_error` | method | `(self, session, patch_obj, function_name, call_args, call_kwargs, exception)` | Called during the execution of a patched API associated with an autologging integration when the original / underlying API invo... | 267 |

## `mlflow/utils/autologging_utils/logging_and_warnings.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `NonMlflowWarningsBehaviorForCurrentThread` | class | `` | Context manager that modifies the behavior of non-MLflow warnings upon entry, according to the specified parameters. | 184 |
| `MlflowEventsAndWarningsBehaviorGlobally` | class | `` | Threadsafe context manager that modifies the behavior of MLflow event logging statements and MLflow warnings upon entry, accord... | 242 |

## `mlflow/utils/autologging_utils/metrics_queue.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `flush_metrics_queue` | function | `()` | Flush the metric queue and log contents in batches to MLflow. | 26 |
| `add_to_metrics_queue` | function | `(key, value, step, time, run_id)` | Add a metric to the metric queue. | 58 |

## `mlflow/utils/autologging_utils/safety.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `exception_safe_function_for_class` | function | `(function)` | Wraps the specified function with broad exception handling to guard against unexpected errors during autologging. | 36 |
| `picklable_exception_safe_function` | function | `(function)` | Wraps the specified function with broad exception handling to guard against unexpected errors during autologging while preservi... | 70 |
| `with_managed_run` | function | `(autologging_integration, patch_function, tags)` | Given a `patch_function`, returns an `augmented_patch_function` that wraps the execution of `patch_function` with an active MLf... | 134 |
| `is_testing` | function | `()` | Indicates whether or not autologging functionality is running in test mode (as determined by the `MLFLOW_AUTOLOGGING_TESTING` e... | 198 |
| `safe_patch` | function | `(autologging_integration, destination, function_name, patch_function, manage_run, extra_tags)` | Patches the specified `function_name` on the specified `destination` class for autologging purposes, preceding its implementati... | 231 |
| `revert_patches` | function | `(autologging_integration)` | Reverts all patches on the specified destination class for autologging disablement purposes. | 709 |
| `AutologgingSession` | class | `` |  | 737 |
| `update_wrapper_extended` | function | `(wrapper, wrapped)` | Update a `wrapper` function to look like the `wrapped` function. | 785 |
| `ValidationExemptArgument` | class | `(NamedTuple)` | A NamedTuple representing the properties of an argument that is exempt from validation  autologging_integration: The name of th... | 862 |
| `ValidationExemptArgument.matches` | method | `(self, autologging_integration, function_name, value, argument_index, argument_name)` | This method checks if the properties provided through the function arguments matches the properties defined in the NamedTuple. | 880 |

## `mlflow/utils/autologging_utils/versioning.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_min_max_version_and_pip_release` | function | `(flavor_name: str, category: Literal['autologging', 'models'])` |  | 46 |
| `is_flavor_supported_for_associated_package_versions` | function | `(flavor_name, check_max_version)` | Returns:     True if the specified flavor is supported for the currently-installed versions of its     associated packages. | 59 |

## `mlflow/utils/checkpoint_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `MlflowModelCheckpointCallbackBase` | class | `` | Callback base class for automatic model checkpointing to MLflow. | 25 |
| `MlflowModelCheckpointCallbackBase.save_checkpoint` | method | `(self, filepath: str)` |  | 94 |
| `MlflowModelCheckpointCallbackBase.check_and_save_checkpoint_if_needed` | method | `(self, current_epoch, global_step, metric_dict)` |  | 97 |
| `download_checkpoint_artifact` | function | `(run_id, epoch, global_step, dst_path)` |  | 168 |

## `mlflow/utils/conda.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_conda_command` | function | `(conda_env_name)` |  | 19 |
| `get_conda_bin_executable` | function | `(executable_name)` | Return path to the specified executable, assumed to be discoverable within the 'bin' subdirectory of a conda installation. | 36 |
| `get_or_create_conda_env` | function | `(conda_env_path, env_id, capture_output, env_root_dir, pip_requirements_override, extra_envs)` | Given a `Project`, creates a conda environment containing the project's dependencies if such a conda environment doesn't alread... | 212 |

## `mlflow/utils/credentials.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `MlflowCreds` | class | `(NamedTuple)` |  | 23 |
| `read_mlflow_creds` | function | `() -> MlflowCreds` |  | 52 |
| `get_default_host_creds` | function | `(store_uri)` |  | 61 |
| `login` | function | `(backend: str, interactive: bool) -> None` | Configure MLflow server authentication and connect MLflow to tracking server. | 76 |

## `mlflow/utils/crypto.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `AESGCMResult` | class | `` | Result of AES-GCM encryption operation. | 87 |
| `EncryptedSecret` | class | `` | Result of secret encryption using envelope encryption. | 101 |
| `RotatedSecret` | class | `` | Result of KEK rotation for a secret. | 119 |
| `KEKManager` | class | `` | Manages Key Encryption Keys (KEK) for MLflow encrypted data (API Keys, etc.). | 135 |
| `KEKManager.get_kek` | method | `(self) -> bytes` | Get the derived KEK. | 212 |
| `decrypt_with_aes_gcm` | function | `(ciphertext: bytes, key: bytes, aad: bytes \| None) -> bytes` | Decrypt ciphertext using AES-256-GCM. | 315 |
| `wrap_dek` | function | `(dek: bytes, kek: bytes) -> bytes` | Wrap (encrypt) a DEK with the KEK using AES-256-GCM. | 359 |
| `unwrap_dek` | function | `(wrapped_dek: bytes, kek: bytes) -> bytes` | Unwrap (decrypt) a DEK using the KEK. | 376 |
| `rotate_secret_encryption` | function | `(encrypted_value: bytes, wrapped_dek: bytes, old_kek_manager: KEKManager, new_kek_manager: KEKManager) -> R...` | Rotate a secret's encryption from old KEK to new KEK. | 600 |

## `mlflow/utils/data_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `parse_s3_uri` | function | `(uri)` | Parse an S3 URI, returning (bucket, path) | 5 |
| `is_uri` | function | `(string)` |  | 15 |
| `is_polars_dataframe` | function | `(data: Any) -> bool` |  | 20 |

## `mlflow/utils/databricks_sql_warehouse.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `ensure_sql_warehouse_running` | function | `(warehouse_id: str) -> None` | Verify the SQL warehouse is in ``RUNNING`` state, starting it and waiting if necessary. | 34 |

## `mlflow/utils/databricks_tracing_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `parse_uc_location` | function | `(location: str) -> tuple[str, str, str \| None]` |  | 26 |
| `uc_location_to_str` | function | `(catalog: str, schema: str, table_prefix: str \| None) -> str` |  | 37 |
| `uc_schema_location_to_proto` | function | `(uc_schema_location: UCSchemaLocation) -> pb.UCSchemaLocation` |  | 41 |
| `uc_schema_location_from_proto` | function | `(proto: pb.UCSchemaLocation) -> UCSchemaLocation` |  | 52 |
| `uc_table_prefix_location_to_proto` | function | `(location: UnityCatalog) -> pb.UcTablePrefixLocation` |  | 67 |
| `uc_table_prefix_location_from_proto` | function | `(proto: pb.UcTablePrefixLocation) -> UnityCatalog` |  | 84 |
| `inference_table_location_to_proto` | function | `(inference_table_location: InferenceTableLocation) -> pb.InferenceTableLocation` |  | 99 |
| `mlflow_experiment_location_to_proto` | function | `(mlflow_experiment_location: MlflowExperimentLocation) -> pb.MlflowExperimentLocation` |  | 105 |
| `trace_location_to_proto` | function | `(trace_location: TraceLocation) -> pb.TraceLocation` |  | 111 |
| `trace_location_type_from_proto` | function | `(proto: pb.TraceLocation.TraceLocationType) -> TraceLocationType` |  | 136 |
| `trace_location_from_proto` | function | `(proto: pb.TraceLocation) -> TraceLocation` |  | 140 |
| `trace_info_to_v4_proto` | function | `(trace_info: TraceInfo) -> pb.TraceInfo` |  | 167 |
| `trace_to_proto` | function | `(trace: Trace) -> pb.Trace` |  | 195 |
| `trace_from_proto` | function | `(proto: pb.Trace, location_id: str) -> Trace` |  | 202 |
| `assessment_to_proto` | function | `(assessment: Assessment) -> pb.Assessment` |  | 209 |
| `get_trace_id_from_assessment_proto` | function | `(proto: pb.Assessment \| assessments_pb2.Assessment) -> str` |  | 266 |

## `mlflow/utils/databricks_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_mlflow_credential_context_by_run_id` | function | `(run_id)` |  | 93 |
| `MlflowCredentialContext` | class | `` | Sets and clears credentials on a context using the provided profile URL. | 102 |
| `acl_path_of_acl_root` | function | `()` |  | 174 |
| `is_databricks_default_tracking_uri` | function | `(tracking_uri)` |  | 191 |
| `is_in_databricks_notebook` | function | `()` |  | 196 |
| `is_in_databricks_job` | function | `()` |  | 206 |
| `is_in_databricks_model_serving_environment` | function | `()` | Check if the code is running in Databricks Model Serving environment. | 213 |
| `is_mlflow_tracing_enabled_in_model_serving` | function | `() -> bool` | This environment variable guards tracing behaviors for models in databricks model serving. | 222 |
| `should_fetch_model_serving_environment_oauth` | function | `()` |  | 232 |
| `is_in_databricks_repo` | function | `()` |  | 240 |
| `is_in_databricks_repo_notebook` | function | `()` |  | 247 |
| `get_databricks_runtime_version` | function | `()` |  | 258 |
| `is_in_databricks_runtime` | function | `()` |  | 277 |
| `is_in_databricks_serverless_runtime` | function | `()` |  | 281 |
| `is_in_databricks_shared_cluster_runtime` | function | `()` |  | 286 |
| `is_databricks_connect` | function | `(spark)` | Return True if current Spark-connect client connects to Databricks cluster. | 296 |
| `DBConnectUDFSandboxInfo` | class | `` |  | 321 |
| `get_dbconnect_udf_sandbox_info` | function | `(spark)` | Get Databricks UDF sandbox info which includes the following fields:  - image_version like   '{major_version}.{minor_version}' ... | 332 |
| `is_databricks_serverless` | function | `(spark)` | Return True if running on Databricks Serverless notebook or on Databricks Connect client that connects to Databricks Serverless. | 402 |
| `is_dbfs_fuse_available` | function | `()` |  | 420 |
| `is_uc_volume_fuse_available` | function | `()` |  | 437 |
| `is_in_cluster` | function | `()` |  | 452 |
| `get_notebook_id` | function | `()` | Should only be called if is_in_databricks_notebook is true | 465 |
| `get_notebook_path` | function | `()` | Should only be called if is_in_databricks_notebook is true | 475 |
| `get_cluster_id` | function | `()` |  | 487 |
| `get_job_group_id` | function | `()` |  | 495 |
| `get_repl_id` | function | `()` | Returns:     The ID of the current Databricks Python REPL. | 506 |
| `get_job_id` | function | `()` |  | 539 |
| `get_job_run_id` | function | `()` |  | 547 |
| `get_job_type` | function | `()` | Should only be called if is_in_databricks_job is true | 555 |
| `get_job_type_info` | function | `()` |  | 564 |
| `get_workload_id` | function | `()` |  | 572 |
| `get_workload_class` | function | `()` |  | 580 |
| `get_webapp_url` | function | `()` | Should only be called if is_in_databricks_notebook or is_in_databricks_jobs is true | 588 |
| `get_workspace_id` | function | `()` |  | 600 |
| `get_browser_hostname` | function | `()` |  | 608 |
| `get_workspace_info_from_dbutils` | function | `()` |  | 615 |
| `get_workspace_url` | function | `()` |  | 638 |
| `warn_on_deprecated_cross_workspace_registry_uri` | function | `(registry_uri)` |  | 644 |
| `get_workspace_info_from_databricks_secrets` | function | `(tracking_uri)` |  | 660 |
| `get_model_dependency_oauth_token` | function | `(should_retry)` |  | 710 |
| `TrackingURIConfigProvider` | class | `(DatabricksConfigProvider)` | TrackingURIConfigProvider extracts `scope` and `key_prefix` from tracking URI of format like `databricks://scope:key_prefix`, t... | 727 |
| `TrackingURIConfigProvider.get_config` | method | `(self)` |  | 743 |
| `get_databricks_host_creds` | function | `(server_uri)` | Reads in configuration necessary to make HTTP requests to a Databricks server. | 756 |
| `check_databricks_sdk_supports_scopes` | function | `()` | Check if the installed databricks-sdk version supports the 'scopes' parameter for WorkspaceClient. | 880 |
| `get_databricks_workspace_client_config` | function | `(server_uri: str, scopes: list[str] \| None)` |  | 905 |
| `get_git_repo_url` | function | `()` |  | 927 |
| `get_git_repo_provider` | function | `()` |  | 935 |
| `get_git_repo_commit` | function | `()` |  | 943 |
| `get_git_repo_relative_path` | function | `()` |  | 951 |
| `get_git_repo_reference` | function | `()` |  | 959 |
| `get_git_repo_reference_type` | function | `()` |  | 967 |
| `get_git_repo_status` | function | `()` |  | 975 |
| `is_running_in_ipython_environment` | function | `()` |  | 982 |
| `get_databricks_run_url` | function | `(tracking_uri: str, run_id: str, artifact_path) -> str \| None` | Obtains a Databricks URL corresponding to the specified MLflow Run, optionally referring to an artifact within the run. | 991 |
| `get_databricks_model_version_url` | function | `(registry_uri: str, name: str, version: str) -> str \| None` | Obtains a Databricks URL corresponding to the specified Model Version. | 1027 |
| `DatabricksWorkspaceInfo` | class | `` |  | 1059 |
| `DatabricksWorkspaceInfo.from_environment` | classmethod | `(cls) -> DatabricksWorkspaceInfoType \| None` |  | 1068 |
| `DatabricksWorkspaceInfo.to_environment` | method | `(self)` |  | 1077 |
| `get_databricks_workspace_info_from_uri` | function | `(tracking_uri: str) -> DatabricksWorkspaceInfo \| None` |  | 1087 |
| `check_databricks_secret_scope_access` | function | `(scope_name)` |  | 1112 |
| `get_sgc_job_run_id` | function | `() -> str \| None` | Retrieves the Serverless GPU Compute (SGC) job run ID from Databricks. | 1127 |
| `get_databricks_env_vars` | function | `(tracking_uri)` |  | 1290 |
| `DatabricksRuntimeVersion` | class | `(NamedTuple)` |  | 1351 |
| `DatabricksRuntimeVersion.parse` | classmethod | `(cls, databricks_runtime: str \| None)` |  | 1358 |
| `get_databricks_runtime_major_minor_version` | function | `()` |  | 1379 |
| `get_databricks_nfs_temp_dir` | function | `()` |  | 1514 |
| `get_databricks_local_temp_dir` | function | `()` |  | 1528 |
| `stage_model_for_databricks_model_serving` | function | `(model_name: str, model_version: str)` |  | 1542 |
| `databricks_api_disabled` | function | `(api_name: str, alternative: str \| None)` | Decorator that disables an API method when used with Databricks. | 1560 |
| `invoke_databricks_app` | function | `(app_invocation_url: str, payload: dict[str, Any], config) -> dict[str, Any]` | Invoke Databricks App /invocations endpoint with OAuth authentication. | 1598 |

## `mlflow/utils/docstring_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `ParamDocs` | class | `(dict)` | Represents a set of parameter documents in the docstring. | 65 |
| `ParamDocs.format` | method | `(self, **kwargs)` | Formats values to be substituted in via the format_docstring() method. | 73 |
| `ParamDocs.format_docstring` | method | `(self, docstring: str) -> str` | Formats placeholders in `docstring`. | 93 |
| `format_docstring` | function | `(param_docs)` | Returns a decorator that replaces param doc placeholders (e.g. | 124 |
| `get_module_min_and_max_supported_ranges` | function | `(flavor_name)` | Extracts the minimum and maximum supported package versions from the provided module name. | 455 |
| `docstring_version_compatibility_warning` | function | `(integration_name)` | Generates a docstring that can be applied as a note stating a version compatibility range for a given flavor and optionally rai... | 486 |

## `mlflow/utils/doctor.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `doctor` | function | `(mask_envs)` | Prints out useful information for debugging issues with MLflow. | 13 |

## `mlflow/utils/download_cloud_file_chunk.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `parse_args` | function | `()` |  | 12 |
| `main` | function | `()` |  | 22 |

## `mlflow/utils/env_manager.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `validate` | function | `(env_manager)` |  | 10 |

## `mlflow/utils/env_pack.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `EnvPackConfig` | class | `` |  | 25 |
| `pack_env_for_databricks_model_serving` | function | `(model_uri: str, *, enforce_pip_requirements: bool, local_model_path: str \| None) -> Generator[str, None, ...` | Generate Databricks artifacts for fast deployment. | 117 |

## `mlflow/utils/environment.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `infer_pip_requirements` | function | `(model_uri, flavor, fallback, timeout, extra_env_vars, uv_project_dir, uv_groups, uv_extras)` | Infers the pip requirements of the specified model by creating a subprocess and loading the model in it to determine which pack... | 410 |
| `Environment` | class | `` |  | 1078 |
| `Environment.get_activate_command` | method | `(self)` |  | 1085 |
| `Environment.execute` | method | `(self, command, command_env, preexec_fn, capture_output, stdout, stderr, stdin, synchronous)` |  | 1088 |

## `mlflow/utils/exception_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_stacktrace` | function | `(error)` |  | 4 |

## `mlflow/utils/file_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `ArtifactProgressBar` | class | `` |  | 64 |
| `ArtifactProgressBar.set_pbar` | method | `(self)` |  | 73 |
| `ArtifactProgressBar.chunks` | classmethod | `(cls, file_size, desc, chunk_size)` |  | 83 |
| `ArtifactProgressBar.files` | classmethod | `(cls, desc, total)` |  | 98 |
| `ArtifactProgressBar.update` | method | `(self)` |  | 103 |
| `is_directory` | function | `(name)` |  | 118 |
| `is_file` | function | `(name)` |  | 122 |
| `exists` | function | `(name)` |  | 126 |
| `list_all` | function | `(root, filter_func, full_path)` | List all entities directly under 'dir_name' that satisfy 'filter_func'  Args:     root: Name of directory to start search. | 130 |
| `list_subdirs` | function | `(dir_name, full_path)` | Equivalent to UNIX command:   ``find $dir_name -depth 1 -type d``  Args:     dir_name: Name of directory to start search. | 148 |
| `list_files` | function | `(dir_name, full_path)` | Equivalent to UNIX command:   ``find $dir_name -depth 1 -type f``  Args:     dir_name: Name of directory to start search. | 163 |
| `find` | function | `(root, name, full_path)` | Search for a file in a root directory. | 178 |
| `mkdir` | function | `(root, name)` | Make directory with name "root/name", or just "root" if name is None. | 194 |
| `make_containing_dirs` | function | `(path)` | Create the base directory for a given file path if it does not exist; also creates parent directories. | 213 |
| `TempDir` | class | `` |  | 223 |
| `TempDir.path` | method | `(self, *path)` |  | 247 |
| `read_file_lines` | function | `(parent_path, file_name)` | Return the contents of the file as an array where each element is a separate line. | 251 |
| `read_file` | function | `(parent_path, file_name)` | Return the contents of the file. | 267 |
| `get_file_info` | function | `(path, rel_path)` | Returns file meta data : location, size, . | 283 |
| `mv` | function | `(target, new_parent)` |  | 299 |
| `write_to` | function | `(filename, data)` |  | 303 |
| `append_to` | function | `(filename, data)` |  | 308 |
| `make_tarfile` | function | `(output_filename, source_dir, archive_name, custom_filter)` |  | 313 |
| `get_parent_dir` | function | `(path)` |  | 419 |
| `relative_path_to_artifact_path` | function | `(path)` |  | 423 |
| `path_to_local_file_uri` | function | `(path)` | Convert local filesystem path to local file uri. | 431 |
| `path_to_local_sqlite_uri` | function | `(path)` | Convert local filesystem path to sqlite uri. | 438 |
| `local_file_uri_to_path` | function | `(uri)` | Convert URI to local filesystem path. | 447 |
| `get_local_path_or_none` | function | `(path_or_uri)` | Check if the argument is a local path (no scheme or file:///) and return local path if true, None otherwise. | 462 |
| `download_file_using_http_uri` | function | `(http_uri, download_path, chunk_size, headers)` | Downloads a file specified using the `http_uri` to a local `download_path`. | 473 |
| `parallelized_download_file_using_http_uri` | function | `(thread_pool_executor, http_uri, download_path, remote_file_path, file_size, uri_type, chunk_size, env, hea...` | Downloads a file specified using the `http_uri` to a local `download_path`. | 508 |
| `download_chunk_retries` | function | `(*, chunks, http_uri, headers, download_path)` |  | 607 |
| `create_tmp_dir` | function | `()` |  | 667 |
| `get_or_create_tmp_dir` | function | `()` | Get or create a temporary directory which will be removed once python process exit. | 676 |
| `get_or_create_nfs_tmp_dir` | function | `()` | Get or create a temporary NFS directory which will be removed once python process exit. | 708 |
| `shutil_copytree_without_file_permissions` | function | `(src_dir, dst_dir)` | Copies the directory src_dir into dst_dir, without preserving filesystem permissions | 741 |
| `contains_path_separator` | function | `(path)` | Returns True if a path contains a path separator, False otherwise. | 762 |
| `contains_percent` | function | `(path)` | Returns True if a path contains a percent character, False otherwise. | 769 |
| `read_chunk` | function | `(path: os.PathLike, size: int, start_byte: int) -> bytes` | Read a chunk of bytes from a file. | 776 |
| `remove_on_error` | function | `(path: os.PathLike, onerror)` | A context manager that removes a file or directory if an exception is raised during execution. | 795 |
| `get_total_file_size` | function | `(path: str \| pathlib.Path) -> int \| None` | Return the size of all files under given path, including files in subdirectories. | 822 |
| `write_yaml` | function | `(root: str, file_name: str, data: dict[str, Any], overwrite: bool, sort_keys: bool, ensure_yaml_extension: ...` | NEVER TOUCH THIS FUNCTION. | 854 |
| `read_yaml` | function | `(root: str, file_name: str) -> dict[str, Any]` | NEVER TOUCH THIS FUNCTION. | 878 |
| `ExclusiveFileLock` | class | `` | Exclusive file lock (only works on Unix system) | 889 |
| `check_tarfile_security` | function | `(archive_path: str) -> None` | Check the tar file content. | 923 |

## `mlflow/utils/git_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_git_repo_url` | function | `(path: str) -> str \| None` | Obtains the url of the git repository associated with the specified path, returning ``None`` if the path does not correspond to... | 7 |
| `get_git_commit` | function | `(path: str) -> str \| None` | Obtains the hash of the latest commit on the current branch of the git repository associated with the specified path, returning... | 29 |
| `get_git_branch` | function | `(path: str) -> str \| None` | Obtains the name of the current branch of the git repository associated with the specified path, returning ``None`` if the path... | 55 |

## `mlflow/utils/gorilla.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `default_filter` | function | `(name, obj)` | Attribute filter. | 71 |
| `DecoratorData` | class | `` | Decorator data. | 95 |
| `Settings` | class | `` | Define the patching behaviour. | 117 |
| `Patch` | class | `` | Describe all the information required to apply a patch. | 170 |
| `apply` | function | `(patch)` | Apply a patch. | 263 |
| `revert` | function | `(patch)` | Revert a patch. | 329 |
| `patch` | function | `(destination, name, settings)` | Decorator to create a patch. | 378 |
| `destination` | function | `(value)` | Modifier decorator to update a patch's destination. | 415 |
| `name` | function | `(value)` | Modifier decorator to update a patch's name. | 441 |
| `settings` | function | `(**kwargs)` | Modifier decorator to update a patch's settings. | 467 |
| `filter` | function | `(value)` | Modifier decorator to force the inclusion or exclusion of an attribute. | 493 |
| `find_patches` | function | `(modules, recursive)` | Find all the patches created through decorators. | 521 |
| `get_original_attribute` | function | `(obj, name, bypass_descriptor_protocol)` | Retrieve an overridden attribute that has been stored. | 562 |
| `get_decorator_data` | function | `(obj, set_default)` | Retrieve any decorator data from an object. | 640 |

## `mlflow/utils/huggingface_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_latest_commit_for_repo` | function | `(repo: str) -> str` | Fetches the latest commit hash for a repository from the HuggingFace model hub. | 17 |
| `is_valid_hf_repo_id` | function | `(maybe_repo_id: str \| None) -> bool` | Check if the given string is a valid HuggingFace repo identifier e.g. | 60 |

## `mlflow/utils/import_hooks/__init__.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `synchronized` | function | `(lock)` |  | 26 |
| `register_generic_import_hook` | function | `(hook, name, hook_dict, overwrite)` |  | 76 |
| `register_import_error_hook` | function | `(hook, name, overwrite)` | Args:     hook: A function or string entrypoint to invoke when the specified module is imported         and an error occurs. | 142 |
| `register_post_import_hook` | function | `(hook, name, overwrite)` | Args:     hook: A function or string entrypoint to invoke when the specified module is imported. | 157 |
| `get_post_import_hooks` | function | `(name)` |  | 171 |
| `discover_post_import_hooks` | function | `(group)` |  | 189 |
| `notify_module_loaded` | function | `(module)` |  | 206 |
| `notify_module_import_error` | function | `(module_name)` |  | 216 |
| `ImportHookFinder` | class | `` |  | 245 |
| `ImportHookFinder.find_module` | method | `(self, fullname, path)` |  | 251 |
| `ImportHookFinder.find_spec` | method | `(self, fullname, path, target)` |  | 297 |
| `when_imported` | function | `(name, error_handler)` |  | 343 |

## `mlflow/utils/jsonpath_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `split_path_respecting_backticks` | function | `(path: str) -> list[str]` | Split path on dots, but keep backticked segments intact. | 26 |
| `jsonpath_extract_values` | function | `(obj: dict[str, Any], path: str) -> list[Any]` | Extract values from nested dict using JSONPath-like dot notation with * wildcard support. | 64 |
| `filter_json_by_fields` | function | `(data: dict[str, Any], field_paths: list[str]) -> dict[str, Any]` | Filter a JSON dict to only include fields specified by the field paths. | 122 |
| `find_matching_paths` | function | `(data: dict[str, Any], wildcard_path: str) -> list[str]` | Find all actual paths in data that match a wildcard pattern. | 155 |
| `get_nested_value_safe` | function | `(data: dict[str, Any], parts: list[str]) -> Any \| None` | Safely get nested value, returning None if path doesn't exist. | 186 |
| `set_nested_value` | function | `(data: dict[str, Any], parts: list[str], value: Any) -> None` | Set a nested value in a dictionary, creating intermediate dicts/lists as needed. | 199 |
| `validate_field_paths` | function | `(field_paths: list[str], sample_data: dict[str, Any], verbose: bool) -> None` | Validate that field paths exist in the data structure. | 237 |
| `get_available_field_suggestions` | function | `(data: dict[str, Any], prefix: str) -> list[str]` | Get a list of available field paths for suggestions. | 310 |

## `mlflow/utils/lazy_load.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `LazyLoader` | class | `(types.ModuleType)` | Class for module lazy loading. | 8 |

## `mlflow/utils/logging_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_mlflow_log_level` | function | `() -> str` | Returns the log level from MLFLOW_LOGGING_LEVEL env var, defaulting to INFO. | 11 |
| `MlflowLoggingStream` | class | `` | A Python stream for use with event logging APIs throughout MLflow (`eprint()`, `logger.info()`, etc.). | 22 |
| `MlflowLoggingStream.write` | method | `(self, text)` |  | 33 |
| `MlflowLoggingStream.flush` | method | `(self)` |  | 37 |
| `MlflowLoggingStream.enabled` | method | `(self, value)` |  | 46 |
| `disable_logging` | function | `()` | Disables the `MlflowLoggingStream` used by event logging APIs throughout MLflow (`eprint()`, `logger.info()`, etc), silencing a... | 53 |
| `enable_logging` | function | `()` | Enables the `MlflowLoggingStream` used by event logging APIs throughout MLflow (`eprint()`, `logger.info()`, etc), emitting all... | 61 |
| `MlflowFormatter` | class | `(logging.Formatter)` | Custom Formatter Class to support colored log ANSI characters might not work natively on older Windows, so disabling the featur... | 70 |
| `MlflowFormatter.format` | method | `(self, record)` |  | 98 |
| `SuppressLogFilter` | class | `(logging.Filter)` |  | 114 |
| `SuppressLogFilter.filter` | method | `(self, record)` |  | 115 |
| `SensitiveQueryParamFilter` | class | `(logging.Filter)` | Logging filter that masks cloud storage credentials embedded in URL query strings. | 144 |
| `SensitiveQueryParamFilter.filter` | method | `(self, record: logging.LogRecord) -> bool` |  | 152 |
| `eprint` | function | `(*args, **kwargs)` |  | 228 |
| `LoggerMessageFilter` | class | `(logging.Filter)` |  | 232 |
| `LoggerMessageFilter.filter` | method | `(self, record)` |  | 238 |
| `suppress_logs` | function | `(module: str, filter_regex: re.Pattern)` | Context manager that suppresses log messages from the specified module that match the specified regular expression. | 245 |
| `suppress_logs_in_thread` | function | `()` | Context manager to suppress logs in the current thread. | 268 |

## `mlflow/utils/mime_type_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_text_extensions` | function | `()` |  | 10 |

## `mlflow/utils/model_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `env_var_tracker` | function | `()` | Context manager for temporarily tracking environment variables accessed. | 504 |

## `mlflow/utils/nfs_on_spark.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_nfs_cache_root_dir` | function | `()` |  | 28 |

## `mlflow/utils/os.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `is_windows` | function | `()` | Returns true if the local system/OS name is Windows. | 4 |

## `mlflow/utils/oss_registry_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_oss_host_creds` | function | `(server_uri)` | Retrieve the host credentials for the OSS server. | 12 |

## `mlflow/utils/plugins.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_entry_points` | function | `(group: str) -> list[importlib.metadata.EntryPoint]` |  | 8 |

## `mlflow/utils/process.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `ShellCommandException` | class | `(Exception)` |  | 10 |
| `ShellCommandException.from_completed_process` | classmethod | `(cls, process)` |  | 12 |
| `cache_return_value_per_process` | function | `(fn)` | A decorator which globally caches the return value of the decorated function. | 154 |

## `mlflow/utils/promptlab_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `create_eval_results_json` | function | `(prompt_parameters, model_input, model_output_parameters, model_output)` |  | 15 |

## `mlflow/utils/proto_json_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `message_to_json` | function | `(message)` | Converts a message to JSON, using snake_case for field names. | 112 |
| `proto_timestamp_to_milliseconds` | function | `(timestamp: str) -> int` | Converts a timestamp string (e.g. | 132 |
| `milliseconds_to_proto_timestamp` | function | `(milliseconds: int) -> str` | Converts milliseconds to a timestamp string (e.g. | 141 |
| `proto_duration_to_milliseconds` | function | `(duration: str) -> int` | Converts a duration string (e.g. | 150 |
| `milliseconds_to_proto_duration` | function | `(milliseconds: int) -> str` | Converts milliseconds to a duration string (e.g. | 159 |
| `parse_dict` | function | `(js_dict, message)` | Parses a JSON dictionary into a message proto, ignoring unknown fields in the JSON. | 168 |
| `set_pb_value` | function | `(proto: Value, value: Any)` | DO NOT USE THIS FUNCTION. | 173 |
| `parse_pb_value` | function | `(proto: Value) -> Any \| None` | DO NOT USE THIS FUNCTION. | 200 |
| `NumpyEncoder` | class | `(JSONEncoder)` | Special json encoder for numpy types. | 220 |
| `NumpyEncoder.try_convert` | method | `(self, o)` |  | 227 |
| `NumpyEncoder.default` | method | `(self, o)` |  | 254 |
| `MlflowInvalidInputException` | class | `(MlflowException)` |  | 262 |
| `MlflowFailedTypeConversion` | class | `(MlflowInvalidInputException)` |  | 267 |
| `cast_df_types_according_to_schema` | function | `(pdf, schema)` |  | 275 |
| `dataframe_from_parsed_json` | function | `(decoded_input, pandas_orient, schema)` | Convert parsed json into pandas.DataFrame. | 345 |
| `dataframe_from_raw_json` | function | `(path_or_str, schema, pandas_orient: str)` | Parse raw json into a pandas.Dataframe. | 408 |
| `convert_data_type` | function | `(data, spec)` | Convert input data to the type specified in the spec. | 455 |
| `parse_instances_data` | function | `(data, schema)` |  | 553 |
| `parse_inputs_data` | function | `(inputs_data_or_path, schema)` | Helper function to cast inputs_data based on the schema. | 604 |
| `parse_tf_serving_input` | function | `(inp_dict, schema)` | Args:     inp_dict: A dict deserialized from a JSON string formatted as described in TF's         serving API doc         (http... | 621 |
| `get_jsonable_input` | function | `(name, data)` |  | 677 |
| `dump_input_data` | function | `(data, inputs_key, params: dict[str, Any] \| None)` | Args:     data: Input data. | 686 |

## `mlflow/utils/provider_filter.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `normalize_provider_name` | function | `(name: str) -> str` |  | 21 |
| `is_provider_allowed` | function | `(provider_name: str) -> bool` |  | 42 |
| `filter_providers` | function | `(providers: list[str]) -> list[str]` |  | 50 |

## `mlflow/utils/providers.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `FieldDict` | class | `(TypedDict)` |  | 40 |
| `AuthModeDict` | class | `(TypedDict)` |  | 48 |
| `ResponseFieldDict` | class | `(TypedDict)` |  | 56 |
| `AuthModeResponseDict` | class | `(TypedDict)` |  | 64 |
| `ProviderConfigResponse` | class | `(TypedDict)` |  | 72 |
| `ModelInfo` | class | `(TypedDict)` | Flat model info used internally by cost_per_token, get_models, etc. | 77 |
| `ModelDict` | class | `(TypedDict)` | Model dictionary returned by get_models() and the gateway API. | 102 |
| `CatalogContextWindow` | class | `(TypedDict)` |  | 127 |
| `CatalogPricingTier` | class | `(TypedDict)` |  | 133 |
| `CatalogLongContextTier` | class | `(CatalogPricingTier)` |  | 140 |
| `CatalogPricing` | class | `(CatalogPricingTier)` |  | 144 |
| `CatalogCapabilities` | class | `(TypedDict)` |  | 149 |
| `CatalogModelEntry` | class | `(TypedDict)` |  | 157 |
| `CatalogFile` | class | `(TypedDict)` |  | 166 |
| `cost_per_token` | function | `(model: str, prompt_tokens: int, completion_tokens: int, custom_llm_provider: str \| None, cache_read_input...` | Calculate cost per token using the bundled model price data. | 334 |
| `get_provider_config_response` | function | `(provider: str) -> ProviderConfigResponse` | Get provider configuration formatted for API response. | 787 |
| `get_all_providers` | function | `() -> list[str]` | Get a list of all providers. | 864 |
| `get_models` | function | `(provider: str \| None) -> list[ModelDict]` | Get a list of models from LiteLLM, optionally filtered by provider. | 879 |

## `mlflow/utils/request_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `TCPKeepAliveHTTPAdapter` | class | `(HTTPAdapter)` | HTTPAdapter with TCP keepalive enabled to detect stale/dead connections faster. | 63 |
| `TCPKeepAliveHTTPAdapter.init_poolmanager` | method | `(self, connections, maxsize, block, **pool_kwargs)` |  | 66 |
| `TCPKeepAliveHTTPAdapter.proxy_manager_for` | method | `(self, proxy, **proxy_kwargs)` |  | 70 |
| `JitteredRetry` | class | `(Retry)` | urllib3 < 2 doesn't support `backoff_jitter`. | 75 |
| `JitteredRetry.get_backoff_time` | method | `(self)` | Source: https://github.com/urllib3/urllib3/commit/214b184923388328919b0a4b0c15bff603aa51be | 84 |
| `augmented_raise_for_status` | function | `(response)` | Wrap the standard `requests.response.raise_for_status()` method and return reason | 105 |
| `download_chunk` | function | `(*, range_start, range_end, headers, download_path, http_uri)` |  | 118 |
| `cloud_storage_http_request` | function | `(method, url, max_retries, backoff_factor, backoff_jitter, retry_codes, timeout, **kwargs)` | Performs an HTTP PUT/GET/PATCH request using Python's `requests` module with automatic retry. | 285 |

## `mlflow/utils/requirements_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `warn_dependency_requirement_mismatches` | function | `(model_requirements: list[str])` | Inspects the model's dependencies and prints a warning if the current Python environment doesn't satisfy them. | 662 |

## `mlflow/utils/rest_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `disable_429_retry` | function | `()` |  | 58 |
| `is_429_retry_disabled` | function | `() -> bool` |  | 66 |
| `http_request` | function | `(host_creds, endpoint, method, max_retries, backoff_factor, backoff_jitter, extra_headers, retry_codes, tim...` | Makes an HTTP request with the specified method to the specified hostname/endpoint. | 101 |
| `get_workspace_client` | function | `(use_secret_scope_token, host, token, databricks_auth_profile, retry_timeout_seconds, timeout)` |  | 309 |
| `http_request_safe` | function | `(host_creds, endpoint, method, **kwargs)` | Wrapper around ``http_request`` that also verifies that the request succeeds with code 200. | 344 |
| `verify_rest_response` | function | `(response, endpoint, expected_status: int)` | Verify the return code and format, raise exception if the request was not successful. | 352 |
| `validate_deployment_timeout_config` | function | `(timeout: int \| None, retry_timeout_seconds: int \| None)` | Validate that total retry timeout is not less than single request timeout. | 447 |
| `extract_api_info_for_service` | function | `(service, path_prefix)` | Return a dictionary mapping each API method to a tuple (path, HTTP method) | 558 |
| `extract_all_api_info_for_service` | function | `(service, path_prefix)` | Return a dictionary mapping each API method to a list of tuples [(path, HTTP method)] | 570 |
| `get_single_trace_endpoint` | function | `(request_id, use_v3)` | Get the endpoint for a single trace. | 582 |
| `get_single_trace_endpoint_v4` | function | `(location: str, trace_id: str) -> str` | Get the endpoint for a single trace using the V4 API. | 597 |
| `get_single_assessment_endpoint_v4` | function | `(location: str, trace_id: str, assessment_id: str) -> str` | Get the endpoint for a single assessment using the V4 API. | 604 |
| `get_logged_model_endpoint` | function | `(model_id: str) -> str` |  | 611 |
| `get_single_assessment_endpoint` | function | `(trace_id: str, assessment_id: str) -> str` | Get the endpoint for a single assessment. | 615 |
| `get_trace_tag_endpoint` | function | `(trace_id)` | Get the endpoint for trace tags. | 626 |
| `call_endpoint` | function | `(host_creds, endpoint, method, json_body, response_proto, extra_headers, retry_timeout_seconds, expected_st...` |  | 631 |
| `call_endpoints` | function | `(host_creds, endpoints, json_body, response_proto, extra_headers)` |  | 681 |
| `MlflowHostCreds` | class | `` | Provides a hostname and optional authentication for talking to an MLflow tracking server. | 694 |

## `mlflow/utils/search_logged_model_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `EntityType` | class | `(Enum)` |  | 14 |
| `EntityType.from_str` | classmethod | `(cls, s: str) -> 'EntityType'` |  | 21 |
| `Entity` | class | `` |  | 37 |
| `Entity.from_str` | classmethod | `(cls, s: str) -> 'Entity'` |  | 47 |
| `Entity.is_numeric` | method | `(self) -> bool` | Does this entity represent a numeric column? | 55 |
| `Entity.validate_op` | method | `(self, op: str) -> None` |  | 63 |
| `Comparison` | class | `` |  | 74 |
| `parse_filter_string` | function | `(filter_string: str \| None) -> list[Comparison]` |  | 80 |

## `mlflow/utils/search_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `SearchUtils` | class | `` |  | 172 |
| `SearchUtils.get_comparison_func` | staticmethod | `(comparator)` |  | 242 |
| `SearchUtils.get_sql_comparison_func` | staticmethod | `(comparator, dialect)` |  | 257 |
| `SearchUtils.translate_key_alias` | staticmethod | `(key)` |  | 328 |
| `SearchUtils.validate_list_supported` | classmethod | `(cls, key: str) -> None` |  | 421 |
| `SearchUtils.parse_search_filter` | classmethod | `(cls, filter_string)` |  | 578 |
| `SearchUtils.is_metric` | classmethod | `(cls, key_type, comparator)` |  | 601 |
| `SearchUtils.is_param` | classmethod | `(cls, key_type, comparator)` |  | 612 |
| `SearchUtils.is_tag` | classmethod | `(cls, key_type, comparator)` |  | 623 |
| `SearchUtils.is_attribute` | classmethod | `(cls, key_type, key_name, comparator)` |  | 634 |
| `SearchUtils.is_string_attribute` | classmethod | `(cls, key_type, key_name, comparator)` |  | 640 |
| `SearchUtils.is_numeric_attribute` | classmethod | `(cls, key_type, key_name, comparator)` |  | 652 |
| `SearchUtils.is_dataset` | classmethod | `(cls, key_type, comparator)` |  | 664 |
| `SearchUtils.filter` | classmethod | `(cls, runs, filter_string)` | Filters a set of runs based on a search filter string. | 763 |
| `SearchUtils.parse_order_by_for_search_runs` | classmethod | `(cls, order_by)` |  | 838 |
| `SearchUtils.parse_order_by_for_search_registered_models` | classmethod | `(cls, order_by)` |  | 844 |
| `SearchUtils.sort` | classmethod | `(cls, runs, order_by_list)` | Sorts a set of runs based on their natural ordering and an overriding set of order_bys. | 922 |
| `SearchUtils.parse_start_offset_from_page_token` | classmethod | `(cls, page_token)` |  | 942 |
| `SearchUtils.create_page_token` | classmethod | `(cls, offset)` |  | 986 |
| `SearchUtils.paginate` | classmethod | `(cls, runs, page_token, max_results)` | Paginates a set of runs based on an offset encoded into the page_token and a max results limit. | 990 |
| `SearchExperimentsUtils` | class | `(SearchUtils)` |  | 1053 |
| `SearchExperimentsUtils.parse_order_by_for_search_experiments` | classmethod | `(cls, order_by)` |  | 1144 |
| `SearchExperimentsUtils.is_attribute` | classmethod | `(cls, key_type, comparator)` |  | 1150 |
| `SearchExperimentsUtils.filter` | classmethod | `(cls, experiments, filter_string)` |  | 1190 |
| `SearchExperimentsUtils.sort` | classmethod | `(cls, experiments, order_by_list)` |  | 1241 |
| `SearchModelUtils` | class | `(SearchUtils)` |  | 1267 |
| `SearchModelUtils.filter` | classmethod | `(cls, registered_models, filter_string)` | Filters a set of registered models based on a search filter string. | 1315 |
| `SearchModelUtils.parse_order_by_for_search_registered_models` | classmethod | `(cls, order_by)` |  | 1331 |
| `SearchModelUtils.sort` | classmethod | `(cls, models, order_by_list)` |  | 1355 |
| `SearchModelVersionUtils` | class | `(SearchUtils)` |  | 1452 |
| `SearchModelVersionUtils.filter` | classmethod | `(cls, model_versions, filter_string)` | Filters a set of model versions based on a search filter string. | 1513 |
| `SearchModelVersionUtils.parse_order_by_for_search_model_versions` | classmethod | `(cls, order_by)` |  | 1526 |
| `SearchModelVersionUtils.sort` | classmethod | `(cls, model_versions, order_by_list)` |  | 1557 |
| `SearchModelVersionUtils.parse_search_filter` | classmethod | `(cls, filter_string)` |  | 1665 |
| `SearchTraceUtils` | class | `(SearchUtils)` | Utility class for searching traces. | 1688 |
| `SearchTraceUtils.filter` | classmethod | `(cls, traces, filter_string)` | Filters a set of traces based on a search filter string. | 1805 |
| `SearchTraceUtils.sort` | classmethod | `(cls, traces, order_by_list)` |  | 1862 |
| `SearchTraceUtils.parse_order_by_for_search_traces` | classmethod | `(cls, order_by)` |  | 1866 |
| `SearchTraceUtils.parse_search_filter_for_search_traces` | classmethod | `(cls, filter_string)` |  | 1873 |
| `SearchTraceUtils.is_request_metadata` | classmethod | `(cls, key_type, comparator)` |  | 1901 |
| `SearchTraceUtils.is_span` | classmethod | `(cls, key_type, key_name, comparator)` |  | 1913 |
| `SearchTraceUtils.is_assessment` | classmethod | `(cls, key_type, key_name, comparator)` |  | 1955 |
| `SearchTraceUtils.is_issue` | classmethod | `(cls, key_type, key_name, comparator)` |  | 1972 |
| `TraceMetricsFilter` | class | `` |  | 2239 |
| `SearchTraceMetricsUtils` | class | `(SearchTraceUtils)` |  | 2247 |
| `SearchTraceMetricsUtils.parse_search_filter` | classmethod | `(cls, filter_string: str) -> TraceMetricsFilter` |  | 2255 |
| `SearchEvaluationDatasetsUtils` | class | `(SearchUtils)` | Utility class for searching evaluation datasets. | 2355 |
| `SearchEvaluationDatasetsUtils.parse_order_by_for_search_evaluation_datasets` | classmethod | `(cls, order_by)` |  | 2412 |
| `SearchEvaluationDatasetsUtils.is_string_attribute` | classmethod | `(cls, type_, key, comparator)` |  | 2418 |
| `SearchEvaluationDatasetsUtils.is_numeric_attribute` | classmethod | `(cls, type_, key, comparator)` |  | 2426 |
| `SearchLoggedModelsUtils` | class | `(SearchUtils)` |  | 2434 |
| `SearchLoggedModelsUtils.validate_list_supported` | classmethod | `(cls, key: str) -> None` | Override to allow logged model attributes to be used with IN/NOT IN. | 2495 |
| `SearchLoggedModelsUtils.filter_logged_models` | classmethod | `(cls, models: list[LoggedModel], filter_string: str \| None, datasets: list[dict[str, Any]] \| None)` | Filters a set of runs based on a search filter string and list of dataset filters. | 2501 |
| `OrderBy` | class | `` |  | 2533 |
| `SearchLoggedModelsUtils.parse_order_by_for_logged_models` | classmethod | `(cls, order_by: dict[str, Any]) -> OrderBy` |  | 2540 |
| `SearchLoggedModelsUtils.sort` | classmethod | `(cls, models, order_by_list)` |  | 2627 |
| `SearchLoggedModelsPaginationToken` | class | `` |  | 2647 |
| `SearchLoggedModelsPaginationToken.to_json` | method | `(self) -> str` |  | 2653 |
| `SearchLoggedModelsPaginationToken.encode` | method | `(self) -> str` |  | 2656 |
| `SearchLoggedModelsPaginationToken.decode` | classmethod | `(cls, token: str) -> 'SearchLoggedModelsPaginationToken'` |  | 2660 |
| `SearchLoggedModelsPaginationToken.validate` | method | `(self, experiment_ids: list[str], filter_string: str \| None, order_by: list[dict[str, Any]] \| None) -> None` |  | 2673 |
| `SearchIssuesUtils` | class | `(SearchUtils)` | Utility class for parsing issue search filters. | 2698 |

## `mlflow/utils/server_cli_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `assert_server_workspace_env_unset` | function | `() -> None` | Ensure the server is not started with ``MLFLOW_WORKSPACE`` set. | 20 |
| `resolve_default_artifact_root` | function | `(serve_artifacts: bool, default_artifact_root: str, backend_store_uri: str) -> str` |  | 34 |
| `artifacts_only_config_validation` | function | `(artifacts_only: bool, backend_store_uri: str, enable_workspaces: bool) -> None` |  | 66 |

## `mlflow/utils/spark_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `is_spark_connect_mode` | function | `()` |  | 1 |
| `get_spark_dataframe_type` | function | `()` |  | 9 |

## `mlflow/utils/string_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `strip_prefix` | function | `(original: str, prefix: str) -> str` |  | 9 |
| `strip_suffix` | function | `(original: str, suffix: str) -> str` |  | 15 |
| `is_string_type` | function | `(item: Any) -> bool` |  | 21 |
| `generate_feature_name_if_not_string` | function | `(s: Any) -> str` |  | 25 |
| `truncate_str_from_middle` | function | `(s: str, max_length: int) -> str` |  | 32 |
| `mslex_quote` | function | `(s: str, for_cmd: bool) -> str` | Quote a string for use as a command line argument in DOS or Windows. | 77 |
| `quote` | function | `(s: str) -> str` |  | 131 |
| `format_table_cell_value` | function | `(field: str, cell_value: Any, values: list[Any] \| None) -> str` | Format cell values for table display with field-specific formatting. | 142 |

## `mlflow/utils/thread_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `ThreadLocalVariable` | class | `` | Class for creating a thread local variable. | 6 |
| `ThreadLocalVariable.get` | method | `(self)` | Get the thread-local variable value. | 23 |
| `ThreadLocalVariable.set` | method | `(self, value)` | Set a value for the thread-local variable. | 44 |
| `ThreadLocalVariable.get_all_thread_values` | method | `(self) -> dict[int, Any]` | Return all thread values as a dict, dict key is the thread ID. | 51 |
| `ThreadLocalVariable.reset` | method | `(self)` | Reset the thread-local variable. | 57 |

## `mlflow/utils/time.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_current_time_millis` | function | `()` | Returns the time in milliseconds since the epoch as an integer number. | 5 |
| `conv_longdate_to_str` | function | `(longdate, local_tz)` |  | 12 |
| `Timer` | class | `` | Measures elapsed time. | 22 |

## `mlflow/utils/timeout.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `MlflowTimeoutError` | class | `(Exception)` |  | 9 |
| `run_with_timeout` | function | `(seconds)` | Context manager to runs a block of code with a timeout. | 14 |

## `mlflow/utils/uri.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `is_local_uri` | function | `(uri, is_tracking_or_registry_uri)` | Returns true if the specified URI is a local file path (/foo or file:/foo). | 28 |
| `is_file_uri` | function | `(uri)` |  | 72 |
| `is_http_uri` | function | `(uri)` |  | 77 |
| `is_databricks_uri` | function | `(uri)` | Databricks URIs look like 'databricks' (default profile) or 'databricks://profile' or 'databricks://secret_scope:secret_key_pre... | 82 |
| `is_fuse_or_uc_volumes_uri` | function | `(uri)` | Validates whether a provided URI is directed to a FUSE mount point or a UC volumes mount point. | 91 |
| `is_uc_volumes_uri` | function | `(uri: str) -> bool` |  | 114 |
| `is_valid_uc_volumes_uri` | function | `(uri: str) -> bool` |  | 119 |
| `is_databricks_unity_catalog_uri` | function | `(uri)` |  | 126 |
| `is_oss_unity_catalog_uri` | function | `(uri)` |  | 131 |
| `construct_db_uri_from_profile` | function | `(profile)` |  | 136 |
| `construct_db_uc_uri_from_profile` | function | `(profile)` | Construct a databricks-uc URI from a profile. | 141 |
| `validate_db_scope_prefix_info` | function | `(scope, prefix)` |  | 159 |
| `get_db_info_from_uri` | function | `(uri)` | Get the Databricks profile specified by the tracking URI (if any), otherwise returns None. | 176 |
| `get_databricks_profile_uri_from_artifact_uri` | function | `(uri, result_scheme)` | Retrieves the netloc portion of the URI as a ``databricks://`` or `databricks-uc://` URI, if it is a proper Databricks profile ... | 203 |
| `remove_databricks_profile_info_from_artifact_uri` | function | `(artifact_uri)` | Only removes the netloc portion of the URI if it is a Databricks profile specification, e.g. | 219 |
| `add_databricks_profile_info_to_artifact_uri` | function | `(artifact_uri, databricks_profile_uri)` | Throws an exception if ``databricks_profile_uri`` is not valid. | 231 |
| `extract_db_type_from_uri` | function | `(db_uri)` | Parse the specified DB URI to extract the database type. | 256 |
| `get_uri_scheme` | function | `(uri_or_path)` |  | 277 |
| `extract_and_normalize_path` | function | `(uri)` |  | 286 |
| `append_to_uri_path` | function | `(uri, *paths)` | Appends the specified POSIX `paths` to the path component of the specified `uri`. | 292 |
| `append_to_uri_query_params` | function | `(uri, *query_params) -> str` | Appends the specified query parameters to an existing URI. | 340 |
| `is_databricks_acled_artifacts_uri` | function | `(artifact_uri)` |  | 380 |
| `is_databricks_model_registry_artifacts_uri` | function | `(artifact_uri)` |  | 386 |
| `is_valid_dbfs_uri` | function | `(uri)` |  | 392 |
| `dbfs_hdfs_uri_to_fuse_path` | function | `(dbfs_uri: str) -> str` | Converts the provided DBFS URI into a DBFS FUSE path  Args:     dbfs_uri: A DBFS URI like "dbfs:/my-directory". | 403 |
| `resolve_uri_if_local` | function | `(local_uri)` | if `local_uri` is passed in as a relative local path, this function resolves it to absolute path relative to current working di... | 430 |
| `generate_tmp_dfs_path` | function | `(dfs_tmp)` |  | 469 |
| `join_paths` | function | `(*paths) -> str` |  | 473 |
| `validate_path_is_safe` | function | `(path)` | Validates that the specified path is safe to join with a trusted prefix. | 481 |
| `validate_path_within_directory` | function | `(base_dir: str, constructed_path: str) -> str` | Validates that the constructed path (after resolving symlinks) is within the base directory. | 515 |
| `validate_query_string` | function | `(query)` |  | 552 |
| `strip_scheme` | function | `(uri: str) -> str` | Strips the scheme from the specified URI. | 572 |
| `is_models_uri` | function | `(uri: str) -> bool` |  | 587 |

## `mlflow/utils/uv_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_uv_version` | function | `() -> Version \| None` | Get the installed uv version. | 31 |
| `is_uv_available` | function | `() -> bool` | Check if uv is installed and meets the minimum version requirement. | 75 |
| `UVProjectInfo` | class | `(NamedTuple)` | Paths for a detected uv project. | 85 |
| `detect_uv_project` | function | `(directory: str \| Path \| None) -> UVProjectInfo \| None` | Detect if the given directory is a uv project. | 92 |
| `export_uv_requirements` | function | `(directory: str \| Path \| None, no_dev: bool, no_hashes: bool, frozen: bool, groups: list[str] \| None, ex...` | Export dependencies from a uv project to pip-compatible requirements. | 120 |
| `copy_uv_project_files` | function | `(dest_dir: str \| Path, source_dir: str \| Path) -> bool` | Copy uv project files to the model artifact directory. | 212 |
| `extract_index_urls_from_uv_lock` | function | `(uv_lock_path: str \| Path) -> list[str]` | Extract private index URLs from a uv.lock file. | 261 |
| `create_uv_sync_pyproject` | function | `(dest_dir: str \| Path, python_version: str, project_name: str) -> Path` | Create a minimal pyproject.toml for uv sync. | 314 |
| `setup_uv_sync_environment` | function | `(env_dir: str \| Path, model_path: str \| Path, python_version: str) -> bool` | Set up a uv project structure for environment restoration via ``uv sync --frozen``. | 357 |
| `run_uv_sync` | function | `(project_dir: str \| Path, frozen: bool, no_dev: bool, capture_output: bool) -> bool` | Run `uv sync` to install dependencies from a uv.lock file. | 408 |
| `has_uv_lock_artifact` | function | `(model_path: str \| Path) -> bool` | Check if a model has a uv.lock artifact. | 467 |

## `mlflow/utils/validation.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `invalid_value` | function | `(path, value, message)` | Compose a standardized error message for invalid parameter values. | 111 |
| `missing_value` | function | `(path)` |  | 123 |
| `not_integer_value` | function | `(path, value)` |  | 127 |
| `exceeds_maximum_length` | function | `(path, limit)` |  | 131 |
| `append_to_json_path` | function | `(currentPath, value)` |  | 135 |
| `bad_path_message` | function | `(name)` |  | 145 |
| `validate_param_and_metric_name` | function | `(name)` |  | 152 |
| `bad_character_message` | function | `()` |  | 163 |
| `path_not_unique` | function | `(name)` |  | 173 |

## `mlflow/utils/warnings_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `color_warning` | function | `(message: str, stacklevel: int, color: str, category: type[Warning])` |  | 17 |

## `mlflow/utils/workspace_context.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `get_request_workspace` | function | `() -> str \| None` | Return the active workspace for the current execution context. | 12 |
| `is_request_workspace_resolved` | function | `() -> bool` | Return whether the server resolved the request workspace. | 30 |
| `set_server_request_workspace` | function | `(workspace: str \| None) -> Token[str \| None]` | Server-only setter: bind the workspace to the request ContextVar without touching env. | 42 |
| `set_workspace` | function | `(workspace: str \| None) -> Token[str \| None]` | Client setter: binds the workspace to the current thread and persists to env so child threads inherit it. | 52 |
| `clear_server_request_workspace` | function | `() -> None` | Clear the request-scoped ContextVar (does not touch the client env). | 67 |
| `WorkspaceContext` | class | `` | Context manager that sets the client workspace (ContextVar + env) for the duration of the block. | 73 |

## `mlflow/utils/workspace_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `resolve_entity_workspace_name` | function | `(workspace: str \| None) -> str` | Determine the workspace to associate with client-facing entities. | 27 |
| `set_workspace_store_uri` | function | `(uri: str \| None) -> None` | Set the global workspace provider URI override. | 57 |
| `resolve_workspace_store_uri` | function | `(workspace_store_uri: str \| None, tracking_uri: str \| None) -> str \| None` | Resolve the workspace provider URI according to precedence rules. | 72 |
| `get_workspace_store_uri` | function | `() -> str \| None` | Get the current workspace provider URI override, if any. | 102 |

## `mlflow/utils/yaml_utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `write_yaml` | function | `(root, file_name, data, overwrite, sort_keys, ensure_yaml_extension)` | Write dictionary data in yaml format. | 21 |
| `overwrite_yaml` | function | `(root, file_name, data, ensure_yaml_extension)` | Safely overwrites a preexisting yaml file, ensuring that file contents are not deleted or corrupted if the write fails. | 54 |
| `read_yaml` | function | `(root, file_name)` | Read data from yaml file and return as dictionary Args:     root: Directory name. | 88 |
| `safe_edit_yaml` | class | `` |  | 109 |

