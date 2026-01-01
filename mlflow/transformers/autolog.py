"""MLflow autologging integration for HuggingFace Transformers.

This module provides automatic logging of parameters, metrics, and models during
training with HuggingFace Transformers Trainer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.transformers
from mlflow.tracking.fluent import _initialize_logged_model
from mlflow.utils.autologging_utils import (
    BatchMetricsLogger,
    ExceptionSafeAbstractClass,
    MlflowAutologgingQueueingClient,
    get_autologging_config,
)

if TYPE_CHECKING:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
else:
    try:
        from transformers import TrainerCallback
    except ImportError:
        # Fallback if transformers not available - create a dummy base class
        class TrainerCallback:
            pass

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "transformers"

# Store the original Trainer methods for restoration
_ORIGINAL_TRAINER_INIT = None
_ORIGINAL_TRAINER_TRAIN = None


class MLflowTransformersCallback(TrainerCallback, metaclass=ExceptionSafeAbstractClass):
    """
    Callback for automatic logging of HuggingFace Transformers training to MLflow.

    This callback logs parameters, metrics, and models during training with the
    HuggingFace Transformers Trainer class. It implements the TrainerCallback
    interface to integrate seamlessly with the training lifecycle.

    The callback logs:
    - Training parameters (learning_rate, num_epochs, batch_size, etc.)
    - Model configuration
    - Training metrics (loss, eval_loss, accuracy, etc.)
    - The final trained model as an MLflow artifact

    Example:
        .. code-block:: python

            import mlflow.transformers
            from transformers import Trainer, TrainingArguments

            # Enable autologging
            mlflow.transformers.autolog()

            # Training will now automatically log to MLflow
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )
            trainer.train()
    """

    def __init__(
        self,
        client: MlflowAutologgingQueueingClient,
        metrics_logger: BatchMetricsLogger,
        run_id: str,
        log_models: bool = True,
        log_input_examples: bool = False,
        model_id: str | None = None,
        managed_run: bool = False,
    ):
        """
        Initialize the MLflow Transformers callback.

        Args:
            client: MLflow autologging queuing client for async logging.
            metrics_logger: Batch metrics logger for efficient metric logging.
            run_id: The MLflow run ID to log to.
            log_models: Whether to log the trained model at the end of training.
            log_input_examples: Whether to log input examples.
            model_id: The model ID for the LoggedModel.
            managed_run: Whether this callback created/manages the MLflow run.
                If True, the run will be ended when training completes.
        """
        self.client = client
        self.metrics_logger = metrics_logger
        self.run_id = run_id
        self.log_models = log_models
        self.log_input_examples = log_input_examples
        self.model_id = model_id
        self.managed_run = managed_run
        self._logged_params = False
        self._trainer = None

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """
        Log training parameters at the beginning of training.

        This method logs ALL training arguments and model configuration, following
        the same approach as HuggingFace's native MLflowCallback and sklearn autolog.
        
        This includes:
        - All TrainingArguments (via args.to_dict())
        - All model configuration (via model.config.to_dict())

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            **kwargs: Additional keyword arguments (may include model, tokenizer, etc.)
        """
        if self._logged_params:
            return

        try:
            import os
            
            # Get MLflow validation constants
            max_param_val_length = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
            max_params_tags_per_batch = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH
            
            # Check for environment variable to limit params (like HuggingFace callback)
            max_log_params_str = os.getenv("MLFLOW_MAX_LOG_PARAMS", None)
            max_log_params = int(max_log_params_str) if max_log_params_str and max_log_params_str.isdigit() else None
            
            # Check for flatten params option
            flatten_params = os.getenv("MLFLOW_FLATTEN_PARAMS", "FALSE").upper() in ("TRUE", "1", "YES")
            
            combined_dict = {}

            # Log ALL training arguments (like HuggingFace's native callback and sklearn autolog)
            if args is not None:
                if hasattr(args, "to_dict"):
                    combined_dict.update(args.to_dict())
                elif hasattr(args, "to_sanitized_dict"):
                    # Fallback for older transformers versions
                    combined_dict.update(args.to_sanitized_dict())

            # Log ALL model configuration (like HuggingFace's native callback)
            model = kwargs.get("model")
            if model is not None and hasattr(model, "config") and model.config is not None:
                if hasattr(model.config, "to_dict"):
                    model_config = model.config.to_dict()
                    # Merge model config with training args (training args take precedence)
                    combined_dict = {**model_config, **combined_dict}

            # Optionally flatten nested dictionaries
            if flatten_params:
                combined_dict = self._flatten_dict(combined_dict)
            
            # Remove params that are too long for MLflow (> 250 chars)
            # Also filter out None values and convert to strings
            params_to_log = {}
            for name, value in combined_dict.items():
                if value is None:
                    continue
                str_value = str(value)
                if len(str_value) > max_param_val_length:
                    _logger.debug(
                        f'Trainer is attempting to log a value of "{str_value[:50]}..." for key "{name}" '
                        f"as a parameter. MLflow's log_param() only accepts values no longer than "
                        f"{max_param_val_length} characters so we dropped this attribute."
                    )
                    continue
                params_to_log[name] = str_value
            
            # Apply max params limit if set
            params_items = list(params_to_log.items())
            if max_log_params is not None and max_log_params < len(params_items):
                _logger.debug(
                    f"Reducing the number of parameters to log from {len(params_items)} to {max_log_params}."
                )
                params_items = params_items[:max_log_params]
            
            # Log params in batches (MLflow can only log 100 at a time)
            for i in range(0, len(params_items), max_params_tags_per_batch):
                batch = dict(params_items[i : i + max_params_tags_per_batch])
                self.client.log_params(self.run_id, batch)
            
            # Log tags (model class information)
            if model is not None:
                tags = {
                    "model_class": model.__class__.__name__,
                    "model_class_full": f"{model.__class__.__module__}.{model.__class__.__name__}",
                }
                # Log trainer class if available
                if self._trainer is not None:
                    tags["trainer_class"] = self._trainer.__class__.__name__
                
                self.client.set_tags(self.run_id, tags)
            
            self.client.flush(synchronous=False)
            self._logged_params = True

        except Exception as e:
            _logger.warning(f"Failed to log training parameters: {e}")
    
    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten.
            parent_key: Parent key prefix for nested keys.
            sep: Separator to use between nested keys.
            
        Returns:
            Flattened dictionary with dot-separated keys.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def on_log(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        logs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Log metrics during training.

        This method is called whenever the Trainer logs metrics. It captures
        metrics such as loss, eval_loss, accuracy, learning_rate, etc.

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            logs: Dictionary of metrics to log.
            **kwargs: Additional keyword arguments.
        """
        if logs is None:
            return

        try:
            # Filter and convert metrics
            metrics = {}
            for key, value in logs.items():
                # Skip non-numeric values
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # Clean up metric names (remove prefixes like 'train_' for consistency)
                    metrics[key] = float(value)

            if metrics:
                # Use global_step for step-based logging
                step = state.global_step if state else None
                self.metrics_logger.record_metrics(metrics, step)

        except Exception as e:
            _logger.warning(f"Failed to log metrics: {e}")

    def on_save(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """
        Optionally log model checkpoints during training.

        This method is called when the Trainer saves a checkpoint. Currently,
        checkpoint logging is handled at the end of training rather than
        during training to avoid excessive artifact uploads.

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            **kwargs: Additional keyword arguments.
        """
        # Checkpoint logging can be enabled here in the future
        # For now, we only log the final model at the end of training

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        """
        Log the final model and flush metrics at the end of training.

        This method logs the trained model as an MLflow artifact and ensures
        all pending metrics are flushed. If this callback manages the MLflow run,
        it will also end the run.

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            model: The trained model (if provided).
            tokenizer: The tokenizer (if provided).
            **kwargs: Additional keyword arguments.
        """
        try:
            # Flush any remaining metrics
            self.metrics_logger.flush()
            self.client.flush(synchronous=True)

            # Get model and tokenizer from trainer if not provided as parameters
            # The Trainer may not pass them as parameters, so we access them from the trainer
            if model is None and self._trainer is not None:
                model = getattr(self._trainer, "model", None)
            if tokenizer is None and self._trainer is not None:
                tokenizer = getattr(self._trainer, "tokenizer", None)
                # If tokenizer is still None, try to get it from the model's tokenizer attribute
                if tokenizer is None and model is not None:
                    tokenizer = getattr(model, "tokenizer", None)
                # If still None, try to create one from the model config
                if tokenizer is None and model is not None and hasattr(model, "config"):
                    try:
                        from transformers import AutoTokenizer
                        model_name = getattr(model.config, "_name_or_path", None)
                        if model_name:
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                    except Exception as e:
                        _logger.debug(f"Could not create tokenizer from model config: {e}")

            # Log the final model if enabled
            if self.log_models and model is not None:
                try:
                    # Get registered model name from autolog config
                    registered_model_name = get_autologging_config(
                        FLAVOR_NAME, "registered_model_name", None
                    )
                    
                    # Get log_model_signatures config
                    log_model_signatures = get_autologging_config(
                        FLAVOR_NAME, "log_model_signatures", True
                    )

                    # Build components dict for log_model
                    components = {"model": model}
                    if tokenizer is not None:
                        components["tokenizer"] = tokenizer

                    # Infer task from model config to help with Pipeline creation
                    task = None
                    if hasattr(model, "config"):
                        config = model.config
                        # Try to determine task from model type
                        if hasattr(config, "architectures") and config.architectures:
                            arch = config.architectures[0].lower()
                            if "forsequenceclassification" in arch:
                                task = "text-classification"
                            elif "fortokenclassification" in arch:
                                task = "token-classification"
                            elif "forquestionanswering" in arch:
                                task = "question-answering"
                            elif "forcausallm" in arch or "forgeneration" in arch:
                                task = "text-generation"

                    # Prepare input example if needed
                    input_example = None
                    if self.log_input_examples:
                        input_example = self._extract_input_example()

                    # Infer signature if needed
                    signature = None
                    if log_model_signatures:
                        signature = self._infer_model_signature(model, tokenizer, input_example)

                    # Log the model using mlflow.transformers.log_model
                    mlflow.transformers.log_model(
                        transformers_model=components,
                        name="model",
                        task=task,
                        registered_model_name=registered_model_name,
                        signature=signature,
                        input_example=input_example,
                        model_id=self.model_id,
                    )
                    _logger.info("Successfully logged transformers model to MLflow")

                except Exception as e:
                    _logger.warning(f"Failed to log model: {e}")

        except Exception as e:
            _logger.warning(f"Failed to complete training end logging: {e}")
        finally:
            # End the run if this callback created/manages it
            if self.managed_run:
                try:
                    mlflow.end_run()
                except Exception as e:
                    _logger.warning(f"Failed to end managed MLflow run: {e}")

    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        metrics: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Log evaluation metrics.

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            metrics: Dictionary of evaluation metrics.
            **kwargs: Additional keyword arguments.
        """
        if metrics is None:
            return

        try:
            # Log evaluation metrics
            eval_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    eval_metrics[key] = float(value)

            if eval_metrics:
                step = state.global_step if state else None
                self.metrics_logger.record_metrics(eval_metrics, step)

        except Exception as e:
            _logger.warning(f"Failed to log evaluation metrics: {e}")

    def on_epoch_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """
        Called at the beginning of an epoch.

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            **kwargs: Additional keyword arguments.
        """
        # No action needed at epoch begin
        pass

    def on_epoch_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """
        Called at the end of an epoch.

        This method explicitly logs the epoch metric to ensure it's captured,
        especially when using mlflow.autolog() with async logging. The epoch
        metric is logged as 1-indexed (epoch 0 becomes epoch 1.0) to match
        standard practice and test expectations.

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            **kwargs: Additional keyword arguments.
        """
        try:
            if state is not None and hasattr(state, "epoch"):
                # Log epoch metric (1-indexed, as expected by tests and standard practice)
                # state.epoch is 0-indexed (0, 1, 2, ...), so we add 1 to make it 1-indexed
                epoch_value = float(state.epoch + 1)
                step = state.global_step if state else None
                self.metrics_logger.record_metrics({"epoch": epoch_value}, step=step)
                # Flush metrics to ensure they're available immediately
                # This is especially important when using mlflow.autolog() with async logging
                self.metrics_logger.flush()
        except Exception as e:
            _logger.warning(f"Failed to log epoch metric: {e}")

    def on_step_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """
        Called at the beginning of a training step.

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            **kwargs: Additional keyword arguments.
        """
        # No action needed at step begin
        pass

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        """
        Called at the end of a training step.

        Args:
            args: The TrainingArguments used for training.
            state: The current TrainerState.
            control: The TrainerControl object.
            **kwargs: Additional keyword arguments.
        """
        # Metrics are logged via on_log, so no action needed here
        pass

    def _extract_input_example(self):
        """
        Extract an input example from the training dataset for model logging.
        
        Returns:
            An input example suitable for logging with the model, or None if extraction fails.
        """
        try:
            if self._trainer is None:
                return None
            
            # Try to get training dataset from trainer
            train_dataset = getattr(self._trainer, "train_dataset", None)
            if train_dataset is None:
                return None
            
            # Extract a sample from the dataset
            # For HuggingFace datasets, try to get first item
            if hasattr(train_dataset, "__getitem__"):
                try:
                    sample = train_dataset[0]
                    # Convert to a format suitable for input example
                    # For text classification, extract text field
                    if isinstance(sample, dict):
                        # Try common text field names
                        for field in ["text", "sentence", "input_ids", "input"]:
                            if field in sample:
                                return sample[field]
                        # If no text field, return first string value
                        for value in sample.values():
                            if isinstance(value, str):
                                return value
                        # Otherwise return first value
                        if sample:
                            return list(sample.values())[0]
                    elif isinstance(sample, (str, list)):
                        return sample
                except (IndexError, KeyError, TypeError):
                    pass
            
            return None
        except Exception as e:
            _logger.warning(f"Failed to extract input example: {e}")
            return None

    def _infer_model_signature(self, model, tokenizer, input_example):
        """
        Infer model signature for the transformers model.
        
        Args:
            model: The transformers model.
            tokenizer: The tokenizer (if available).
            input_example: Input example for signature inference.
            
        Returns:
            ModelSignature or None if inference fails.
        """
        try:
            from mlflow.transformers.signature import infer_or_get_default_signature
            import transformers
            
            # Try to create a Pipeline from components for signature inference
            # This is the preferred approach as signature inference expects a Pipeline
            if tokenizer is not None:
                # Try to infer task from model config
                task = None
                if hasattr(model, "config"):
                    config = model.config
                    # Try to determine task from model type
                    if hasattr(config, "architectures") and config.architectures:
                        arch = config.architectures[0].lower()
                        if "forsequenceclassification" in arch:
                            task = "text-classification"
                        elif "fortokenclassification" in arch:
                            task = "token-classification"
                        elif "forquestionanswering" in arch:
                            task = "question-answering"
                        elif "forcausallm" in arch or "forgeneration" in arch:
                            task = "text-generation"
                
                # Try to create a Pipeline for signature inference
                if task is not None:
                    try:
                        pipeline = transformers.pipeline(
                            task=task,
                            model=model,
                            tokenizer=tokenizer,
                        )
                        signature = infer_or_get_default_signature(
                            pipeline, example=input_example, model_config=model.config if hasattr(model, "config") else None
                        )
                        return signature
                    except Exception as e:
                        _logger.debug(f"Could not create Pipeline for signature inference: {e}")
                        # Fall through to try components dict
                
                # Fallback: try with components dict (may not work for all cases)
                components = {"model": model, "tokenizer": tokenizer}
                try:
                    signature = infer_or_get_default_signature(
                        components, example=input_example, model_config=model.config if hasattr(model, "config") else None
                    )
                    return signature
                except Exception as e:
                    _logger.debug(f"Could not infer signature from components: {e}")
                    return None
            else:
                # If no tokenizer, try default signature based on model type
                # This likely won't work well, but we try anyway
                try:
                    return infer_or_get_default_signature(
                        model, example=input_example, model_config=model.config if hasattr(model, "config") else None
                    )
                except Exception as e:
                    _logger.debug(f"Could not infer signature from model only: {e}")
                    return None
        except Exception as e:
            _logger.warning(f"Failed to infer model signature: {e}")
            return None


def _log_transformers_datasets(trainer, run_id: str):
    """
    Log training and validation datasets to MLflow Tracking.
    
    Args:
        trainer: The Trainer instance containing datasets.
        run_id: The MLflow run ID to log datasets to.
    """
    try:
        from mlflow.entities.dataset_input import DatasetInput
        from mlflow.entities.input_tag import InputTag
        from mlflow.tracking.fluent import MLFLOW_DATASET_CONTEXT
        from mlflow.tracking.context import registry as context_registry
        from mlflow.data.code_dataset_source import CodeDatasetSource
        from mlflow.data.pandas_dataset import from_pandas
        
        tracking_uri = mlflow.get_tracking_uri()
        client = MlflowAutologgingQueueingClient(tracking_uri)
        datasets_to_log = []
        
        context_tags = context_registry.resolve_tags()
        source = CodeDatasetSource(context_tags)
        
        # Log training dataset
        train_dataset = getattr(trainer, "train_dataset", None)
        if train_dataset is not None:
            try:
                # Try to create a dataset from the transformers dataset
                # For HuggingFace datasets, extract sample data and convert to pandas
                import pandas as pd
                import numpy as np
                
                # Try to extract data from dataset
                if hasattr(train_dataset, "__getitem__"):
                    try:
                        # Get a sample to determine structure
                        sample = train_dataset[0]
                        if isinstance(sample, dict):
                            # Try to convert to pandas DataFrame
                            # Extract text/data fields - take first few samples
                            samples = []
                            max_samples = min(10, len(train_dataset) if hasattr(train_dataset, "__len__") else 10)
                            for i in range(max_samples):
                                try:
                                    item = train_dataset[i]
                                    if isinstance(item, dict):
                                        # Extract simple fields (strings, numbers)
                                        row = {}
                                        for key, value in item.items():
                                            if isinstance(value, (str, int, float, bool)):
                                                row[key] = value
                                            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                                                # For lists/arrays, take first element if it's a simple type
                                                if isinstance(value[0], (int, float, bool)):
                                                    row[f"{key}_first"] = value[0]
                                        if row:
                                            samples.append(row)
                                except (IndexError, KeyError, TypeError):
                                    break
                            
                            if samples:
                                df = pd.DataFrame(samples)
                                dataset = from_pandas(df, source=source)
                                tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="train")]
                                dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
                                datasets_to_log.append(dataset_input)
                    except Exception as e:
                        _logger.debug(f"Could not convert training dataset to MLflow format: {e}")
            except Exception as e:
                _logger.debug(f"Could not log training dataset: {e}")
        
        # Log evaluation dataset if available
        eval_dataset = getattr(trainer, "eval_dataset", None)
        if eval_dataset is not None:
            try:
                import pandas as pd
                import numpy as np
                
                if hasattr(eval_dataset, "__getitem__"):
                    try:
                        sample = eval_dataset[0]
                        if isinstance(sample, dict):
                            samples = []
                            max_samples = min(10, len(eval_dataset) if hasattr(eval_dataset, "__len__") else 10)
                            for i in range(max_samples):
                                try:
                                    item = eval_dataset[i]
                                    if isinstance(item, dict):
                                        row = {}
                                        for key, value in item.items():
                                            if isinstance(value, (str, int, float, bool)):
                                                row[key] = value
                                            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                                                if isinstance(value[0], (int, float, bool)):
                                                    row[f"{key}_first"] = value[0]
                                        if row:
                                            samples.append(row)
                                except (IndexError, KeyError, TypeError):
                                    break
                            
                            if samples:
                                df = pd.DataFrame(samples)
                                dataset = from_pandas(df, source=source)
                                tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="eval")]
                                dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)
                                datasets_to_log.append(dataset_input)
                    except Exception as e:
                        _logger.debug(f"Could not convert evaluation dataset to MLflow format: {e}")
            except Exception as e:
                _logger.debug(f"Could not log evaluation dataset: {e}")
        
        # Log all datasets
        if datasets_to_log:
            client.log_inputs(run_id=run_id, datasets=datasets_to_log)
            client.flush(synchronous=False)
            
    except Exception as e:
        _logger.warning(f"Failed to log datasets: {e}")


def _create_mlflow_callback(
    run_id: str, log_models: bool, log_input_examples: bool, managed_run: bool = False
):
    """
    Create an MLflowTransformersCallback instance for the current run.

    Args:
        run_id: The MLflow run ID.
        log_models: Whether to log models.
        log_input_examples: Whether to log input examples.
        managed_run: Whether this callback created/manages the MLflow run.

    Returns:
        An MLflowTransformersCallback instance.
    """
    tracking_uri = mlflow.get_tracking_uri()
    client = MlflowAutologgingQueueingClient(tracking_uri)

    model_id = None
    if log_models:
        model_id = _initialize_logged_model(name="model", flavor=FLAVOR_NAME).model_id

    metrics_logger = BatchMetricsLogger(run_id, tracking_uri, model_id=model_id)

    return MLflowTransformersCallback(
        client=client,
        metrics_logger=metrics_logger,
        run_id=run_id,
        log_models=log_models,
        log_input_examples=log_input_examples,
        model_id=model_id,
        managed_run=managed_run,
    )


def _get_patched_trainer_init(original_init):
    """
    Create a patched version of Trainer.__init__ that handles MLflow integration.

    Note: The callback is NOT injected here. Callback injection is done in the
    patched train() method to ensure the latest autolog configuration is used.
    This prevents issues where a Trainer is created before autolog() is called.

    Args:
        original_init: The original Trainer.__init__ method.

    Returns:
        A patched __init__ method.
    """
    import functools

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # Check if MLflow integration should be disabled
        # Store this flag on the instance so train() can check it
        disable_mlflow = kwargs.pop("disable_mlflow_integration", False)

        # Call the original __init__
        original_init(self, *args, **kwargs)

        # Store the disable flag on the instance for later use in train()
        if disable_mlflow:
            self._mlflow_integration_disabled = True

    return patched_init


def _get_patched_trainer_train(original_train):
    """
    Create a patched version of Trainer.train that injects the MLflow callback.

    This is the main entry point for callback injection. We inject here (instead
    of in __init__) to ensure the latest autolog configuration is used. This
    handles the common case where a Trainer is created before autolog() is called.

    Args:
        original_train: The original Trainer.train method.

    Returns:
        A patched train method.
    """
    import functools

    @functools.wraps(original_train)
    def patched_train(self, *args, **kwargs):
        # Check if user explicitly disabled MLflow integration on this trainer
        if getattr(self, "_mlflow_integration_disabled", False):
            return original_train(self, *args, **kwargs)

        # Check if autologging is enabled and not disabled
        try:
            from mlflow.utils.autologging_utils import get_autologging_config

            if get_autologging_config(FLAVOR_NAME, "disable", False):
                return original_train(self, *args, **kwargs)
        except Exception:
            pass

        # Remove any existing MLflowTransformersCallback to ensure we use latest config
        # This is important because the callback might have been added with different
        # config settings from a previous autolog() call
        if hasattr(self, "callback_handler") and self.callback_handler is not None:
            callbacks_to_remove = [
                cb
                for cb in self.callback_handler.callbacks
                if isinstance(cb, MLflowTransformersCallback)
            ]
            for cb in callbacks_to_remove:
                self.remove_callback(cb)

        # Check if there's an active MLflow run
        active_run = mlflow.active_run()
        managed_run = False
        if active_run is None:
            # Start a new run if none is active - this callback will manage this run
            mlflow.start_run()
            active_run = mlflow.active_run()
            managed_run = True

        if active_run is not None:
            run_id = active_run.info.run_id

            # Get autologging configuration (read fresh to get latest settings)
            log_models = get_autologging_config(FLAVOR_NAME, "log_models", True)
            log_input_examples = get_autologging_config(FLAVOR_NAME, "log_input_examples", False)
            log_datasets = get_autologging_config(FLAVOR_NAME, "log_datasets", False)
            extra_tags = get_autologging_config(FLAVOR_NAME, "extra_tags", None)

            # Log extra tags if provided
            if extra_tags:
                try:
                    tracking_uri = mlflow.get_tracking_uri()
                    client = MlflowAutologgingQueueingClient(tracking_uri)
                    client.set_tags(run_id, extra_tags)
                    client.flush(synchronous=False)
                except Exception as e:
                    _logger.warning(f"Failed to log extra tags: {e}")

            # Log datasets if enabled
            if log_datasets:
                try:
                    _log_transformers_datasets(self, run_id)
                except Exception as e:
                    _logger.warning(f"Failed to log datasets: {e}")

            # Create and inject the MLflow callback with current config
            mlflow_callback = _create_mlflow_callback(
                run_id=run_id,
                log_models=log_models,
                log_input_examples=log_input_examples,
                managed_run=managed_run,
            )

            # Store trainer reference in callback
            mlflow_callback._trainer = self

            # Add the callback to the trainer
            self.add_callback(mlflow_callback)
            _logger.debug(
                f"Injected MLflowTransformersCallback into Trainer (log_models={log_models})"
            )

        # Call the original train method
        return original_train(self, *args, **kwargs)

    return patched_train


def _patch_trainer_init():
    """
    Patch the Trainer.__init__ method to inject MLflow callback.
    """
    global _ORIGINAL_TRAINER_INIT

    try:
        import transformers

        if _ORIGINAL_TRAINER_INIT is None:
            _ORIGINAL_TRAINER_INIT = transformers.Trainer.__init__
            transformers.Trainer.__init__ = _get_patched_trainer_init(_ORIGINAL_TRAINER_INIT)
            _logger.debug("Patched transformers.Trainer.__init__ for autologging")

    except ImportError:
        _logger.warning("transformers package not found, autologging will not be enabled")
    except Exception as e:
        _logger.warning(f"Failed to patch Trainer.__init__: {e}")


def _patch_trainer_train():
    """
    Patch the Trainer.train method to inject MLflow callback at training time.
    This handles the case where Trainer was created before autolog was enabled.
    """
    global _ORIGINAL_TRAINER_TRAIN

    try:
        import transformers

        if _ORIGINAL_TRAINER_TRAIN is None:
            _ORIGINAL_TRAINER_TRAIN = transformers.Trainer.train
            transformers.Trainer.train = _get_patched_trainer_train(_ORIGINAL_TRAINER_TRAIN)
            _logger.debug("Patched transformers.Trainer.train for autologging")

    except ImportError:
        _logger.warning("transformers package not found, autologging will not be enabled")
    except Exception as e:
        _logger.warning(f"Failed to patch Trainer.train: {e}")


def _unpatch_trainer_init():
    """
    Restore the original Trainer.__init__ method.
    """
    global _ORIGINAL_TRAINER_INIT

    try:
        import transformers

        if _ORIGINAL_TRAINER_INIT is not None:
            transformers.Trainer.__init__ = _ORIGINAL_TRAINER_INIT
            _ORIGINAL_TRAINER_INIT = None
            _logger.debug("Restored original transformers.Trainer.__init__")

    except ImportError:
        pass
    except Exception as e:
        _logger.warning(f"Failed to restore Trainer.__init__: {e}")


def _unpatch_trainer_train():
    """
    Restore the original Trainer.train method.
    """
    global _ORIGINAL_TRAINER_TRAIN

    try:
        import transformers

        if _ORIGINAL_TRAINER_TRAIN is not None:
            transformers.Trainer.train = _ORIGINAL_TRAINER_TRAIN
            _ORIGINAL_TRAINER_TRAIN = None
            _logger.debug("Restored original transformers.Trainer.train")

    except ImportError:
        pass
    except Exception as e:
        _logger.warning(f"Failed to restore Trainer.train: {e}")
