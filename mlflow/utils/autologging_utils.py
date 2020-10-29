import inspect
import functools
import warnings
import time
import contextlib
import concurrent

import mlflow
from mlflow.utils import gorilla
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient



INPUT_EXAMPLE_SAMPLE_ROWS = 5
ENSURE_AUTOLOGGING_ENABLED_TEXT = (
    "please ensure that autologging is enabled before constructing the dataset."
)


def try_mlflow_log(fn, *args, **kwargs):
    """
    Catch exceptions and log a warning to avoid autolog throwing.
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn("Logging to MLflow failed: " + str(e), stacklevel=2)


def get_unspecified_default_args(user_args, user_kwargs, all_param_names, all_default_values):
    """
    Determine which default values are used in a call, given args and kwargs that are passed in.

    :param user_args: list of arguments passed in by the user
    :param user_kwargs: dictionary of kwargs passed in by the user
    :param all_param_names: names of all of the parameters of the function
    :param all_default_values: values of all default parameters
    :return: a dictionary mapping arguments not specified by the user -> default value
    """
    num_args_without_default_value = len(all_param_names) - len(all_default_values)

    # all_default_values correspond to the last len(all_default_values) elements of the arguments
    default_param_names = all_param_names[num_args_without_default_value:]

    default_args = dict(zip(default_param_names, all_default_values))

    # The set of keyword arguments that should not be logged with default values
    user_specified_arg_names = set(user_kwargs.keys())

    num_user_args = len(user_args)

    # This checks if the user passed values for arguments with default values
    if num_user_args > num_args_without_default_value:
        num_default_args_passed_as_positional = num_user_args - num_args_without_default_value
        # Adding the set of positional arguments that should not be logged with default values
        names_to_exclude = default_param_names[:num_default_args_passed_as_positional]
        user_specified_arg_names.update(names_to_exclude)

    return {
        name: value for name, value in default_args.items() if name not in user_specified_arg_names
    }


def log_fn_args_as_params(fn, args, kwargs, unlogged=[]):  # pylint: disable=W0102
    """
    Log parameters explicitly passed to a function.
    :param fn: function whose parameters are to be logged
    :param args: arguments explicitly passed into fn
    :param kwargs: kwargs explicitly passed into fn
    :param unlogged: parameters not to be logged
    :return: None
    """
    # all_default_values has length n, corresponding to values of the
    # last n elements in all_param_names
    pos_params, _, _, pos_defaults, kw_params, kw_defaults, _ = inspect.getfullargspec(fn)

    kw_params = list(kw_params) if kw_params else []
    pos_defaults = list(pos_defaults) if pos_defaults else []
    all_param_names = pos_params + kw_params
    all_default_values = pos_defaults + [kw_defaults[param] for param in kw_params]

    # Checking if default values are present for logging. Known bug that getargspec will return an
    # empty argspec for certain functions, despite the functions having an argspec.
    if all_default_values is not None and len(all_default_values) > 0:
        # Logging the default arguments not passed by the user
        defaults = get_unspecified_default_args(args, kwargs, all_param_names, all_default_values)

        for name in [name for name in defaults.keys() if name in unlogged]:
            del defaults[name]
        try_mlflow_log(mlflow.log_params, defaults)

    # Logging the arguments passed by the user
    args_dict = dict(
        (param_name, param_val)
        for param_name, param_val in zip(all_param_names, args)
        if param_name not in unlogged
    )

    if args_dict:
        try_mlflow_log(mlflow.log_params, args_dict)

    # Logging the kwargs passed by the user
    for param_name in kwargs:
        if param_name not in unlogged:
            try_mlflow_log(mlflow.log_param, param_name, kwargs[param_name])


def wrap_patch(destination, name, patch, settings=None):
    """
    Apply a patch while preserving the attributes (e.g. __doc__) of an original function.

    :param destination: Patch destination
    :param name: Name of the attribute at the destination
    :param patch: Patch function
    :param settings: Settings for gorilla.Patch
    """
    if settings is None:
        settings = gorilla.Settings(allow_hit=True, store_hit=True)

    original = getattr(destination, name)
    wrapped = functools.wraps(original)(patch)
    patch = gorilla.Patch(destination, name, wrapped, settings=settings)
    gorilla.apply(patch)


class _InputExampleInfo:
    """
    Stores info about the input example collection before it is needed.

    For example, in xgboost and lightgbm, an InputExampleInfo object is attached to the dataset,
    where its value is read later by the train method.

    Exactly one of input_example or error_msg should be populated.
    """

    def __init__(self, input_example=None, error_msg=None):
        self.input_example = input_example
        self.error_msg = error_msg


def resolve_input_example_and_signature(
    get_input_example, infer_model_signature, log_input_example, log_model_signature, logger
):
    """
    Handles the logic of calling functions to gather the input example and infer the model
    signature.

    :param get_input_example: function which returns an input example, usually sliced from a
                              dataset. This function can raise an exception, its message will be
                              shown to the user in a warning in the logs.
    :param infer_model_signature: function which takes an input example and returns the signature
                                  of the inputs and outputs of the model. This function can raise
                                  an exception, its message will be shown to the user in a warning
                                  in the logs.
    :param log_input_example: whether to log errors while collecting the input example, and if it
                              succeeds, whether to return the input example to the user. We collect
                              it even if this parameter is False because it is needed for inferring
                              the model signature.
    :param log_model_signature: whether to infer and return the model signature.
    :param logger: the logger instance used to log warnings to the user during input example
                   collection and model signature inference.

    :return: A tuple of input_example and signature. Either or both could be None based on the
             values of log_input_example and log_model_signature.
    """

    input_example = None
    input_example_user_msg = None
    input_example_failure_msg = None
    if log_input_example or log_model_signature:
        try:
            input_example = get_input_example()
        except Exception as e:  # pylint: disable=broad-except
            input_example_failure_msg = str(e)
            input_example_user_msg = "Failed to gather input example: " + str(e)

    model_signature = None
    model_signature_user_msg = None
    if log_model_signature:
        try:
            if input_example is None:
                raise Exception(
                    "could not sample data to infer model signature: " + input_example_failure_msg
                )
            model_signature = infer_model_signature(input_example)
        except Exception as e:  # pylint: disable=broad-except
            model_signature_user_msg = "Failed to infer model signature: " + str(e)

    if log_input_example and input_example_user_msg is not None:
        logger.warning(input_example_user_msg)
    if log_model_signature and model_signature_user_msg is not None:
        logger.warning(model_signature_user_msg)

    return input_example if log_input_example else None, model_signature


def _timed_log_batch(batch_metrics_handler, run_id, metrics):
    start = time.time()
    MlflowClient().log_batch(run_id=run_id, metrics=metrics)
    end = time.time()
    return batch_metrics_handler, end - start + 0.03


def _update_avg_time_callback(future):
    batch_metrics_handler, log_batch_time = future.result()

    batch_metrics_handler.total_log_batch_time += log_batch_time
    batch_metrics_handler.num_log_batch += 1


class BatchMetricsHandler: # BatchMetricsLogger maybe?
    BATCH_LOG_INTERVAL = 23

    def __init__(self):
        # data is an array of tuples of the form (timestamp, metrics at timestamp)
        self.data = []
        self.current_step = 0
        self.accumulated_training_time = 0
        self.total_log_batch_time = 0
        self.num_log_batch = 0
        self.previous_training_timestamp = None
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _purge(self):
        print("purging")

        run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        final_metrics = []

        for idx, entry in enumerate(self.data):
            step = self.current_step - len(self.data) + idx + 1
            timestamp = entry[0]
            metrics_at_timestamp = entry[1]

            for key, value in metrics_at_timestamp.items():
                final_metrics.append(Metric(key, value, timestamp, step))

        future = self._thread_pool.submit(
            _timed_log_batch,
            batch_metrics_handler=self,
            run_id=run_id,
            metrics=final_metrics,
        )
        future.add_done_callback(_update_avg_time_callback)

        self.data = []

    def _should_purge(self, current_timestamp):
        if self.previous_training_timestamp is None:
            self.previous_training_timestamp = current_timestamp
            return False
        
        training_time = current_timestamp - self.previous_training_timestamp

        self.previous_training_timestamp = current_timestamp

        self.accumulated_training_time += training_time

        if self.num_log_batch == 0:
            return True

        if self.accumulated_training_time >= self.total_log_batch_time / self.num_log_batch:
            self.accumulated_training_time = 0
            return True

        return False

    # metrics is a dict representing the set of metrics collected during one iteration
    def record_metrics(self, metrics):
        current_timestamp = time.time()

        

        self.data.append([int(time.time() * 1000), metrics])
        # print(self.current_step)

        # print("accumulated_training_time " + str(self.accumulated_training_time))
        # print("total_log_batch_time " + str(self.total_log_batch_time))
        # print("num_log_batch " + str(self.num_log_batch))
        # print("previous_training_timestamp " + str(self.previous_training_timestamp))

        if self._should_purge(current_timestamp):
            self._purge()

        self.current_step += 1

@contextlib.contextmanager
def with_batch_metrics_handler():
    batch_metrics_handler = BatchMetricsHandler()
    yield batch_metrics_handler
    batch_metrics_handler._purge()

    # blocks until all submitted tasks are complete
    batch_metrics_handler._thread_pool.shutdown(wait=True)