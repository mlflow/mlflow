from packaging.version import Version
import logging
import xgboost

# Suppress a false positive pylint error: https://github.com/PyCQA/pylint/issues/1630
# pylint: disable=unused-import
from mlflow.utils.autologging_utils import ExceptionSafeAbstractClass

_logger = logging.getLogger(__name__)


def _patch_metric_names(metric_dict):
    # XGBoost provides some metrics with "@", e.g. "ndcg@3" that are not valid MLflow metric names
    patched_metrics = {
        metric_name.replace("@", "_at_"): value for metric_name, value in metric_dict.items()
    }
    changed_keys = set(patched_metrics.keys()) - set(metric_dict.keys())
    if changed_keys:
        _logger.info(
            "Identified one or more metrics with names containing the invalid character `@`."
            " These metric names have been sanitized by replacing `@` with `_at_`, as follows: %s",
            ", ".join(changed_keys),
        )

    return patched_metrics


def autolog_callback(env, metrics_logger, eval_results):
    metric_dict = _patch_metric_names(dict(env.evaluation_result_list))
    metrics_logger.record_metrics(metric_dict, env.iteration)
    eval_results.append(metric_dict)


IS_TRAINING_CALLBACK_SUPPORTED = Version(xgboost.__version__.replace("SNAPSHOT", "dev")) >= Version(
    "1.3.0"
)

if IS_TRAINING_CALLBACK_SUPPORTED:

    class AutologCallback(
        xgboost.callback.TrainingCallback,
        metaclass=ExceptionSafeAbstractClass,
    ):
        def __init__(self, metrics_logger, eval_results):
            # pylint: disable=super-init-not-called
            self.metrics_logger = metrics_logger
            self.eval_results = eval_results

        def after_iteration(self, model, epoch, evals_log):  # pylint: disable=unused-argument
            """
            Run after each iteration. Return True when training should stop.
            """
            # `evals_log` is a nested dict (type: Dict[str, Dict[str, List[float]]])
            # that looks like this:
            # {
            #   "train": {
            #     "auc": [0.5, 0.6, 0.7, ...],
            #     ...
            #   },
            #   ...
            # }
            evaluation_result_dict = {}
            for data_name, metric_dict in evals_log.items():
                metric_dict = _patch_metric_names(metric_dict)
                for metric_name, metric_values_on_each_iter in metric_dict.items():
                    key = f"{data_name}-{metric_name}"
                    # The last element in `metric_values_on_each_iter` corresponds to
                    # the metric on the current iteration
                    evaluation_result_dict[key] = metric_values_on_each_iter[-1]

            self.metrics_logger.record_metrics(evaluation_result_dict, epoch)
            self.eval_results.append(evaluation_result_dict)

            # Return `False` to indicate training should not stop
            return False
