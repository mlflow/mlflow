from packaging.version import Version
import xgboost

# Suppress a false positive pylint error: https://github.com/PyCQA/pylint/issues/1630
# pylint: disable=unused-import
from mlflow.utils.autologging_utils import ExceptionSafeAbstractClass


def autolog_callback(env, metrics_logger, eval_results):
    metrics_logger.record_metrics(dict(env.evaluation_result_list), env.iteration)
    eval_results.append(dict(env.evaluation_result_list))


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

        def after_iteration(self, model, epoch, evals_log):
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
                for metric_name, metric_values_on_each_iter in metric_dict.items():
                    key = "{}-{}".format(data_name, metric_name)
                    # The last element in `metric_values_on_each_iter` corresponds to
                    # the meric on the current iteration
                    evaluation_result_dict[key] = metric_values_on_each_iter[-1]

            self.metrics_logger.record_metrics(evaluation_result_dict, epoch)
            self.eval_results.append(evaluation_result_dict)

            # Return `False` to indicate training should not stop
            return False
