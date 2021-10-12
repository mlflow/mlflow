import os
import shutil
import tempfile
from packaging.version import Version

import numpy as np
import xgboost

import mlflow
# Pylint doesn't detect objects used in class keyword arguments (e.g., metaclass) and considers
# `ExceptionSafeAbstractClass` as 'unused-import': https://github.com/PyCQA/pylint/issues/1630
# To avoid this bug, disable 'unused-import' on this line.
from .autologging_utils import ( # pylint: disable=unused-import
    exception_safe_function,
    ExceptionSafeAbstractClass,
)


def record_eval_results(eval_results, metrics_logger):
    """
    Create a callback function that records evaluation results.
    """
    # TODO: Remove `replace("SNAPSHOT", "dev")` once the following issue is addressed:
    #       https://github.com/dmlc/xgboost/issues/6984
    if Version(xgboost.__version__.replace("SNAPSHOT", "dev")) >= Version("1.3.0"):
        # In xgboost >= 1.3.0, user-defined callbacks should inherit
        # `xgboost.callback.TrainingCallback`:
        # https://xgboost.readthedocs.io/en/latest/python/callbacks.html#defining-your-own-callback  # noqa

        class Callback(
            xgboost.callback.TrainingCallback, metaclass=ExceptionSafeAbstractClass,
        ):
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

                metrics_logger.record_metrics(evaluation_result_dict, epoch)
                eval_results.append(evaluation_result_dict)

                # Return `False` to indicate training should not stop
                return False

        return Callback()

    else:

        @exception_safe_function
        def callback(env):
            metrics_logger.record_metrics(dict(env.evaluation_result_list), env.iteration)
            eval_results.append(dict(env.evaluation_result_list))

        return callback


def log_feature_importance_plot(features, importance, importance_type):
    """
    Log feature importance plot.
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    features = np.array(features)

    # Structure the supplied `importance` values as a `num_features`-by-`num_classes` matrix
    importances_per_class_by_feature = np.array(importance)
    if importances_per_class_by_feature.ndim <= 1:
        # In this case, the supplied `importance` values are not given per class. Rather,
        # one importance value is given per feature. For consistency with the assumed
        # `num_features`-by-`num_classes` matrix structure, we coerce the importance
        # values to a `num_features`-by-1 matrix
        indices = np.argsort(importance)
        # Sort features and importance values by magnitude during transformation to a
        # `num_features`-by-`num_classes` matrix
        features = features[indices]
        importances_per_class_by_feature = np.array(
            [[importance] for importance in importances_per_class_by_feature[indices]]
        )
        # In this case, do not include class labels on the feature importance plot because
        # only one importance value has been provided per feature, rather than an
        # one importance value for each class per feature
        label_classes_on_plot = False
    else:
        importance_value_magnitudes = np.abs(importances_per_class_by_feature).sum(axis=1)
        indices = np.argsort(importance_value_magnitudes)
        features = features[indices]
        importances_per_class_by_feature = importances_per_class_by_feature[indices]
        label_classes_on_plot = True

    ret_feature_importance = importances_per_class_by_feature.tolist()
    num_classes = importances_per_class_by_feature.shape[1]
    num_features = len(features)

    # If num_features > 10, increase the figure height to prevent the plot
    # from being too dense.
    w, h = [6.4, 4.8]  # matplotlib's default figure size
    h = h + 0.1 * num_features if num_features > 10 else h
    h = h + 0.1 * num_classes if num_classes > 1 else h
    fig, ax = plt.subplots(figsize=(w, h))
    # When importance values are provided for each class per feature, we want to ensure
    # that the same color is used for all bars in the bar chart that have the same class
    colors_to_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"][:num_classes]
    color_cycler = cycler(color=colors_to_cycle)
    ax.set_prop_cycle(color_cycler)

    # The following logic operates on one feature at a time, adding a bar to the bar chart
    # for each class that reflects the importance of the feature to predictions of that
    # class
    feature_ylocs = np.arange(num_features)
    # Define offsets on the y-axis that are used to evenly space the bars for each class
    # around the y-axis position of each feature
    offsets_per_yloc = np.linspace(-0.5, 0.5, num_classes) / 2 if num_classes > 1 else [0]
    for feature_idx, (feature_yloc, importances_per_class) in enumerate(
            zip(feature_ylocs, importances_per_class_by_feature)
    ):
        for class_idx, (offset, class_importance) in enumerate(
                zip(offsets_per_yloc, importances_per_class)
        ):
            (bar,) = ax.barh(
                feature_yloc + offset,
                class_importance,
                align="center",
                # Set the bar height such that importance value bars for a particular
                # feature are spaced properly relative to each other (no overlap or gaps)
                # and relative to importance value bars for other features
                height=(0.5 / max(num_classes - 1, 1)),
            )
            if label_classes_on_plot and feature_idx == 0:
                # Only set a label the first time a bar for a particular class is plotted to
                # avoid duplicate legend entries. If we were to set a label for every bar,
                # the legend would contain `num_features` labels for each class.
                bar.set_label("Class {}".format(class_idx))

    ax.set_yticks(feature_ylocs)
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance ({})".format(importance_type))
    if label_classes_on_plot:
        ax.legend()
    fig.tight_layout()

    tmpdir = tempfile.mkdtemp()
    try:
        # pylint: disable=undefined-loop-variable
        filepath = os.path.join(tmpdir, "feature_importance_{}.png".format(importance_type))
        fig.savefig(filepath)
        mlflow.log_artifact(filepath)
    finally:
        plt.close(fig)
        shutil.rmtree(tmpdir)

    return {features[i] : ret_feature_importance[i] for i in range(num_features)}
