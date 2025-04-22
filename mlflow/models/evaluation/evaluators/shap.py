import functools
import logging
from typing import Optional

import numpy as np
from packaging.version import Version
from sklearn.pipeline import Pipeline as sk_Pipeline

import mlflow
from mlflow import MlflowException
from mlflow.models.evaluation.base import EvaluationMetric, EvaluationResult, _ModelType
from mlflow.models.evaluation.default_evaluator import (
    BuiltInEvaluator,
    _extract_predict_fn,
    _extract_raw_model,
    _get_dataframe_with_renamed_columns,
)
from mlflow.models.evaluation.evaluators.classifier import (
    _is_continuous,
    _suppress_class_imbalance_errors,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel

_logger = logging.getLogger(__name__)


_SUPPORTED_SHAP_ALGORITHMS = ("exact", "permutation", "partition", "kernel")
_DEFAULT_SAMPLE_ROWS_FOR_SHAP = 2000


def _shap_predict_fn(x, predict_fn, feature_names):
    return predict_fn(_get_dataframe_with_renamed_columns(x, feature_names))


class ShapEvaluator(BuiltInEvaluator):
    """
    A built-in evaluator to get SHAP explainability insights for classifier and regressor models.

    This evaluator often run with the main evaluator for the model like ClassifierEvaluator.
    """

    name = "shap"

    @classmethod
    def can_evaluate(cls, *, model_type, evaluator_config, **kwargs):
        return model_type in (_ModelType.CLASSIFIER, _ModelType.REGRESSOR) and evaluator_config.get(
            "log_model_explainability", True
        )

    def _evaluate(
        self,
        model: Optional["mlflow.pyfunc.PyFuncModel"],
        extra_metrics: list[EvaluationMetric],
        custom_artifacts=None,
        **kwargs,
    ) -> Optional[EvaluationResult]:
        if isinstance(model, _ServedPyFuncModel):
            _logger.warning(
                "Skipping model explainability because a model server is used for environment "
                "restoration."
            )
            return

        model_loader_module, raw_model = _extract_raw_model(model)
        if model_loader_module == "mlflow.spark":
            # TODO: Shap explainer need to manipulate on each feature values,
            #  but spark model input dataframe contains Vector type feature column
            #  which shap explainer does not support.
            #  To support this, we need expand the Vector type feature column into
            #  multiple scalar feature columns and pass it to shap explainer.
            _logger.warning(
                "Logging model explainability insights is not currently supported for PySpark "
                "models."
            )
            return

        self.y_true = self.dataset.labels_data
        self.label_list = self.evaluator_config.get("label_list")
        self.pos_label = self.evaluator_config.get("pos_label")

        if not (np.issubdtype(self.y_true.dtype, np.number) or self.y_true.dtype == np.bool_):
            # Note: python bool type inherits number type but np.bool_ does not inherit np.number.
            _logger.warning(
                "Skip logging model explainability insights because it requires all label "
                "values to be numeric or boolean."
            )
            return

        algorithm = self.evaluator_config.get("explainability_algorithm", None)
        if algorithm is not None and algorithm not in _SUPPORTED_SHAP_ALGORITHMS:
            raise MlflowException(
                message=f"Specified explainer algorithm {algorithm} is unsupported. Currently only "
                f"support {','.join(_SUPPORTED_SHAP_ALGORITHMS)} algorithms.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if algorithm != "kernel":
            feature_dtypes = list(self.X.get_original().dtypes)
            for feature_dtype in feature_dtypes:
                if not np.issubdtype(feature_dtype, np.number):
                    _logger.warning(
                        "Skip logging model explainability insights because the shap explainer "
                        f"{algorithm} requires all feature values to be numeric, and each feature "
                        "column must only contain scalar values."
                    )
                    return

        try:
            import shap
            from matplotlib import pyplot
        except ImportError:
            _logger.warning(
                "SHAP or matplotlib package is not installed, so model explainability insights "
                "will not be logged."
            )
            return

        if Version(shap.__version__) < Version("0.40"):
            _logger.warning(
                "Shap package version is lower than 0.40, Skip log model explainability."
            )
            return

        sample_rows = self.evaluator_config.get(
            "explainability_nsamples", _DEFAULT_SAMPLE_ROWS_FOR_SHAP
        )

        X_df = self.X.copy_to_avoid_mutation()

        sampled_X = shap.sample(X_df, sample_rows, random_state=0)

        mode_or_mean_dict = _compute_df_mode_or_mean(X_df)
        sampled_X = sampled_X.fillna(mode_or_mean_dict)

        # shap explainer might call provided `predict_fn` with a `numpy.ndarray` type
        # argument, this might break some model inference, so convert the argument into
        # a pandas dataframe.
        # The `shap_predict_fn` calls model's predict function, we need to restore the input
        # dataframe with original column names, because some model prediction routine uses
        # the column name.

        predict_fn = _extract_predict_fn(model)
        shap_predict_fn = functools.partial(
            _shap_predict_fn, predict_fn=predict_fn, feature_names=self.dataset.feature_names
        )

        if self.label_list is None:
            # If label list is not specified, infer label list from model output.
            # We need to copy the input data as the model might mutate the input data.
            y_pred = predict_fn(X_df.copy()) if predict_fn else self.dataset.predictions_data
            self.label_list = np.unique(np.concatenate([self.y_true, y_pred]))

        try:
            if algorithm:
                if algorithm == "kernel":
                    # We need to lazily import shap, so lazily import `_PatchedKernelExplainer`
                    from mlflow.models.evaluation._shap_patch import _PatchedKernelExplainer

                    kernel_link = self.evaluator_config.get(
                        "explainability_kernel_link", "identity"
                    )
                    if kernel_link not in ["identity", "logit"]:
                        raise ValueError(
                            "explainability_kernel_link config can only be set to 'identity' or "
                            f"'logit', but got '{kernel_link}'."
                        )
                    background_X = shap.sample(X_df, sample_rows, random_state=3)
                    background_X = background_X.fillna(mode_or_mean_dict)

                    explainer = _PatchedKernelExplainer(
                        shap_predict_fn, background_X, link=kernel_link
                    )
                else:
                    explainer = shap.Explainer(
                        shap_predict_fn,
                        sampled_X,
                        feature_names=self.dataset.feature_names,
                        algorithm=algorithm,
                    )
            else:
                if (
                    raw_model
                    and not len(self.label_list) > 2
                    and not isinstance(raw_model, sk_Pipeline)
                ):
                    # For mulitnomial classifier, shap.Explainer may choose Tree/Linear explainer
                    # for raw model, this case shap plot doesn't support it well, so exclude the
                    # multinomial_classifier case here.
                    explainer = shap.Explainer(
                        raw_model, sampled_X, feature_names=self.dataset.feature_names
                    )
                else:
                    # fallback to default explainer
                    explainer = shap.Explainer(
                        shap_predict_fn, sampled_X, feature_names=self.dataset.feature_names
                    )

            _logger.info(f"Shap explainer {explainer.__class__.__name__} is used.")

            if algorithm == "kernel":
                shap_values = shap.Explanation(
                    explainer.shap_values(sampled_X), feature_names=self.dataset.feature_names
                )
            else:
                shap_values = explainer(sampled_X)
        except Exception as e:
            # Shap evaluation might fail on some edge cases, e.g., unsupported input data values
            # or unsupported model on specific shap explainer. Catch exception to prevent it
            # breaking the whole `evaluate` function.

            if not self.evaluator_config.get("ignore_exceptions", True):
                raise e

            _logger.warning(
                f"Shap evaluation failed. Reason: {e!r}. "
                "Set logging level to DEBUG to see the full traceback."
            )
            _logger.debug("", exc_info=True)
            return
        try:
            mlflow.shap.log_explainer(explainer, artifact_path="explainer")
        except Exception as e:
            # TODO: The explainer saver is buggy, if `get_underlying_model_flavor` return "unknown",
            #   then fallback to shap explainer saver, and shap explainer will call `model.save`
            #   for sklearn model, there is no `.save` method, so error will happen.
            _logger.warning(
                f"Logging explainer failed. Reason: {e!r}. "
                "Set logging level to DEBUG to see the full traceback."
            )
            _logger.debug("", exc_info=True)

        def _adjust_color_bar():
            pyplot.gcf().axes[-1].set_aspect("auto")
            pyplot.gcf().axes[-1].set_box_aspect(50)

        def _adjust_axis_tick():
            pyplot.xticks(fontsize=10)
            pyplot.yticks(fontsize=10)

        def plot_beeswarm():
            shap.plots.beeswarm(shap_values, show=False, color_bar=True)
            _adjust_color_bar()
            _adjust_axis_tick()

        with _suppress_class_imbalance_errors(ValueError, log_warning=False):
            self._log_image_artifact(
                plot_beeswarm,
                "shap_beeswarm_plot",
            )

        def plot_summary():
            shap.summary_plot(shap_values, show=False, color_bar=True)
            _adjust_color_bar()
            _adjust_axis_tick()

        with _suppress_class_imbalance_errors(TypeError, log_warning=False):
            self._log_image_artifact(
                plot_summary,
                "shap_summary_plot",
            )

        def plot_feature_importance():
            shap.plots.bar(shap_values, show=False)
            _adjust_axis_tick()

        with _suppress_class_imbalance_errors(IndexError, log_warning=False):
            self._log_image_artifact(
                plot_feature_importance,
                "shap_feature_importance_plot",
            )

        return EvaluationResult(
            metrics=self.aggregate_metrics,
            artifacts=self.artifacts,
            run_id=self.run_id,
        )


def _compute_df_mode_or_mean(df):
    """
    Compute mean (for continuous columns) and compute mode (for other columns) for the
    input dataframe, return a dict, key is column name, value is the corresponding mode or
    mean value, this function calls `_is_continuous` to determine whether the
    column is continuous column.
    """
    continuous_cols = [c for c in df.columns if _is_continuous(df[c])]
    df_cont = df[continuous_cols]
    df_non_cont = df.drop(continuous_cols, axis=1)

    means = {} if df_cont.empty else df_cont.mean().to_dict()
    modes = {} if df_non_cont.empty else df_non_cont.mode().loc[0].to_dict()
    return {**means, **modes}
