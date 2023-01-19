import logging
import operator
import os
from pathlib import Path
from typing import Dict, Any
from collections import namedtuple

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep
from mlflow.recipes.step import StepClass
from mlflow.recipes.steps.train import TrainStep
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.metrics import (
    _get_builtin_metrics,
    _get_custom_metrics,
    _get_primary_metric,
    _get_model_type_from_template,
    _load_custom_metrics,
    _get_extended_task,
    transform_multiclass_metric,
)
from mlflow.recipes.utils.step import get_merged_eval_metrics, validate_classification_config
from mlflow.recipes.utils.tracking import (
    get_recipe_tracking_config,
    apply_recipe_tracking_config,
    TrackingConfig,
    get_run_tags_env_vars,
)
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.fluent import _get_experiment_id, _set_experiment_primary_metric
from mlflow.utils.databricks_utils import get_databricks_run_url
from mlflow.utils.string_utils import strip_prefix

_logger = logging.getLogger(__name__)


_FEATURE_IMPORTANCE_PLOT_FILE = "feature_importance.png"


MetricValidationResult = namedtuple(
    "MetricValidationResult", ["metric", "greater_is_better", "value", "threshold", "validated"]
)


class EvaluateStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], recipe_root: str) -> None:
        super().__init__(step_config, recipe_root)
        self.tracking_config = TrackingConfig.from_dict(self.step_config)

    def _validate_and_apply_step_config(self):
        self.target_col = self.step_config.get("target_col")
        if self.target_col is None:
            raise MlflowException(
                "Missing target_col config in recipe config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.recipe = self.step_config.get("recipe")
        if self.recipe is None:
            raise MlflowException(
                "Missing recipe config in recipe config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.positive_class = self.step_config.get("positive_class")
        self.extended_task = _get_extended_task(self.recipe, self.positive_class)
        self.model_validation_status = "UNKNOWN"
        self.primary_metric = _get_primary_metric(
            self.step_config.get("primary_metric"), self.extended_task
        )
        self.user_defined_custom_metrics = {
            metric.name: metric
            for metric in _get_custom_metrics(self.step_config, self.extended_task)
        }
        self.evaluation_metrics = {
            metric.name: metric for metric in _get_builtin_metrics(self.extended_task)
        }
        self.evaluation_metrics.update(self.user_defined_custom_metrics)
        if self.primary_metric is not None and self.primary_metric not in self.evaluation_metrics:
            raise MlflowException(
                f"The primary metric '{self.primary_metric}' is a custom metric, but its"
                " corresponding custom metric configuration is missing from `recipe.yaml`.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _validate_validation_criteria(self):
        """
        Validates validation criteria don't contain undefined metrics
        """
        val_metrics = {vc["metric"] for vc in self.step_config.get("validation_criteria", [])}
        if not val_metrics:
            return
        undefined_metrics = val_metrics.difference(self.evaluation_metrics.keys())
        if undefined_metrics:
            raise MlflowException(
                f"Validation criteria contain undefined metrics: {sorted(undefined_metrics)}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _check_validation_criteria(self, metrics, validation_criteria):
        """
        return a list of `MetricValidationResult` tuple instances.
        """
        summary = []
        for val_criterion in validation_criteria:
            metric_name = val_criterion["metric"]
            metric_val = metrics.get(metric_name)
            if metric_val is None:
                raise MlflowException(
                    f"The metric {metric_name} is defined in the recipe's validation criteria"
                    " but was not returned from mlflow evaluation.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            greater_is_better = self.evaluation_metrics[metric_name].greater_is_better
            comp_func = operator.ge if greater_is_better else operator.le
            threshold = val_criterion["threshold"]
            validated = comp_func(metric_val, threshold)
            summary.append(
                MetricValidationResult(
                    metric=metric_name,
                    greater_is_better=greater_is_better,
                    value=metric_val,
                    threshold=threshold,
                    validated=validated,
                )
            )
        return summary

    def _run(self, output_directory):
        def my_warn(*args, **kwargs):
            import sys
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            stacklevel = 1 if "stacklevel" not in kwargs else kwargs["stacklevel"]
            frame = sys._getframe(stacklevel)
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            message = f"{timestamp} {filename}:{lineno}: {args[0]}\n"
            open(os.path.join(output_directory, "warning_logs.txt"), "a").write(message)

        import warnings

        original_warn = warnings.warn
        warnings.warn = my_warn
        try:
            import pandas as pd

            open(os.path.join(output_directory, "warning_logs.txt"), "w")

            self._validate_validation_criteria()

            test_df_path = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="split",
                relative_path="test.parquet",
            )
            test_df = pd.read_parquet(test_df_path)
            validate_classification_config(self.task, self.positive_class, test_df, self.target_col)

            validation_df_path = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="split",
                relative_path="validation.parquet",
            )
            validation_df = pd.read_parquet(validation_df_path)

            run_id_path = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="train",
                relative_path="run_id",
            )
            run_id = Path(run_id_path).read_text()

            model_uri = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="train",
                relative_path=TrainStep.SKLEARN_MODEL_ARTIFACT_RELATIVE_PATH,
            )

            apply_recipe_tracking_config(self.tracking_config)
            exp_id = _get_experiment_id()

            primary_metric_greater_is_better = self.evaluation_metrics[
                self.primary_metric
            ].greater_is_better

            _set_experiment_primary_metric(
                exp_id, f"test_{self.primary_metric}", primary_metric_greater_is_better
            )

            with mlflow.start_run(run_id=run_id):
                eval_metrics = {}
                for dataset_name, dataset, evaluator_config in (
                    (
                        "validation",
                        validation_df,
                        {
                            "explainability_algorithm": "kernel",
                            "explainability_nsamples": 10,
                            "metric_prefix": "val_",
                        },
                    ),
                    (
                        "test",
                        test_df,
                        {
                            "log_model_explainability": False,
                            "metric_prefix": "test_",
                        },
                    ),
                ):
                    if self.extended_task == "classification/binary":
                        evaluator_config["pos_label"] = self.positive_class
                    eval_result = mlflow.evaluate(
                        model=model_uri,
                        data=dataset,
                        targets=self.target_col,
                        model_type=_get_model_type_from_template(self.recipe),
                        evaluators="default",
                        custom_metrics=_load_custom_metrics(
                            self.recipe_root,
                            self.evaluation_metrics.values(),
                        ),
                        evaluator_config=evaluator_config,
                    )
                    eval_result.save(os.path.join(output_directory, f"eval_{dataset_name}"))
                    eval_metrics[dataset_name] = {
                        transform_multiclass_metric(
                            strip_prefix(k, evaluator_config["metric_prefix"]), self.extended_task
                        ): v
                        for k, v in eval_result.metrics.items()
                    }

                validation_results = self._validate_model(eval_metrics, output_directory)

            card = self._build_profiles_and_card(
                run_id, model_uri, eval_metrics, validation_results, output_directory
            )
            card.save_as_html(output_directory)
            self._log_step_card(run_id, self.name)
            return card
        finally:
            warnings.warn = original_warn

    def _validate_model(self, eval_metrics, output_directory):
        validation_criteria = self.step_config.get("validation_criteria")
        validation_results = None
        if validation_criteria:
            validation_results = self._check_validation_criteria(
                eval_metrics["test"], validation_criteria
            )
            self.model_validation_status = (
                "VALIDATED" if all(cr.validated for cr in validation_results) else "REJECTED"
            )
        else:
            self.model_validation_status = "UNKNOWN"
        Path(output_directory, "model_validation_status").write_text(self.model_validation_status)
        return validation_results

    def _build_profiles_and_card(
        self, run_id, model_uri, eval_metrics, validation_results, output_directory
    ):
        """
        Constructs data profiles of predictions and errors and a step card instance corresponding
        to the current evaluate step state.

        :param run_id: The ID of the MLflow Run to which to log model evaluation results.
        :param model_uri: The URI of the model being evaluated.
        :param eval_metrics: the evaluation result keyed by dataset name from `mlflow.evaluate`.
        :param validation_results: a list of `MetricValidationResult` instances
        :param output_directory: output directory used by the evaluate step.
        """
        import pandas as pd

        # Build card
        card = BaseCard(self.recipe_name, self.name)
        # Tab 0: model performance summary.
        metric_df = (
            get_merged_eval_metrics(
                eval_metrics,
                ordered_metric_names=[self.primary_metric, *self.user_defined_custom_metrics],
            )
            .reset_index()
            .rename(columns={"index": "Metric"})
        )

        def row_style(row):
            if row.Metric == self.primary_metric or row.Metric in self.user_defined_custom_metrics:
                return pd.Series("font-weight: bold", row.index)
            else:
                return pd.Series("", row.index)

        metric_table_html = BaseCard.render_table(
            metric_df.style.format({"training": "{:.6g}", "validation": "{:.6g}"}).apply(
                row_style, axis=1
            )
        )

        card.add_tab(
            "Model Performance (Test)",
            "<h3 class='section-title'>Summary Metrics</h3>"
            "<b>NOTE</b>: Use evaluation metrics over test dataset with care. "
            "Fine-tuning model over the test dataset is not advised."
            "{{ METRICS }} ",
        ).add_html("METRICS", metric_table_html)

        # Tab 1: model validation results, if exists.
        if validation_results is not None:

            def get_icon(validated):
                return (
                    # check mark button emoji
                    "\u2705"
                    if validated
                    # cross mark emoji
                    else "\u274c"
                )

            result_df = pd.DataFrame(validation_results).assign(
                validated=lambda df: df["validated"].map(get_icon)
            )

            criteria_html = BaseCard.render_table(
                result_df.style.format({"value": "{:.6g}", "threshold": "{:.6g}"})
            )
            card.add_tab("Model Validation", "{{ METRIC_VALIDATION_RESULTS }}").add_html(
                "METRIC_VALIDATION_RESULTS",
                "<h3 class='section-title'>Model Validation Results (Test Dataset)</h3> "
                + criteria_html,
            )

        # Tab 2: Classifier plots.
        if self.recipe == "classification/v1":
            classifiers_plot_tab = card.add_tab(
                "Model Performance Plots",
                "{{ CONFUSION_MATRIX }} {{CONFUSION_MATRIX_PLOT}}"
                + "{{ LIFT_CURVE }} {{LIFT_CURVE_PLOT}}"
                + "{{ PR_CURVE }} {{PR_CURVE_PLOT}}"
                + "{{ ROC_CURVE }} {{ROC_CURVE_PLOT}}",
            )
            confusion_matrix_path = os.path.join(
                output_directory,
                "eval_validation/artifacts",
                "confusion_matrix.png",
            )
            if os.path.exists(confusion_matrix_path):
                classifiers_plot_tab.add_html(
                    "CONFUSION_MATRIX",
                    '<h3 class="section-title">Confusion Matrix Plot</h3>',
                )
                classifiers_plot_tab.add_image(
                    "CONFUSION_MATRIX_PLOT", confusion_matrix_path, width=400
                )

            lift_curve_path = os.path.join(
                output_directory,
                "eval_validation/artifacts",
                "lift_curve_plot.png",
            )
            if os.path.exists(lift_curve_path):
                classifiers_plot_tab.add_html(
                    "LIFT_CURVE",
                    '<h3 class="section-title">Lift Curve Plot</h3>',
                )
                classifiers_plot_tab.add_image("LIFT_CURVE_PLOT", lift_curve_path, width=400)

            pr_curve_path = os.path.join(
                output_directory,
                "eval_validation/artifacts",
                "precision_recall_curve_plot.png",
            )
            if os.path.exists(pr_curve_path):
                classifiers_plot_tab.add_html(
                    "PR_CURVE",
                    '<h3 class="section-title">Precision Recall Curve Plot</h3>',
                )
                classifiers_plot_tab.add_image("PR_CURVE_PLOT", pr_curve_path, width=400)

            roc_curve_path = os.path.join(
                output_directory,
                "eval_validation/artifacts",
                "roc_curve_plot.png",
            )
            if os.path.exists(roc_curve_path):
                classifiers_plot_tab.add_html(
                    "ROC_CURVE",
                    '<h3 class="section-title">ROC Curve Plot</h3>',
                )
                classifiers_plot_tab.add_image("ROC_CURVE_PLOT", roc_curve_path, width=400)

        # Tab 3: SHAP plots.
        shap_plot_tab = card.add_tab(
            "Feature Importance",
            '<h3 class="section-title">Feature Importance on Validation Dataset</h3>'
            '<h3 class="section-title">SHAP Bar Plot</h3>{{SHAP_BAR_PLOT}}'
            '<h3 class="section-title">SHAP Beeswarm Plot</h3>{{SHAP_BEESWARM_PLOT}}',
        )

        shap_bar_plot_path = os.path.join(
            output_directory, "eval_validation/artifacts", "shap_feature_importance_plot.png"
        )
        shap_beeswarm_plot_path = os.path.join(
            output_directory,
            "eval_validation/artifacts",
            "shap_beeswarm_plot.png",
        )
        shap_plot_tab.add_image("SHAP_BAR_PLOT", shap_bar_plot_path, width=800)
        shap_plot_tab.add_image("SHAP_BEESWARM_PLOT", shap_beeswarm_plot_path, width=800)

        # Tab 3: Warning log outputs.
        warning_output_path = os.path.join(output_directory, "warning_logs.txt")
        if os.path.exists(warning_output_path):
            warnings_output_tab = card.add_tab("Warning Logs", "{{ STEP_WARNINGS }}")
            warnings_output_tab.add_html(
                "STEP_WARNINGS", f"<pre>{open(warning_output_path).read()}</pre>"
            )

        # Tab 4: Run summary.
        run_summary_card_tab = card.add_tab(
            "Run Summary",
            "{{ RUN_ID }} "
            + "{{ MODEL_URI }}"
            + "{{ VALIDATION_STATUS }}"
            + "{{ EXE_DURATION }}"
            + "{{ LAST_UPDATE_TIME }}",
        ).add_markdown(
            "VALIDATION_STATUS", f"**Validation status:** `{self.model_validation_status}`"
        )
        run_url = get_databricks_run_url(
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run_id,
        )
        model_uri = f"runs:/{run_id}/train/{TrainStep.MODEL_ARTIFACT_RELATIVE_PATH}"
        model_url = get_databricks_run_url(
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run_id,
            artifact_path=f"train/{TrainStep.MODEL_ARTIFACT_RELATIVE_PATH}",
        )

        if run_url is not None:
            run_summary_card_tab.add_html(
                "RUN_ID", f"<b>MLflow Run ID:</b> <a href={run_url}>{run_id}</a><br><br>"
            )
        else:
            run_summary_card_tab.add_markdown("RUN_ID", f"**MLflow Run ID:** `{run_id}`")

        if model_url is not None:
            run_summary_card_tab.add_html(
                "MODEL_URI", f"<b>MLflow Model URI:</b> <a href={model_url}>{model_uri}</a>"
            )
        else:
            run_summary_card_tab.add_markdown("MODEL_URI", f"**MLflow Model URI:** `{model_uri}`")

        return card

    @classmethod
    def from_recipe_config(cls, recipe_config, recipe_root):
        step_config = {}
        if recipe_config.get("steps", {}).get("evaluate", {}) is not None:
            step_config.update(recipe_config.get("steps", {}).get("evaluate", {}))
        step_config["target_col"] = recipe_config.get("target_col")
        if "positive_class" in recipe_config:
            step_config["positive_class"] = recipe_config.get("positive_class")
        if recipe_config.get("custom_metrics") is not None:
            step_config["custom_metrics"] = recipe_config["custom_metrics"]
        if recipe_config.get("primary_metric") is not None:
            step_config["primary_metric"] = recipe_config["primary_metric"]
        step_config["recipe"] = recipe_config.get("recipe")
        step_config.update(
            get_recipe_tracking_config(
                recipe_root_path=recipe_root,
                recipe_config=recipe_config,
            ).to_dict()
        )
        return cls(step_config, recipe_root)

    @property
    def name(self):
        return "evaluate"

    @property
    def environment(self):
        environ = get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
        environ.update(get_run_tags_env_vars(recipe_root_path=self.recipe_root))
        return environ

    def step_class(self):
        return StepClass.TRAINING
