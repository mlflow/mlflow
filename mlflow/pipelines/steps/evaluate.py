import logging
import operator
import os
from pathlib import Path
from typing import Dict, Any
from collections import namedtuple

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.step import StepClass
from mlflow.pipelines.steps.train import TrainStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.metrics import (
    _get_builtin_metrics,
    _get_custom_metrics,
    _get_primary_metric,
    _get_model_type_from_template,
    _load_custom_metric_functions,
)
from mlflow.pipelines.utils.step import get_merged_eval_metrics
from mlflow.pipelines.utils.tracking import (
    get_pipeline_tracking_config,
    apply_pipeline_tracking_config,
    TrackingConfig,
    get_run_tags_env_vars,
)
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.fluent import _get_experiment_id, _set_experiment_primary_metric
from mlflow.utils.databricks_utils import get_databricks_run_url

_logger = logging.getLogger(__name__)


_FEATURE_IMPORTANCE_PLOT_FILE = "feature_importance.png"


MetricValidationResult = namedtuple(
    "MetricValidationResult", ["metric", "greater_is_better", "value", "threshold", "validated"]
)


class EvaluateStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str) -> None:
        super().__init__(step_config, pipeline_root)
        self.tracking_config = TrackingConfig.from_dict(self.step_config)

    def _validate_and_apply_step_config(self):
        self.target_col = self.step_config.get("target_col")
        if self.target_col is None:
            raise MlflowException(
                "Missing target_col config in pipeline config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.template = self.step_config.get("template_name")
        if self.template is None:
            raise MlflowException(
                "Missing template_name config in pipeline config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.model_validation_status = "UNKNOWN"
        self.primary_metric = _get_primary_metric(self.step_config)
        self.user_defined_custom_metrics = {
            metric.name: metric for metric in _get_custom_metrics(self.step_config)
        }
        self.evaluation_metrics = {
            metric.name: metric for metric in _get_builtin_metrics(self.template)
        }
        self.evaluation_metrics.update(self.user_defined_custom_metrics)
        if self.primary_metric is not None and self.primary_metric not in self.evaluation_metrics:
            raise MlflowException(
                f"The primary metric '{self.primary_metric}' is a custom metric, but its"
                " corresponding custom metric configuration is missing from `pipeline.yaml`.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _validate_validation_criteria(self):
        """
        Validates validation criteria don't contain undefined metrics
        """
        val_metrics = set(vc["metric"] for vc in self.step_config.get("validation_criteria", []))
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
                    f"The metric {metric_name} is defined in the pipeline's validation criteria"
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
        import pandas as pd

        self._validate_validation_criteria()

        test_df_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="split",
            relative_path="test.parquet",
        )
        test_df = pd.read_parquet(test_df_path)

        validation_df_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="split",
            relative_path="validation.parquet",
        )
        validation_df = pd.read_parquet(validation_df_path)

        run_id_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="train",
            relative_path="run_id",
        )
        run_id = Path(run_id_path).read_text()

        model_uri = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="train",
            relative_path=TrainStep.MODEL_ARTIFACT_RELATIVE_PATH,
        )

        apply_pipeline_tracking_config(self.tracking_config)
        exp_id = _get_experiment_id()

        primary_metric_greater_is_better = self.evaluation_metrics[
            self.primary_metric
        ].greater_is_better

        _set_experiment_primary_metric(
            exp_id, f"{self.primary_metric}_on_data_test", primary_metric_greater_is_better
        )

        with mlflow.start_run(run_id=run_id):
            eval_metrics = {}
            for dataset_name, dataset, evaluator_config in (
                (
                    "validation",
                    validation_df,
                    {"explainability_algorithm": "kernel", "explainability_nsamples": 10},
                ),
                ("test", test_df, {"log_model_explainability": False}),
            ):
                eval_result = mlflow.evaluate(
                    model=model_uri,
                    data=dataset,
                    targets=self.target_col,
                    model_type=_get_model_type_from_template(self.template),
                    evaluators="default",
                    dataset_name=dataset_name,
                    custom_metrics=_load_custom_metric_functions(
                        self.pipeline_root,
                        self.evaluation_metrics.values(),
                    ),
                    evaluator_config=evaluator_config,
                )
                eval_result.save(os.path.join(output_directory, f"eval_{dataset_name}"))
                eval_metrics[dataset_name] = eval_result.metrics

            validation_results = self._validate_model(eval_metrics, output_directory)

        card = self._build_profiles_and_card(
            run_id, model_uri, eval_metrics, validation_results, output_directory
        )
        card.save_as_html(output_directory)
        self._log_step_card(run_id, self.name)
        return card

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
        card = BaseCard(self.pipeline_name, self.name)
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

        # Tab 2: SHAP plots.
        shap_plot_tab = card.add_tab(
            "Feature Importance",
            '<h3 class="section-title">Feature Importance on Validation Dataset</h3>'
            '<h3 class="section-title">SHAP Bar Plot</h3>{{SHAP_BAR_PLOT}}'
            '<h3 class="section-title">SHAP Beeswarm Plot</h3>{{SHAP_BEESWARM_PLOT}}',
        )

        shap_bar_plot_path = os.path.join(
            output_directory,
            "eval_validation/artifacts",
            "shap_feature_importance_plot_on_data_validation.png",
        )
        shap_beeswarm_plot_path = os.path.join(
            output_directory,
            "eval_validation/artifacts",
            "shap_beeswarm_plot_on_data_validation.png",
        )
        shap_plot_tab.add_image("SHAP_BAR_PLOT", shap_bar_plot_path, width=800)
        shap_plot_tab.add_image("SHAP_BEESWARM_PLOT", shap_beeswarm_plot_path, width=800)

        # Tab 3: Run summary.
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
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = {}
        if pipeline_config.get("steps", {}).get("evaluate", {}) is not None:
            step_config.update(pipeline_config.get("steps", {}).get("evaluate", {}))
        step_config["target_col"] = pipeline_config.get("target_col")
        step_config["metrics"] = pipeline_config.get("metrics")
        step_config["template_name"] = pipeline_config.get("template")
        step_config.update(
            get_pipeline_tracking_config(
                pipeline_root_path=pipeline_root,
                pipeline_config=pipeline_config,
            ).to_dict()
        )
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "evaluate"

    @property
    def environment(self):
        environ = get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
        environ.update(get_run_tags_env_vars(pipeline_root_path=self.pipeline_root))
        return environ

    def step_class(self):
        return StepClass.TRAINING
