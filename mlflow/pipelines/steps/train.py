import importlib
import logging
import os
import re
import sys
import datetime

import cloudpickle

import mlflow
from mlflow.entities import SourceType, ViewType
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.metrics import (
    BUILTIN_PIPELINE_METRICS,
    _get_primary_metric,
    _get_custom_metrics,
    _load_custom_metric_functions,
)
from mlflow.pipelines.utils.step import get_merged_eval_metrics, get_pandas_data_profile
from mlflow.pipelines.utils.tracking import (
    get_pipeline_tracking_config,
    apply_pipeline_tracking_config,
    TrackingConfig,
    get_run_tags_env_vars,
    log_code_snapshot,
)
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.databricks_utils import get_databricks_run_url
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_TYPE, MLFLOW_PIPELINE_TEMPLATE_NAME

_logger = logging.getLogger(__name__)


class TrainStep(BaseStep):

    MODEL_ARTIFACT_RELATIVE_PATH = "model"

    def __init__(self, step_config, pipeline_root, pipeline_config=None):
        super().__init__(step_config, pipeline_root)
        self.pipeline_config = pipeline_config
        self.tracking_config = TrackingConfig.from_dict(step_config)
        self.target_col = self.step_config.get("target_col")
        self.train_module_name, self.estimator_method_name = self.step_config[
            "estimator_method"
        ].rsplit(".", 1)
        self.primary_metric = _get_primary_metric(self.step_config)
        self.evaluation_metrics = {metric.name: metric for metric in BUILTIN_PIPELINE_METRICS}
        self.evaluation_metrics.update(
            {metric.name: metric for metric in _get_custom_metrics(self.step_config)}
        )
        if self.primary_metric is not None and self.primary_metric not in self.evaluation_metrics:
            raise MlflowException(
                f"The primary metric {self.primary_metric} is a custom metric, but its"
                " corresponding custom metric configuration is missing from `pipeline.yaml`.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _run(self, output_directory):
        import pandas as pd
        import shutil
        from sklearn.pipeline import make_pipeline
        from mlflow.models.signature import infer_signature

        apply_pipeline_tracking_config(self.tracking_config)

        transformed_training_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="transform",
            relative_path="transformed_training_data.parquet",
        )
        train_df = pd.read_parquet(transformed_training_data_path)
        X_train, y_train = train_df.drop(columns=[self.target_col]), train_df[self.target_col]

        transformed_validation_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="transform",
            relative_path="transformed_validation_data.parquet",
        )
        validation_df = pd.read_parquet(transformed_validation_data_path)

        raw_training_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="split",
            relative_path="train.parquet",
        )
        raw_train_df = pd.read_parquet(raw_training_data_path)
        raw_X_train = raw_train_df.drop(columns=[self.target_col])

        raw_validation_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="split",
            relative_path="validation.parquet",
        )
        raw_validation_df = pd.read_parquet(raw_validation_data_path)

        transformer_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="transform",
            relative_path="transformer.pkl",
        )

        sys.path.append(self.pipeline_root)
        estimator_fn = getattr(
            importlib.import_module(self.train_module_name), self.estimator_method_name
        )
        estimator = estimator_fn()
        mlflow.autolog(log_models=False)

        tags = {
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.PIPELINE),
            MLFLOW_PIPELINE_TEMPLATE_NAME: self.step_config["template_name"],
        }
        with mlflow.start_run(tags=tags) as run:
            estimator.fit(X_train, y_train)

            if hasattr(estimator, "best_score_"):
                mlflow.log_metric("best_cv_score", estimator.best_score_)
            if hasattr(estimator, "best_params_"):
                mlflow.log_params(estimator.best_params_)

            # TODO: log this as a pyfunc model
            code_paths = [os.path.join(self.pipeline_root, "steps")]
            estimator_schema = infer_signature(X_train, estimator.predict(X_train.copy()))
            logged_estimator = mlflow.sklearn.log_model(
                estimator,
                f"{self.name}/estimator",
                signature=estimator_schema,
                code_paths=code_paths,
            )
            # Create a pipeline consisting of the transformer+model for test data evaluation
            with open(transformer_path, "rb") as f:
                transformer = cloudpickle.load(f)
            mlflow.sklearn.log_model(transformer, "transform/transformer", code_paths=code_paths)
            model = make_pipeline(transformer, estimator)
            model_schema = infer_signature(raw_X_train, model.predict(raw_X_train.copy()))
            model_info = mlflow.sklearn.log_model(
                model, f"{self.name}/model", signature=model_schema, code_paths=code_paths
            )
            output_model_path = get_step_output_path(
                pipeline_root_path=self.pipeline_root,
                step_name=self.name,
                relative_path=TrainStep.MODEL_ARTIFACT_RELATIVE_PATH,
            )
            if os.path.exists(output_model_path) and os.path.isdir(output_model_path):
                shutil.rmtree(output_model_path)
            mlflow.sklearn.save_model(model, output_model_path)

            with open(os.path.join(output_directory, "run_id"), "w") as f:
                f.write(run.info.run_id)
            log_code_snapshot(
                self.pipeline_root, run.info.run_id, pipeline_config=self.pipeline_config
            )

            eval_metrics = {}
            for dataset_name, dataset in {
                "training": train_df,
                "validation": validation_df,
            }.items():
                eval_result = mlflow.evaluate(
                    model=logged_estimator.model_uri,
                    data=dataset,
                    targets=self.target_col,
                    model_type="regressor",
                    evaluators="default",
                    dataset_name=dataset_name,
                    custom_metrics=_load_custom_metric_functions(
                        self.pipeline_root,
                        self.evaluation_metrics.values(),
                    ),
                    evaluator_config={
                        "log_model_explainability": False,
                    },
                )
                eval_result.save(os.path.join(output_directory, f"eval_{dataset_name}"))
                eval_metrics[dataset_name] = eval_result.metrics

        target_data = raw_validation_df[self.target_col]
        prediction_result = model.predict(raw_validation_df.drop(self.target_col, axis=1))
        pred_and_error_df = pd.DataFrame(
            {
                "target": target_data,
                "prediction": prediction_result,
                "error": prediction_result - target_data,
            }
        )
        train_predictions = model.predict(raw_train_df.drop(self.target_col, axis=1))
        worst_examples_df = BaseStep._generate_worst_examples_dataframe(
            raw_train_df, train_predictions, self.target_col
        )
        leaderboard_df = None
        try:
            leaderboard_df = self._get_leaderboard_df(run, eval_metrics)
        except Exception as e:
            _logger.warning("Failed to build model leaderboard due to unexpected failure: %s", e)

        card = self._build_step_card(
            eval_metrics=eval_metrics,
            pred_and_error_df=pred_and_error_df,
            model=model,
            model_schema=model_schema,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            worst_examples_df=worst_examples_df,
            leaderboard_df=leaderboard_df,
        )
        card.save_as_html(output_directory)
        for step_name in ("ingest", "split", "transform", "train"):
            self._log_step_card(run.info.run_id, step_name)

        return card

    def _get_leaderboard_df(self, run, eval_metrics):
        import pandas as pd

        mlflow_client = MlflowClient()
        exp_id = _get_experiment_id()

        primary_metric_greater_is_better = self.evaluation_metrics[
            self.primary_metric
        ].greater_is_better
        primary_metric_order = "DESC" if primary_metric_greater_is_better else "ASC"

        search_max_results = 100
        search_result = mlflow_client.search_runs(
            experiment_ids=exp_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=search_max_results,
            order_by=[f"metrics.{self.primary_metric}_on_data_validation {primary_metric_order}"],
        )

        metric_names = self.evaluation_metrics.keys()
        metric_keys = [f"{metric_name}_on_data_validation" for metric_name in metric_names]

        leaderboard_items = []
        for old_run in search_result:
            if all(metric_key in old_run.data.metrics for metric_key in metric_keys):
                leaderboard_items.append(
                    {
                        "Run ID": old_run.info.run_id,
                        "Run Time": datetime.datetime.fromtimestamp(
                            old_run.info.start_time // 1000
                        ),
                        **{
                            metric_name: old_run.data.metrics[metric_key]
                            for metric_name, metric_key in zip(metric_names, metric_keys)
                        },
                    }
                )

        top_leaderboard_items = [
            {"Model Rank": i + 1, **t} for i, t in enumerate(leaderboard_items[:2])
        ]

        if (
            len(top_leaderboard_items) == 2
            and top_leaderboard_items[0][self.primary_metric]
            == top_leaderboard_items[1][self.primary_metric]
        ):
            # If top1 and top2 model primary metrics are equal,
            # then their rank are both 1.
            top_leaderboard_items[1]["Model Rank"] = "1"

        top_leaderboard_item_index_values = ["Best", "2nd Best"][: len(top_leaderboard_items)]

        latest_model_item = {
            "Run ID": run.info.run_id,
            "Run Time": datetime.datetime.fromtimestamp(run.info.start_time // 1000),
            **eval_metrics["validation"],
        }

        for i, leaderboard_item in enumerate(leaderboard_items):
            latest_value = latest_model_item[self.primary_metric]
            historical_value = leaderboard_item[self.primary_metric]
            if (primary_metric_greater_is_better and latest_value >= historical_value) or (
                not primary_metric_greater_is_better and latest_value <= historical_value
            ):
                latest_model_item["Model Rank"] = str(i + 1)
                break
        else:
            latest_model_item["Model Rank"] = f"> {len(leaderboard_items)}"

        # metric columns order: primary metric, custom metrics, builtin metrics.
        def sorter(m):
            if m == self.primary_metric:
                return 0, m
            elif self.evaluation_metrics[m].custom_function is not None:
                return 1, m
            else:
                return 2, m

        metric_columns = sorted(metric_names, key=sorter)

        leaderboard_df = (
            pd.DataFrame.from_records(
                [latest_model_item, *top_leaderboard_items],
                columns=["Model Rank", *metric_columns, "Run Time", "Run ID"],
            )
            .apply(
                lambda s: s.map(lambda x: "{:.6g}".format(x))  # pylint: disable=unnecessary-lambda
                if s.name in metric_names
                else s,  # pylint: disable=unnecessary-lambda
                axis=0,
            )
            .set_axis(["Latest"] + top_leaderboard_item_index_values, axis="index")
            .transpose()
        )
        return leaderboard_df

    def _build_step_card(
        self,
        eval_metrics,
        pred_and_error_df,
        model,
        model_schema,
        run_id,
        model_uri,
        worst_examples_df,
        leaderboard_df=None,
    ):
        import pandas as pd
        from sklearn.utils import estimator_html_repr
        from sklearn import set_config

        card = BaseCard(self.pipeline_name, self.name)
        # Tab 0: model performance summary.
        metric_df = (
            get_merged_eval_metrics(eval_metrics, ordered_metric_names=[self.primary_metric])
            .reset_index()
            .rename(columns={"index": "Metric"})
        )

        def row_style(row):
            if row.Metric == self.primary_metric:
                return pd.Series("font-weight: bold", row.index)
            else:
                return pd.Series("", row.index)

        metric_table_html = BaseCard.render_table(
            metric_df.style.format({"training": "{:.6g}", "validation": "{:.6g}"}).apply(
                row_style, axis=1
            )
        )

        # Tab 1: Model performance summary metrics.
        card.add_tab(
            "Model Performance Summary Metrics",
            "<h3 class='section-title'>Summary Metrics</h3>{{ METRICS }} ",
        ).add_html("METRICS", metric_table_html)

        # Tab 2: Prediction and error data profile.
        pred_and_error_df_profile = get_pandas_data_profile(
            pred_and_error_df.reset_index(drop=True),
            "Predictions and Errors (Validation Dataset)",
        )
        card.add_tab("Profile of Predictions and Errors", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", pred_and_error_df_profile
        )
        # Tab 3: Model architecture.
        set_config(display="diagram")
        model_repr = estimator_html_repr(model)
        card.add_tab("Model Architecture", "{{MODEL_ARCH}}").add_html("MODEL_ARCH", model_repr)

        # Tab 4: Inferred model (transformer + estimator) schema.
        def render_schema(inputs, title):
            from mlflow.types import ColSpec

            table = BaseCard.render_table(
                (
                    {
                        "Name": "  " + (spec.name or "-"),
                        "Type": repr(spec.type) if isinstance(spec, ColSpec) else repr(spec),
                    }
                    for spec in inputs
                )
            )
            return '<div style="margin: 5px"><h2>{title}</h2>{table}</div>'.format(
                title=title, table=table
            )

        schema_tables = [render_schema(model_schema.inputs.inputs, "Inputs")]
        if model_schema.outputs:
            schema_tables += [render_schema(model_schema.outputs.inputs, "Outputs")]

        card.add_tab("Model Schema", "{{MODEL_SCHEMA}}").add_html(
            "MODEL_SCHEMA",
            '<div style="display: flex">{tables}</div>'.format(tables="\n".join(schema_tables)),
        )

        # Tab 5: Examples with Largest Prediction Error
        (
            card.add_tab(
                "Training Examples with Largest Prediction Error", "{{ WORST_EXAMPLES_TABLE }}"
            ).add_html("WORST_EXAMPLES_TABLE", BaseCard.render_table(worst_examples_df))
        )

        # Tab 6: Leaderboard
        if leaderboard_df is not None:
            (
                card.add_tab("Leaderboard", "{{ LEADERBOARD_TABLE }}").add_html(
                    "LEADERBOARD_TABLE", BaseCard.render_table(leaderboard_df, hide_index=False)
                )
            )

        # Tab 7: Run summary.
        run_card_tab = card.add_tab(
            "Run Summary",
            "{{ RUN_ID }} " + "{{ MODEL_URI }}" + "{{ EXE_DURATION }}" + "{{ LAST_UPDATE_TIME }}",
        )
        run_url = get_databricks_run_url(
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run_id,
        )
        model_url = get_databricks_run_url(
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run_id,
            artifact_path=re.sub(r"^.*?%s" % run_id, "", model_uri),
        )

        if run_url is not None:
            run_card_tab.add_html(
                "RUN_ID", f"<b>MLflow Run ID:</b> <a href={run_url}>{run_id}</a><br><br>"
            )
        else:
            run_card_tab.add_markdown("RUN_ID", f"**MLflow Run ID:** `{run_id}`")

        if model_url is not None:
            run_card_tab.add_html(
                "MODEL_URI", f"<b>MLflow Model URI:</b> <a href={model_url}>{model_uri}</a>"
            )
        else:
            run_card_tab.add_markdown("MODEL_URI", f"**MLflow Model URI:** `{model_uri}`")

        return card

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["train"]
            step_config["metrics"] = pipeline_config.get("metrics")
            step_config["template_name"] = pipeline_config.get("template")
            step_config.update(
                get_pipeline_tracking_config(
                    pipeline_root_path=pipeline_root,
                    pipeline_config=pipeline_config,
                ).to_dict()
            )
        except KeyError:
            raise MlflowException(
                "Config for train step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        step_config["target_col"] = pipeline_config.get("target_col")
        return cls(step_config, pipeline_root, pipeline_config=pipeline_config)

    @property
    def name(self):
        return "train"

    @property
    def environment(self):
        environ = get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
        environ.update(get_run_tags_env_vars(pipeline_root_path=self.pipeline_root))
        return environ
