import importlib
import logging
import os
import re
import sys
import datetime
import yaml

import cloudpickle

import mlflow
from mlflow.entities import SourceType, ViewType
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE, BAD_REQUEST
from mlflow.pipelines.artifacts import ModelArtifact, RunArtifact, HyperParametersArtifact
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.step import StepClass
from mlflow.pipelines.utils.execution import (
    get_step_output_path,
    _MLFLOW_PIPELINES_EXECUTION_TARGET_STEP_NAME_ENV_VAR,
)
from mlflow.pipelines.utils.metrics import (
    _get_error_fn,
    _get_builtin_metrics,
    _get_primary_metric,
    _get_custom_metrics,
    _get_model_type_from_template,
    _load_custom_metric_functions,
)
from mlflow.pipelines.utils.step import (
    get_merged_eval_metrics,
    get_pandas_data_profiles,
)
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
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_TYPE,
    MLFLOW_PIPELINE_TEMPLATE_NAME,
    MLFLOW_PIPELINE_PROFILE_NAME,
    MLFLOW_PIPELINE_STEP_NAME,
)

_logger = logging.getLogger(__name__)


class TrainStep(BaseStep):
    MODEL_ARTIFACT_RELATIVE_PATH = "model"

    def __init__(self, step_config, pipeline_root, pipeline_config=None):
        super().__init__(step_config, pipeline_root)
        self.tracking_config = TrackingConfig.from_dict(self.step_config)
        self.pipeline_config = pipeline_config

    def _validate_and_apply_step_config(self):
        self.task = self.step_config.get("template_name", "regression/v1").rsplit("/", 1)[0]
        if "using" in self.step_config:
            if self.step_config["using"] not in ["estimator_spec", "automl/flaml"]:
                raise MlflowException(
                    f"Invalid train step configuration value {self.step_config['using']} for "
                    f"key 'using'. Supported values are: ['estimator_spec']",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            self.step_config["using"] = "estimator_spec"

        if "tuning" in self.step_config:
            if "enabled" in self.step_config["tuning"] and isinstance(
                self.step_config["tuning"]["enabled"], bool
            ):
                self.step_config["tuning_enabled"] = self.step_config["tuning"]["enabled"]
            else:
                raise MlflowException(
                    "The 'tuning' configuration in the train step must include an "
                    "'enabled' key whose value is either true or false.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if self.step_config["tuning_enabled"]:
                if "sample_fraction" in self.step_config["tuning"]:
                    sample_fraction = float(self.step_config["tuning"]["sample_fraction"])
                    if sample_fraction > 0 and sample_fraction <= 1.0:
                        self.step_config["sample_fraction"] = sample_fraction
                    else:
                        raise MlflowException(
                            "The tuning 'sample_fraction' configuration in the train step "
                            "must be between 0 and 1.",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                else:
                    self.step_config["sample_fraction"] = 1.0

                if "algorithm" not in self.step_config["tuning"]:
                    self.step_config["tuning"]["algorithm"] = "hyperopt.rand.suggest"

                if "parallelism" not in self.step_config["tuning"]:
                    self.step_config["tuning"]["parallelism"] = 1

                if "max_trials" not in self.step_config["tuning"]:
                    raise MlflowException(
                        "The 'max_trials' configuration in the train step must be provided.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

                if "parameters" not in self.step_config["tuning"]:
                    raise MlflowException(
                        "The 'parameters' configuration in the train step must be provided "
                        " when tuning is enabled.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

        else:
            self.step_config["tuning_enabled"] = False

        if "estimator_params" not in self.step_config:
            self.step_config["estimator_params"] = {}

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
        self.skip_data_profiling = self.step_config.get("skip_data_profiling", False)
        if (
            "estimator_method" not in self.step_config
            and self.step_config["using"] == "estimator_spec"
        ):
            raise MlflowException(
                "Missing 'estimator_method' configuration in the train step, "
                "which is using 'estimator_spec'.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        self.primary_metric = _get_primary_metric(self.step_config)
        self.user_defined_custom_metrics = {
            metric.name: metric for metric in _get_custom_metrics(self.step_config)
        }
        self.evaluation_metrics = {
            metric.name: metric for metric in _get_builtin_metrics(self.template)
        }
        self.evaluation_metrics.update(self.user_defined_custom_metrics)
        self.evaluation_metrics_greater_is_better = {
            metric.name: metric.greater_is_better for metric in _get_builtin_metrics(self.template)
        }
        self.evaluation_metrics_greater_is_better.update(
            {
                metric.name: metric.greater_is_better
                for metric in _get_custom_metrics(self.step_config)
            }
        )
        if self.primary_metric is not None and self.primary_metric not in self.evaluation_metrics:
            raise MlflowException(
                f"The primary metric '{self.primary_metric}' is a custom metric, but its"
                " corresponding custom metric configuration is missing from `pipeline.yaml`.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.code_paths = [os.path.join(self.pipeline_root, "steps")]

    @classmethod
    def construct_search_space_from_yaml(cls, params):
        from hyperopt import hp

        search_space = {}
        for param_name, param_details in params.items():
            if "values" in param_details:
                param_details_to_pass = param_details["values"]
                search_space[param_name] = hp.choice(param_name, param_details_to_pass)
            elif "distribution" in param_details:
                hp_tuning_fn = getattr(hp, param_details["distribution"])
                param_details_to_pass = param_details.copy()
                param_details_to_pass.pop("distribution")
                search_space[param_name] = hp_tuning_fn(param_name, **param_details_to_pass)
            else:
                raise MlflowException(
                    f"Parameter {param_name} must contain either a list of 'values' or a "
                    f"'distribution' following hyperopt parameter expressions",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        return search_space

    @classmethod
    def is_tuning_param_equal(cls, tuning_param, logged_param):
        if isinstance(tuning_param, bool):
            return tuning_param == bool(logged_param)
        elif isinstance(tuning_param, int):
            return tuning_param == int(logged_param)
        elif isinstance(tuning_param, float):
            return tuning_param == float(logged_param)
        elif isinstance(tuning_param, str):
            return tuning_param.strip() == logged_param.strip()
        else:
            return tuning_param == logged_param

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

        tags = {
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.PIPELINE),
            MLFLOW_PIPELINE_TEMPLATE_NAME: self.step_config["template_name"],
            MLFLOW_PIPELINE_PROFILE_NAME: self.step_config["profile"],
            MLFLOW_PIPELINE_STEP_NAME: os.getenv(
                _MLFLOW_PIPELINES_EXECUTION_TARGET_STEP_NAME_ENV_VAR
            ),
        }

        best_estimator_params = None
        mlflow.autolog(log_models=False, silent=True)
        with mlflow.start_run(tags=tags) as run:

            estimator = self._resolve_estimator(
                X_train, y_train, validation_df, run, output_directory
            )
            estimator.fit(X_train, y_train)

            logged_estimator = self._log_estimator_to_mlflow(estimator, X_train)

            # Create a pipeline consisting of the transformer+model for test data evaluation
            with open(transformer_path, "rb") as f:
                transformer = cloudpickle.load(f)
            mlflow.sklearn.log_model(
                transformer, "transform/transformer", code_paths=self.code_paths
            )
            model = make_pipeline(transformer, estimator)
            model_schema = infer_signature(raw_X_train, model.predict(raw_X_train.copy()))
            model_info = mlflow.sklearn.log_model(
                model, f"{self.name}/model", signature=model_schema, code_paths=self.code_paths
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
                    model_type=_get_model_type_from_template(self.template),
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
        error_fn = _get_error_fn(self.template)
        pred_and_error_df = pd.DataFrame(
            {
                "target": target_data,
                "prediction": prediction_result,
                "error": error_fn(prediction_result, target_data.to_numpy()),
            }
        )
        train_predictions = model.predict(raw_train_df.drop(self.target_col, axis=1))
        worst_examples_df = BaseStep._generate_worst_examples_dataframe(
            raw_train_df,
            train_predictions,
            error_fn(train_predictions, raw_train_df[self.target_col].to_numpy()),
            self.target_col,
        )
        leaderboard_df = None
        try:
            leaderboard_df = self._get_leaderboard_df(run, eval_metrics)
        except Exception as e:
            _logger.warning("Failed to build model leaderboard due to unexpected failure: %s", e)
        tuning_df = None
        if self.step_config["tuning_enabled"]:
            try:
                tuning_df = self._get_tuning_df(run, params=best_estimator_params.keys())
            except Exception as e:
                _logger.warning(
                    "Failed to build tuning results table due to unexpected failure: %s", e
                )

        card = self._build_step_card(
            eval_metrics=eval_metrics,
            pred_and_error_df=pred_and_error_df,
            model=model,
            model_schema=model_schema,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            worst_examples_df=worst_examples_df,
            train_df=raw_train_df,
            output_directory=output_directory,
            leaderboard_df=leaderboard_df,
            tuning_df=tuning_df,
        )
        card.save_as_html(output_directory)
        for step_name in ("ingest", "split", "transform", "train"):
            self._log_step_card(run.info.run_id, step_name)

        return card

    def _get_user_defined_estimator(self, X_train, y_train, validation_df, run, output_directory):
        train_module_name, estimator_method_name = self.step_config["estimator_method"].rsplit(
            ".", 1
        )
        sys.path.append(self.pipeline_root)
        estimator_fn = getattr(importlib.import_module(train_module_name), estimator_method_name)
        estimator_hardcoded_params = self.step_config["estimator_params"]
        if self.step_config["tuning_enabled"]:
            estimator_hardcoded_params, best_hp_params = self._tune_and_get_best_estimator_params(
                run.info.run_id,
                estimator_hardcoded_params,
                estimator_fn,
                X_train,
                y_train,
                validation_df,
            )
            best_combined_params = dict(estimator_hardcoded_params, **best_hp_params)
            estimator = estimator_fn(best_combined_params)
            all_estimator_params = estimator.get_params()
            default_params = dict(
                set(all_estimator_params.items()) - set(best_combined_params.items())
            )
            self._write_best_parameters_outputs(
                output_directory,
                best_hp_params=best_hp_params,
                best_hardcoded_params=estimator_hardcoded_params,
                default_params=default_params,
            )
        elif len(estimator_hardcoded_params) > 0:
            estimator = estimator_fn(estimator_hardcoded_params)
            all_estimator_params = estimator.get_params()
            default_params = dict(
                set(all_estimator_params.items()) - set(estimator_hardcoded_params.items())
            )
            self._write_best_parameters_outputs(
                output_directory,
                best_hardcoded_params=estimator_hardcoded_params,
                default_params=default_params,
            )
        else:
            estimator = estimator_fn()
            default_params = estimator.get_params()
            self._write_best_parameters_outputs(output_directory, default_params=default_params)
        return estimator

    def _resolve_estimator_plugin(self, plugin_str, X_train, y_train, output_directory):
        plugin_str = plugin_str.replace("/", ".")
        estimator_fn = getattr(
            importlib.import_module(f"mlflow.pipelines.steps.{plugin_str}"),
            "get_estimator_and_best_params",
        )
        estimator, best_parameters = estimator_fn(
            X_train,
            y_train,
            self.task,
            self.step_config,
            self.pipeline_root,
            self.evaluation_metrics,
            self.primary_metric,
        )
        self.best_estimator_name = estimator.__class__.__name__
        self.best_estimator_class = (
            f"{estimator.__class__.__module__}.{estimator.__class__.__name__}"
        )
        self.best_parameters = best_parameters
        self._write_best_parameters_outputs(output_directory, automl_params=best_parameters)
        return estimator

    def _resolve_estimator(self, X_train, y_train, validation_df, run, output_directory):
        using_plugin = self.step_config.get("using", "estimator_spec")
        if using_plugin == "estimator_spec":
            return self._get_user_defined_estimator(
                X_train, y_train, validation_df, run, output_directory
            )
        else:
            return self._resolve_estimator_plugin(using_plugin, X_train, y_train, output_directory)

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

    def _get_tuning_df(self, run, params=None):
        exp_id = _get_experiment_id()
        primary_metric_tag = f"metrics.{self.primary_metric}_on_data_validation"
        order_str = (
            "DESC" if self.evaluation_metrics_greater_is_better[self.primary_metric] else "ASC"
        )
        tuning_runs = mlflow.search_runs(
            [exp_id],
            filter_string=f"tags.mlflow.parentRunId like '{run.info.run_id}'",
            order_by=[f"{primary_metric_tag} {order_str}", "attribute.start_time ASC"],
        )
        if params:
            params = [f"params.{param}" for param in params]
            tuning_runs = tuning_runs.filter(
                [f"metrics.{self.primary_metric}_on_data_validation", *params]
            )
        else:
            tuning_runs = tuning_runs.filter([f"metrics.{self.primary_metric}_on_data_validation"])
        tuning_runs = tuning_runs.reset_index().rename(
            columns={"index": "Model Rank", primary_metric_tag: self.primary_metric}
        )
        tuning_runs["Model Rank"] += 1
        return tuning_runs.head(10)

    def _build_step_card(
        self,
        eval_metrics,
        pred_and_error_df,
        model,
        model_schema,
        run_id,
        model_uri,
        worst_examples_df,
        train_df,
        output_directory,
        leaderboard_df=None,
        tuning_df=None,
    ):
        import pandas as pd
        from sklearn.utils import estimator_html_repr
        from sklearn import set_config

        card = BaseCard(self.pipeline_name, self.name)
        # Tab 0: model performance summary.
        metric_df = (
            get_merged_eval_metrics(
                eval_metrics,
                ordered_metric_names=[
                    self.primary_metric,
                    *self.user_defined_custom_metrics.keys(),
                ],
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

        # Tab 1: Model performance.
        card.add_tab(
            "Model Performance",
            "<h3 class='section-title'>Summary Metrics (Validation)</h3>{{ METRICS }} ",
        ).add_html("METRICS", metric_table_html)

        if not self.skip_data_profiling:
            # Tab 2: Prediction and error data profile.
            pred_and_error_df_profile = get_pandas_data_profiles(
                [
                    [
                        "Predictions and Errors (Validation Dataset)",
                        pred_and_error_df.reset_index(drop=True),
                    ]
                ]
            )
            card.add_tab("Data Profile (Predictions)", "{{PROFILE}}").add_pandas_profile(
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
            card.add_tab("Worst Predictions", "{{ WORST_EXAMPLES_TABLE }}").add_html(
                "WORST_EXAMPLES_TABLE", BaseCard.render_table(worst_examples_df)
            )
        )

        # Tab 6: Worst predictions profile vs train profile.
        if not self.skip_data_profiling:
            worst_prediction_profile = get_pandas_data_profiles(
                [
                    ["Worst Predictions", worst_examples_df.reset_index(drop=True)],
                    ["Train", train_df.reset_index(drop=True)],
                ]
            )
            card.add_tab(
                "Data Profile (Worst vs Train)", "{{ WORST_EXAMPLES_COMP }}"
            ).add_pandas_profile("WORST_EXAMPLES_COMP", worst_prediction_profile)

        # Tab 7: Leaderboard
        if leaderboard_df is not None:
            (
                card.add_tab("Leaderboard", "{{ LEADERBOARD_TABLE }}").add_html(
                    "LEADERBOARD_TABLE", BaseCard.render_table(leaderboard_df, hide_index=False)
                )
            )

        # Tab 8: Best Parameters (AutoML and Tuning)
        is_automl_run = self.step_config["using"].startswith("automl")
        best_parameters_yaml = os.path.join(output_directory, "best_parameters.yaml")

        if os.path.exists(best_parameters_yaml):
            best_parameters_card_tab = card.add_tab(
                f"Best Parameters {' (AutoML)' if is_automl_run else ''}",
                "{{ BEST_PARAMETERS }} ",
            )

            if is_automl_run:
                automl_estimator_str = (
                    f"<b>Best estimator:</b><br>"
                    f"<pre>{self.best_estimator_name}</pre><br>"
                    f"<b>Best estimator class:</b><br>"
                    f"<pre>{self.best_estimator_class}</pre><br><br>"
                )
            else:
                automl_estimator_str = ""

            best_parameters = open(best_parameters_yaml).read()
            best_parameters_card_tab.add_html(
                "BEST_PARAMETERS",
                f"{automl_estimator_str}<b>Best parameters:</b><br>"
                f"<pre>{best_parameters}</pre><br><br>",
            )

        # Tab 9: HP trials
        if tuning_df is not None:
            tuning_trials_card_tab = card.add_tab(
                "Tuning Trials",
                "{{ SEARCH_SPACE }}" + "{{ TUNING_TABLE_TITLE }}" + "{{ TUNING_TABLE }}",
            )
            tuning_params = yaml.dump(self.step_config["tuning"]["parameters"])
            tuning_trials_card_tab.add_html(
                "SEARCH_SPACE",
                f"<b>Tuning search space:</b> <br><pre>{tuning_params}</pre><br><br>",
            )
            tuning_trials_card_tab.add_html("TUNING_TABLE_TITLE", "<b>Tuning results:</b><br>")
            tuning_trials_card_tab.add_html(
                "TUNING_TABLE",
                BaseCard.render_table(
                    tuning_df.style.apply(
                        lambda row: pd.Series("font-weight: bold", row.index),
                        axis=1,
                        subset=tuning_df.index[0],
                    ),
                    hide_index=True,
                ),
            )

        # Tab 10: Run summary.
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
        step_config = {}
        if pipeline_config.get("steps", {}).get("train", {}) is not None:
            step_config.update(pipeline_config.get("steps", {}).get("train", {}))
        step_config["metrics"] = pipeline_config.get("metrics")
        step_config["template_name"] = pipeline_config.get("template")
        step_config["profile"] = pipeline_config.get("profile")
        step_config["target_col"] = pipeline_config.get("target_col")
        step_config.update(
            get_pipeline_tracking_config(
                pipeline_root_path=pipeline_root,
                pipeline_config=pipeline_config,
            ).to_dict()
        )
        return cls(step_config, pipeline_root, pipeline_config=pipeline_config)

    @property
    def name(self):
        return "train"

    @property
    def environment(self):
        environ = get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
        environ.update(get_run_tags_env_vars(pipeline_root_path=self.pipeline_root))
        return environ

    def get_artifacts(self):
        return [
            ModelArtifact(
                "model", self.pipeline_root, self.name, self.tracking_config.tracking_uri
            ),
            RunArtifact("run", self.pipeline_root, self.name, self.tracking_config.tracking_uri),
            HyperParametersArtifact("best_parameters", self.pipeline_root, self.name),
        ]

    def step_class(self):
        return StepClass.TRAINING

    def _tune_and_get_best_estimator_params(
        self,
        parent_run_id,
        estimator_hardcoded_params,
        estimator_fn,
        X_train,
        y_train,
        validation_df,
    ):
        tuning_params = self.step_config["tuning"]
        try:
            from hyperopt import fmin, Trials, SparkTrials, space_eval
        except ModuleNotFoundError:
            raise MlflowException(
                "Hyperopt not installed and is required if tuning is enabled",
                error_code=BAD_REQUEST,
            )

        # wrap training in objective fn
        def objective(X_train, y_train, validation_df, hyperparameter_args, on_worker=False):
            if on_worker:
                client = MlflowClient()
                parent_tags = client.get_run(parent_run_id).data.tags
                child_run = client.create_run(
                    _get_experiment_id(), tags={**parent_tags, "mlflow.parentRunId": parent_run_id}
                )
                run_args = {"run_id": child_run.info.run_id}
            else:
                run_args = {"nested": True}
            with mlflow.start_run(**run_args) as tuning_run:
                estimator_args = dict(estimator_hardcoded_params, **hyperparameter_args)
                estimator = estimator_fn(estimator_args)

                sample_fraction = self.step_config["sample_fraction"]

                # if sparktrials, then read from broadcast
                if tuning_params["parallelism"] > 1:
                    X_train = X_train.value
                    y_train = y_train.value
                    validation_df = validation_df.value

                X_train_sampled = X_train.sample(frac=sample_fraction, random_state=42)
                y_train_sampled = y_train.sample(frac=sample_fraction, random_state=42)

                estimator.fit(X_train_sampled, y_train_sampled)

                logged_estimator = self._log_estimator_to_mlflow(
                    estimator, X_train_sampled, on_worker=on_worker
                )

                eval_result = mlflow.evaluate(
                    model=logged_estimator.model_uri,
                    data=validation_df,
                    targets=self.target_col,
                    model_type="regressor",
                    evaluators="default",
                    dataset_name="validation",
                    custom_metrics=_load_custom_metric_functions(
                        self.pipeline_root,
                        self.evaluation_metrics.values(),
                    ),
                    evaluator_config={
                        "log_model_explainability": False,
                    },
                )
                autologged_params = mlflow.get_run(run_id=tuning_run.info.run_id).data.params
                manual_log_params = {}
                for param_name, param_value in estimator_args.items():
                    if param_name in autologged_params:
                        if not TrainStep.is_tuning_param_equal(
                            param_value, autologged_params[param_name]
                        ):
                            _logger.warning(
                                f"Failed to log search space parameter due to conflict. "
                                f"Attempted to log search space parameter {param_name} as "
                                f"{param_value}, but {param_name} is already logged as "
                                f"{autologged_params[param_name]} during training. "
                                f"Are you passing `estimator_params` properly to the "
                                f" estimator?"
                            )
                    else:
                        manual_log_params[param_name] = param_value

                if len(manual_log_params) > 0:
                    mlflow.log_params(manual_log_params)

                # return +/- metric
                sign = -1 if self.evaluation_metrics_greater_is_better[self.primary_metric] else 1
                return sign * eval_result.metrics[self.primary_metric]

        search_space = TrainStep.construct_search_space_from_yaml(tuning_params["parameters"])
        algo_type, algo_name = tuning_params["algorithm"].rsplit(".", 1)
        tuning_algo = getattr(importlib.import_module(algo_type, "hyperopt"), algo_name)
        max_trials = tuning_params["max_trials"]
        parallelism = tuning_params["parallelism"]

        if parallelism > 1:
            from pyspark.sql import SparkSession

            spark_session = SparkSession.builder.config(
                "spark.databricks.mlflow.trackHyperopt.enabled", "false"
            ).getOrCreate()
            sc = spark_session.sparkContext

            X_train = sc.broadcast(X_train)
            y_train = sc.broadcast(y_train)
            validation_df = sc.broadcast(validation_df)

            hp_trials = SparkTrials(parallelism, spark_session=spark_session)
            on_worker = True
        else:
            hp_trials = Trials()
            on_worker = False

        fmin_kwargs = {
            "fn": lambda params: objective(
                X_train, y_train, validation_df, params, on_worker=on_worker
            ),
            "space": search_space,
            "algo": tuning_algo,
            "max_evals": max_trials,
            "trials": hp_trials,
        }
        if "early_stop_fn" in tuning_params:
            train_module_name, early_stop_fn_name = tuning_params["early_stop_fn"].rsplit(".", 1)
            early_stop_fn = getattr(importlib.import_module(train_module_name), early_stop_fn_name)
            fmin_kwargs["early_stop_fn"] = early_stop_fn
        best_hp_params = fmin(**fmin_kwargs)
        best_hp_params = space_eval(search_space, best_hp_params)
        best_hp_estimator_loss = hp_trials.best_trial["result"]["loss"]
        if len(estimator_hardcoded_params) > 1:
            hardcoded_estimator_loss = objective(
                X_train, y_train, validation_df, estimator_hardcoded_params
            )

            if best_hp_estimator_loss < hardcoded_estimator_loss:
                best_hardcoded_params = {
                    param_name: param_value
                    for param_name, param_value in estimator_hardcoded_params.items()
                    if param_name not in best_hp_params
                }
            else:
                best_hp_params = {}
                best_hardcoded_params = estimator_hardcoded_params
        else:
            best_hardcoded_params = {}
        return (best_hardcoded_params, best_hp_params)

    def _log_estimator_to_mlflow(self, estimator, X_train_sampled, on_worker=False):
        from mlflow.models.signature import infer_signature

        if hasattr(estimator, "best_score_") and (type(estimator.best_score_) in [int, float]):
            mlflow.log_metric("best_cv_score", estimator.best_score_)
        if hasattr(estimator, "best_params_"):
            mlflow.log_params(estimator.best_params_)

        if on_worker:
            mlflow.log_params(estimator.get_params())
            estimator_tags = {
                "estimator_name": estimator.__class__.__name__,
                "estimator_class": (
                    estimator.__class__.__module__ + "." + estimator.__class__.__name__
                ),
            }
            mlflow.set_tags(estimator_tags)
        estimator_schema = infer_signature(
            X_train_sampled, estimator.predict(X_train_sampled.copy())
        )
        logged_estimator = mlflow.sklearn.log_model(
            estimator,
            f"{self.name}/estimator",
            signature=estimator_schema,
            code_paths=self.code_paths,
        )
        return logged_estimator

    def _write_best_parameters_outputs(  # pylint: disable=dangerous-default-value
        self,
        output_directory,
        best_hp_params={},
        best_hardcoded_params={},
        automl_params={},
        default_params={},
    ):
        if best_hp_params or best_hardcoded_params or automl_params or default_params:
            best_parameters_path = os.path.join(output_directory, "best_parameters.yaml")
            if os.path.exists(best_parameters_path):
                os.remove(best_parameters_path)
            with open(best_parameters_path, "a") as file:
                self._write_one_param_output(automl_params, file, "automl parameters")
                self._write_one_param_output(best_hp_params, file, "tuned hyperparameters")
                self._write_one_param_output(best_hardcoded_params, file, "hardcoded parameters")
                self._write_one_param_output(default_params, file, "default parameters")
            mlflow.log_artifact(best_parameters_path, artifact_path="train")

    def _write_one_param_output(self, params, file, caption):
        if params:
            file.write(f"# {caption} \n")
            self._safe_dump_with_numeric_values(params, file, default_flow_style=False)
            file.write("\n")

    def _safe_dump_with_numeric_values(self, data, file, **kwargs):
        import numpy as np

        processed_data = {}
        for key, value in data.items():
            if isinstance(value, np.floating):
                processed_data[key] = float(value)
            elif isinstance(value, np.integer):
                processed_data[key] = int(value)
            else:
                processed_data[key] = value

        if len(processed_data) > 0:
            yaml.safe_dump(processed_data, file, **kwargs)
