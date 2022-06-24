import importlib
import logging
import os
import sys

import cloudpickle
import pandas as pd

import mlflow
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.models.signature import infer_signature
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.step import get_merged_eval_metrics
from mlflow.pipelines.utils.tracking import (
    get_pipeline_tracking_config,
    apply_pipeline_tracking_config,
    TrackingConfig,
    get_run_tags_env_vars,
    log_code_snapshot,
)
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.types import ColSpec

_logger = logging.getLogger(__name__)


class TrainStep(BaseStep):
    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
        self.tracking_config = TrackingConfig.from_dict(step_config)
        self.target_col = self.step_config.get("target_col")
        self.train_module_name, self.train_method_name = self.step_config["train_method"].rsplit(
            ".", 1
        )
        self.primary_metric = (self.step_config.get("metrics") or {}).get(
            "primary", "root_mean_squared_error"
        )

    def _run(self, output_directory):
        from sklearn.pipeline import make_pipeline
        import shutil

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
        train_fn = getattr(importlib.import_module(self.train_module_name), self.train_method_name)
        estimator = train_fn()
        mlflow.autolog(log_models=False)

        with mlflow.start_run() as run:
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
                relative_path="model",
            )
            if os.path.exists(output_model_path) and os.path.isdir(output_model_path):
                shutil.rmtree(output_model_path)
            mlflow.sklearn.save_model(model, output_model_path)

            with open(os.path.join(output_directory, "run_id"), "w") as f:
                f.write(run.info.run_id)
            log_code_snapshot(self.pipeline_root, run.info.run_id)

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
                    custom_metrics=self._load_custom_metric_functions(),
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
        worst_examples_df = BaseStep._generate_worst_examples_dataframe(
            raw_validation_df, prediction_result, self.target_col
        )

        card = self._build_step_card(
            eval_metrics=eval_metrics,
            pred_and_error_df=pred_and_error_df,
            model=model,
            model_schema=model_schema,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            worst_examples_df=worst_examples_df,
        )
        card.save_as_html(output_directory)
        for step_name in ("ingest", "split", "transform", "train"):
            self._log_step_card(run.info.run_id, step_name)

        return card

    def _build_step_card(
        self,
        eval_metrics,
        pred_and_error_df,
        model,
        model_schema,
        run_id,
        model_uri,
        worst_examples_df,
    ):
        from pandas_profiling import ProfileReport
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
        card.add_tab(
            "Model Performance Summary Metrics",
            "<h3 class='section-title'>Summary Metrics</h3>{{ METRICS }} ",
        ).add_html("METRICS", metric_table_html)

        # Tab 1: Prediction and error data profile.
        pred_and_error_df_profile = ProfileReport(
            pred_and_error_df.reset_index(drop=True),
            title="Predictions and Errors (Validation Dataset)",
            minimal=True,
            progress_bar=False,
        )
        card.add_tab("Profile of Predictions and Errors", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", pred_and_error_df_profile
        )
        # Tab 2: Model architecture.
        set_config(display="diagram")
        model_repr = estimator_html_repr(model)
        card.add_tab("Model Architecture", "{{MODEL_ARCH}}").add_html("MODEL_ARCH", model_repr)

        # Tab 3: Inferred model (transformer + estimator) schema.
        def render_schema(inputs, title):
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

        # Tab 4: Examples with Largest Prediction Error
        (
            card.add_tab(
                "Examples with Largest Prediction Error", "{{ WORST_EXAMPLES_TABLE }}"
            ).add_html("WORST_EXAMPLES_TABLE", BaseCard.render_table(worst_examples_df))
        )

        # Tab 5: Run summary.
        (
            card.add_tab(
                "Run Summary",
                "{{ RUN_ID }} "
                + "{{ MODEL_URI }}"
                + "{{ EXE_DURATION }}"
                + "{{ LAST_UPDATE_TIME }}",
            )
            .add_markdown("RUN_ID", f"**MLflow Run ID:** `{run_id}`")
            .add_markdown("MODEL_URI", f"**MLflow Model URI:** `{model_uri}`")
        )

        return card

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["train"]
            step_config["metrics"] = pipeline_config.get("metrics")
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
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "train"

    @property
    def environment(self):
        environ = get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
        environ.update(get_run_tags_env_vars())
        return environ
