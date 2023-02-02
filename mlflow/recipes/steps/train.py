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
from mlflow.recipes.artifacts import (
    ModelArtifact,
    RunArtifact,
    HyperParametersArtifact,
    DataframeArtifact,
)
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep
from mlflow.recipes.step import StepClass
from mlflow.recipes.utils.execution import (
    get_step_output_path,
    _MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME_ENV_VAR,
)
from mlflow.recipes.utils.metrics import (
    _get_error_fn,
    _get_builtin_metrics,
    _get_primary_metric,
    _get_custom_metrics,
    _get_model_type_from_template,
    _load_custom_metrics,
    _get_extended_task,
    transform_multiclass_metrics_dict,
)
from mlflow.recipes.utils.step import (
    get_merged_eval_metrics,
    get_pandas_data_profiles,
    validate_classification_config,
)
from mlflow.recipes.utils.tracking import (
    get_recipe_tracking_config,
    apply_recipe_tracking_config,
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
    MLFLOW_RECIPE_TEMPLATE_NAME,
    MLFLOW_RECIPE_PROFILE_NAME,
    MLFLOW_RECIPE_STEP_NAME,
)
from mlflow.utils.string_utils import strip_prefix
from mlflow.recipes.utils.wrapped_recipe_model import WrappedRecipeModel
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir

_REBALANCING_CUTOFF = 5000
_REBALANCING_DEFAULT_RATIO = 0.3
_USER_DEFINED_TRAIN_STEP_MODULE = "steps.train"

_logger = logging.getLogger(__name__)


class TrainStep(BaseStep):
    MODEL_ARTIFACT_RELATIVE_PATH = "model"
    SKLEARN_MODEL_ARTIFACT_RELATIVE_PATH = "sk_model"
    PREDICTED_TRAINING_DATA_RELATIVE_PATH = "predicted_training_data.parquet"

    def __init__(self, step_config, recipe_root, recipe_config=None):
        super().__init__(step_config, recipe_root)
        self.tracking_config = TrackingConfig.from_dict(self.step_config)
        self.recipe_config = recipe_config

    def _validate_and_apply_step_config(self):
        if "using" in self.step_config:
            if self.step_config["using"] not in ["custom", "automl/flaml"]:
                raise MlflowException(
                    f"Invalid train step configuration value {self.step_config['using']} for "
                    f"key 'using'. Supported values are: ['custom', 'automl/flaml']",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            self.step_config["using"] = "custom"

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
        self.rebalance_training_data = self.step_config.get("rebalance_training_data", True)
        self.skip_data_profiling = self.step_config.get("skip_data_profiling", False)
        if "estimator_method" not in self.step_config and self.step_config["using"] == "custom":
            raise MlflowException(
                "Missing 'estimator_method' configuration in the train step, "
                "which is using 'custom'.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        self.predict_prefix = self.step_config.get("predict_prefix", "predicted_")
        self.predict_scores_for_all_classes = self.step_config.get(
            "predict_scores_for_all_classes", True
        )
        self.primary_metric = _get_primary_metric(
            self.step_config.get("primary_metric"), self.extended_task
        )
        builtin_metrics = _get_builtin_metrics(self.extended_task)
        custom_metrics = _get_custom_metrics(self.step_config, self.extended_task)
        self.user_defined_custom_metrics = {metric.name: metric for metric in custom_metrics}
        self.evaluation_metrics = {metric.name: metric for metric in builtin_metrics}
        self.evaluation_metrics.update(self.user_defined_custom_metrics)
        self.evaluation_metrics_greater_is_better = {
            metric.name: metric.greater_is_better for metric in builtin_metrics
        }
        self.evaluation_metrics_greater_is_better.update(
            {metric.name: metric.greater_is_better for metric in custom_metrics}
        )
        if self.primary_metric is not None and self.primary_metric not in self.evaluation_metrics:
            raise MlflowException(
                f"The primary metric '{self.primary_metric}' is a custom metric, but its"
                " corresponding custom metric configuration is missing from `recipe.yaml`.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.code_paths = [os.path.join(self.recipe_root, "steps")]

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

    def _fitted_estimator(self, *args):
        if "calibrate_proba" in self.step_config:
            return self._calibrated_classifier_fitted_estimator(*args)
        else:
            return self._label_encoded_fitted_estimator(*args)

    def _calibrated_classifier_fitted_estimator(self, estimator, X_train, y_train):
        original_estimator = estimator.fit(X_train, y_train)
        if "classification" in self.recipe:
            from sklearn.calibration import CalibratedClassifierCV

            estimator = CalibratedClassifierCV(
                estimator, method=self.step_config["calibrate_proba"]
            )
            estimator.fit(X_train, y_train)

        return estimator, {"original_estimator": original_estimator}

    def _label_encoded_fitted_estimator(self, estimator, X_train, y_train):
        from sklearn.preprocessing import LabelEncoder

        label_encoder = None
        target_column_class_labels = None
        encoded_y_train = y_train
        if "classification" in self.recipe:
            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)
            target_column_class_labels = label_encoder.classes_
            import pandas as pd

            encoded_y_train = pd.Series(label_encoder.transform(y_train))

        def inverse_label_encoder(predicted_output):
            if not label_encoder:
                return predicted_output

            return label_encoder.inverse_transform(predicted_output)

        estimator.fit(X_train, encoded_y_train)
        original_predict = estimator.predict

        def wrapped_predict(*args, **kwargs):
            return inverse_label_encoder(original_predict(*args, **kwargs))

        estimator.predict = wrapped_predict

        return estimator, {"target_column_class_labels": target_column_class_labels}

    def _run(self, output_directory):
        def my_warn(*args, **kwargs):
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
            import numpy as np
            import shutil
            import sklearn
            from sklearn.pipeline import make_pipeline
            from sklearn.utils.class_weight import compute_class_weight
            from mlflow.models.signature import infer_signature

            open(os.path.join(output_directory, "warning_logs.txt"), "w")

            apply_recipe_tracking_config(self.tracking_config)

            transformed_training_data_path = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="transform",
                relative_path="transformed_training_data.parquet",
            )
            train_df = pd.read_parquet(transformed_training_data_path)
            validate_classification_config(
                self.task, self.positive_class, train_df, self.target_col
            )
            self.using_rebalancing = False
            if self.extended_task == "classification/binary":
                classes = np.unique(train_df[self.target_col])
                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=classes,
                    y=train_df[self.target_col],
                )
                self.original_class_weights = dict(zip(classes, class_weights))
                if self.rebalance_training_data and len(classes) == 2:
                    if len(train_df) > _REBALANCING_CUTOFF:
                        self.using_rebalancing = True
                        train_df = self._rebalance_classes(train_df)
                    else:
                        _logger.info(
                            f"Training data has less than {_REBALANCING_CUTOFF} rows, "
                            f"skipping rebalancing."
                        )

            X_train, y_train = train_df.drop(columns=[self.target_col]), train_df[self.target_col]

            transformed_validation_data_path = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="transform",
                relative_path="transformed_validation_data.parquet",
            )
            validation_df = pd.read_parquet(transformed_validation_data_path)

            raw_training_data_path = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="split",
                relative_path="train.parquet",
            )
            raw_train_df = pd.read_parquet(raw_training_data_path)
            raw_X_train = raw_train_df.drop(columns=[self.target_col])

            raw_validation_data_path = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="split",
                relative_path="validation.parquet",
            )
            raw_validation_df = pd.read_parquet(raw_validation_data_path)

            transformer_path = get_step_output_path(
                recipe_root_path=self.recipe_root,
                step_name="transform",
                relative_path="transformer.pkl",
            )

            tags = {
                MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.RECIPE),
                MLFLOW_RECIPE_TEMPLATE_NAME: self.step_config["recipe"],
                MLFLOW_RECIPE_PROFILE_NAME: self.step_config["profile"],
                MLFLOW_RECIPE_STEP_NAME: os.getenv(
                    _MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME_ENV_VAR
                ),
            }

            run_name = self.tracking_config.run_name
            best_estimator_params = None
            mlflow.autolog(log_models=False, silent=True)
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                estimator = self._resolve_estimator(
                    X_train, y_train, validation_df, run, output_directory
                )
                fitted_estimator, additional_fitted_args = self._fitted_estimator(
                    estimator, X_train, y_train
                )
                logged_estimator = self._log_estimator_to_mlflow(fitted_estimator, X_train)

                # Create a recipe consisting of the transformer+model for test data evaluation
                with open(transformer_path, "rb") as f:
                    transformer = cloudpickle.load(f)
                mlflow.sklearn.log_model(
                    transformer, "transform/transformer", code_paths=self.code_paths
                )

                trained_pipeline = make_pipeline(transformer, fitted_estimator)
                # Creating a wrapped recipe model which exposes a single predict function
                # so it can output both predict and predict_proba(for a classification problem)
                # at the same time.
                wrapped_model = WrappedRecipeModel(
                    self.predict_scores_for_all_classes,
                    self.predict_prefix,
                    target_column_class_labels=additional_fitted_args.get(
                        "target_column_class_labels"
                    ),
                )

                model_uri = get_step_output_path(
                    recipe_root_path=self.recipe_root,
                    step_name=self.name,
                    relative_path=TrainStep.MODEL_ARTIFACT_RELATIVE_PATH,
                )
                sklearn_model_uri = get_step_output_path(
                    recipe_root_path=self.recipe_root,
                    step_name=self.name,
                    relative_path=TrainStep.SKLEARN_MODEL_ARTIFACT_RELATIVE_PATH,
                )
                if os.path.exists(model_uri):
                    shutil.rmtree(model_uri)
                if os.path.exists(sklearn_model_uri):
                    shutil.rmtree(sklearn_model_uri)

                # Saving the sklearn model as a separate output since `mlflow.evaluate()`, which is
                # used in evaluate step of the recipe, needs this model's sklearn representation
                # to computes metrics (the pyfunc representation of the user-facing model logged to
                # MLflow Tracking is not currently compatible with `mlflow.evaluate()`)
                mlflow.sklearn.save_model(trained_pipeline, sklearn_model_uri)
                artifacts = {"model_path": sklearn_model_uri}
                with TempDir() as tmp:
                    # Saving a temp model so that the output schema (signature) of the model's
                    # pyfunc representation can be inferred and included when logging the model
                    # to MLflow Tracking. Unfortunately, there is currently no easy way to infer
                    # the model's signature without first saving a copy of it, and there is no easy
                    # way to add an inferred signature to an existing model
                    pyfunc_model_tmp_path = os.path.join(tmp.path(), "pyfunc_model")
                    mlflow.pyfunc.save_model(
                        path=pyfunc_model_tmp_path,
                        python_model=wrapped_model,
                        artifacts=artifacts,
                    )
                    tempModel = mlflow.pyfunc.load_model(pyfunc_model_tmp_path)
                    model_schema = infer_signature(
                        raw_X_train, tempModel.predict(raw_X_train.copy())
                    )
                    mlflow.pyfunc.save_model(
                        path=model_uri,
                        python_model=wrapped_model,
                        artifacts=artifacts,
                        signature=model_schema,
                        code_path=self.code_paths,
                    )
                model = mlflow.pyfunc.load_model(model_uri)
                # Adding a sklearn flavor to the pyfunc model so models could be loaded easily
                # using mlflow.sklearn.load_model
                tmp_model_info = Model.load(model_uri)
                model_data_subpath = os.path.join(
                    "artifacts", TrainStep.SKLEARN_MODEL_ARTIFACT_RELATIVE_PATH, "model.pkl"
                )
                model_info = Model(
                    artifact_path="train/model",
                    run_id=run.info.run_id,
                    utc_time_created=tmp_model_info.utc_time_created,
                    flavors=tmp_model_info.flavors,
                    signature=tmp_model_info.signature,  # ModelSignature
                    saved_input_example_info=tmp_model_info.saved_input_example_info,
                    model_uuid=tmp_model_info.model_uuid,
                    mlflow_version=tmp_model_info.mlflow_version,
                    metadata=tmp_model_info.metadata,
                )
                model_info.add_flavor(
                    mlflow.sklearn.FLAVOR_NAME,
                    pickled_model=model_data_subpath,
                    sklearn_version=sklearn.__version__,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    code="code",
                )
                model_info.save(f"{model_uri}/MLmodel")
                mlflow.log_artifacts(model_uri, "train/model")

                with open(os.path.join(output_directory, "run_id"), "w") as f:
                    f.write(run.info.run_id)
                log_code_snapshot(
                    self.recipe_root, run.info.run_id, recipe_config=self.recipe_config
                )

                eval_metrics = {}
                for dataset_name, (dataset, metric_prefix) in {
                    "training": (train_df, "training_"),
                    "validation": (validation_df, "val_"),
                }.items():
                    eval_config = {
                        "log_model_explainability": False,
                        "metric_prefix": metric_prefix,
                    }
                    if self.positive_class is not None:
                        eval_config["pos_label"] = self.positive_class
                    eval_result = mlflow.evaluate(
                        model=logged_estimator.model_uri,
                        data=dataset,
                        targets=self.target_col,
                        model_type=_get_model_type_from_template(self.recipe),
                        evaluators="default",
                        custom_metrics=_load_custom_metrics(
                            self.recipe_root,
                            self.evaluation_metrics.values(),
                        ),
                        evaluator_config=eval_config,
                    )
                    eval_result.save(os.path.join(output_directory, f"eval_{dataset_name}"))
                    eval_metrics[dataset_name] = {
                        strip_prefix(k, metric_prefix): v for k, v in eval_result.metrics.items()
                    }

            target_data = raw_validation_df[self.target_col]
            prediction_result = model.predict(raw_validation_df.drop(self.target_col, axis=1))

            use_probability_for_error_rate = False
            if isinstance(prediction_result, pd.DataFrame) and {
                f"{self.predict_prefix}label",
                f"{self.predict_prefix}score",
            }.issubset(prediction_result.columns):
                if self.positive_class:
                    prediction_result_for_error = prediction_result[
                        f"{self.predict_prefix}score_{self.positive_class}"
                    ].values
                    # use_probability_for_error_rate to true to compute error function
                    # based on positive class
                    use_probability_for_error_rate = True
                else:
                    prediction_result_for_error = prediction_result[
                        f"{self.predict_prefix}score"
                    ].values

                prediction_result = prediction_result[f"{self.predict_prefix}label"].values
            else:
                prediction_result_for_error = prediction_result
            error_fn = _get_error_fn(
                self.recipe,
                use_probability=use_probability_for_error_rate,
                positive_class=self.positive_class,
            )
            pred_and_error_df = pd.DataFrame(
                {
                    "target": target_data,
                    "prediction": prediction_result,
                    "error": error_fn(prediction_result_for_error, target_data.to_numpy()),
                }
            )
            calibrated_plot = None
            train_predictions = model.predict(raw_train_df.drop(self.target_col, axis=1))
            if isinstance(train_predictions, pd.DataFrame) and {
                f"{self.predict_prefix}label",
                f"{self.predict_prefix}score",
            }.issubset(train_predictions.columns):
                predicted_training_data = raw_train_df.assign(
                    predicted_data=train_predictions[f"{self.predict_prefix}label"],
                    predicted_score=train_predictions[f"{self.predict_prefix}score"].values,
                )
                if self.positive_class:
                    worst_examples_df = BaseStep._generate_worst_examples_dataframe(
                        raw_train_df,
                        train_predictions[f"{self.predict_prefix}label"].values,
                        error_fn(
                            train_predictions[
                                f"{self.predict_prefix}score_{self.positive_class}"
                            ].values,
                            raw_train_df[self.target_col].to_numpy(),
                        ),
                        self.target_col,
                    )

                    if "calibrate_proba" in self.step_config and hasattr(
                        additional_fitted_args.get("original_estimator"), "predict_proba"
                    ):
                        from sklearn.calibration import CalibrationDisplay

                        calibrated_plot = CalibrationDisplay.from_estimator(
                            additional_fitted_args.get("original_estimator"),
                            raw_train_df.drop(self.target_col, axis=1),
                            raw_train_df[self.target_col],
                            pos_label=self.positive_class,
                        )
                else:
                    # compute worst examples data_frame only if positive class exists
                    worst_examples_df = pd.DataFrame()

            else:
                predicted_training_data = raw_train_df.assign(predicted_data=train_predictions)
                worst_examples_df = BaseStep._generate_worst_examples_dataframe(
                    raw_train_df,
                    train_predictions,
                    error_fn(train_predictions, raw_train_df[self.target_col].to_numpy()),
                    self.target_col,
                )
            predicted_training_data.to_parquet(
                os.path.join(output_directory, TrainStep.PREDICTED_TRAINING_DATA_RELATIVE_PATH)
            )

            leaderboard_df = None
            try:
                leaderboard_df = self._get_leaderboard_df(run, eval_metrics)
            except Exception as e:
                _logger.warning(
                    "Failed to build model leaderboard due to unexpected failure: %s", e
                )
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
                model_uri=model_uri,
                worst_examples_df=worst_examples_df,
                train_df=raw_train_df,
                output_directory=output_directory,
                leaderboard_df=leaderboard_df,
                tuning_df=tuning_df,
                calibrated_plot=calibrated_plot,
            )
            card.save_as_html(output_directory)
            for step_name in ("ingest", "split", "transform", "train"):
                self._log_step_card(run.info.run_id, step_name)
            return card
        finally:
            warnings.warn = original_warn

    def _get_user_defined_estimator(self, X_train, y_train, validation_df, run, output_directory):
        sys.path.append(self.recipe_root)
        estimator_fn = getattr(
            importlib.import_module(_USER_DEFINED_TRAIN_STEP_MODULE),
            self.step_config["estimator_method"],
        )
        estimator_hardcoded_params = self.step_config["estimator_params"]

        # if using rebalancing pass in original class weights to preserve original distribution
        if self.using_rebalancing:
            estimator_hardcoded_params["class_weight"] = self.original_class_weights

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
            default_params_keys = all_estimator_params.keys() - best_combined_params.keys()
            default_params = {k: all_estimator_params[k] for k in default_params_keys}
            self._write_best_parameters_outputs(
                output_directory,
                best_hp_params=best_hp_params,
                best_hardcoded_params=estimator_hardcoded_params,
                default_params=default_params,
            )
        elif len(estimator_hardcoded_params) > 0:
            estimator = estimator_fn(estimator_hardcoded_params)
            all_estimator_params = estimator.get_params()
            default_params_keys = all_estimator_params.keys() - estimator_hardcoded_params.keys()
            default_params = {k: all_estimator_params[k] for k in default_params_keys}
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
        estimator_fn = importlib.import_module(
            f"mlflow.recipes.steps.{plugin_str}"
        ).get_estimator_and_best_params
        estimator, best_parameters = estimator_fn(
            X_train,
            y_train,
            self.task,
            self.extended_task,
            self.step_config,
            self.recipe_root,
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
        using_plugin = self.step_config.get("using", "custom")
        if using_plugin == "custom":
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
            order_by=[f"metrics.{self.primary_metric} {primary_metric_order}"],
        )

        metric_names = self.evaluation_metrics.keys()

        leaderboard_items = []
        for old_run in search_result:
            if all(metric_key in old_run.data.metrics for metric_key in metric_names):
                leaderboard_items.append(
                    {
                        "Run ID": old_run.info.run_id,
                        "Run Time": datetime.datetime.fromtimestamp(
                            old_run.info.start_time // 1000
                        ),
                        **{
                            metric_name: old_run.data.metrics[metric_key]
                            for metric_name, metric_key in zip(metric_names, metric_names)
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
                lambda s: s.map(lambda x: f"{x:.6g}")  # pylint: disable=unnecessary-lambda
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
        primary_metric_tag = f"metrics.{self.primary_metric}"
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
            tuning_runs = tuning_runs.filter([f"metrics.{self.primary_metric}", *params])
        else:
            tuning_runs = tuning_runs.filter([f"metrics.{self.primary_metric}"])
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
        calibrated_plot=None,
    ):
        import pandas as pd
        from sklearn.utils import estimator_html_repr
        from sklearn import set_config

        card = BaseCard(self.recipe_name, self.name)
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
        model_repr = estimator_html_repr(
            mlflow.sklearn.load_model(
                os.path.join(model_uri, "artifacts", TrainStep.SKLEARN_MODEL_ARTIFACT_RELATIVE_PATH)
            )
        )
        card.add_tab("Model Architecture", "{{MODEL_ARCH}}").add_html("MODEL_ARCH", model_repr)

        # Tab 4: Inferred model (transformer + estimator) schema.
        def render_schema(inputs, title):
            from mlflow.types import ColSpec

            table = BaseCard.render_table(
                {
                    "Name": "  " + (spec.name or "-"),
                    "Type": repr(spec.type) if isinstance(spec, ColSpec) else repr(spec),
                }
                for spec in inputs
            )
            return f'<div style="margin: 5px"><h2>{title}</h2>{table}</div>'

        schema_tables = [render_schema(model_schema.inputs.inputs, "Inputs")]
        if model_schema.outputs:
            schema_tables += [render_schema(model_schema.outputs.inputs, "Outputs")]

        card.add_tab("Model Schema", "{{MODEL_SCHEMA}}").add_html(
            "MODEL_SCHEMA",
            '<div style="display: flex">{tables}</div>'.format(tables="\n".join(schema_tables)),
        )

        # Tab 5: Examples with Largest Prediction Error
        if not worst_examples_df.empty:
            (
                card.add_tab("Worst Predictions", "{{ WORST_EXAMPLES_TABLE }}").add_html(
                    "WORST_EXAMPLES_TABLE", BaseCard.render_table(worst_examples_df)
                )
            )

        if calibrated_plot:
            calibrated_plot_location = os.path.join(output_directory, "calibrated_plot_location")
            calibrated_plot.figure_.savefig(calibrated_plot_location, format="png")
            (
                card.add_tab("Prob. Calibration", "{{ CALIBRATED_PLOT }}").add_image(
                    "CALIBRATED_PLOT", calibrated_plot_location
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

        # Tab 10: Warning log outputs.
        warning_output_path = os.path.join(output_directory, "warning_logs.txt")
        if os.path.exists(warning_output_path):
            warnings_output_tab = card.add_tab("Warning Logs", "{{ STEP_WARNINGS }}")
            warnings_output_tab.add_html(
                "STEP_WARNINGS", f"<pre>{open(warning_output_path).read()}</pre>"
            )

        # Tab 11: Run summary.
        run_card_tab = card.add_tab(
            "Run Summary",
            "{{ RUN_ID }} " + "{{ MODEL_URI }}" + "{{ EXE_DURATION }}" + "{{ LAST_UPDATE_TIME }}",
        )
        model_uri_path = f"runs:/{run_id}/train/model"
        run_url = get_databricks_run_url(
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run_id,
        )
        model_url = get_databricks_run_url(
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run_id,
            artifact_path=re.sub(r"^.*?%s" % run_id, "", model_uri_path),
        )

        if run_url is not None:
            run_card_tab.add_html(
                "RUN_ID", f"<b>MLflow Run ID:</b> <a href={run_url}>{run_id}</a><br><br>"
            )
        else:
            run_card_tab.add_markdown("RUN_ID", f"**MLflow Run ID:** `{run_id}`")

        if model_url is not None:
            run_card_tab.add_html(
                "MODEL_URI", f"<b>MLflow Model URI:</b> <a href={model_url}>{model_uri_path}</a>"
            )
        else:
            run_card_tab.add_markdown("MODEL_URI", f"**MLflow Model URI:** `{model_uri_path}`")

        return card

    @classmethod
    def from_recipe_config(cls, recipe_config, recipe_root):
        step_config = {}
        if recipe_config.get("steps", {}).get("train", {}) is not None:
            step_config.update(recipe_config.get("steps", {}).get("train", {}))
        if recipe_config.get("custom_metrics") is not None:
            step_config["custom_metrics"] = recipe_config["custom_metrics"]
        if recipe_config.get("primary_metric") is not None:
            step_config["primary_metric"] = recipe_config["primary_metric"]
        step_config["recipe"] = recipe_config.get("recipe")
        step_config["profile"] = recipe_config.get("profile")
        step_config["target_col"] = recipe_config.get("target_col")
        if "positive_class" in recipe_config:
            step_config["positive_class"] = recipe_config.get("positive_class")
        step_config.update(
            get_recipe_tracking_config(
                recipe_root_path=recipe_root,
                recipe_config=recipe_config,
            ).to_dict()
        )
        return cls(step_config, recipe_root, recipe_config=recipe_config)

    @property
    def name(self):
        return "train"

    @property
    def environment(self):
        environ = get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
        environ.update(get_run_tags_env_vars(recipe_root_path=self.recipe_root))
        return environ

    def get_artifacts(self):
        return [
            ModelArtifact("model", self.recipe_root, self.name, self.tracking_config.tracking_uri),
            RunArtifact("run", self.recipe_root, self.name, self.tracking_config.tracking_uri),
            HyperParametersArtifact("best_parameters", self.recipe_root, self.name),
            DataframeArtifact(
                "predicted_training_data",
                self.recipe_root,
                self.name,
                TrainStep.PREDICTED_TRAINING_DATA_RELATIVE_PATH,
            ),
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
            run_name = self.tracking_config.run_name
            if on_worker:
                client = MlflowClient()
                parent_tags = client.get_run(parent_run_id).data.tags
                child_run = client.create_run(
                    _get_experiment_id(),
                    tags={**parent_tags, "mlflow.parentRunId": parent_run_id},
                    run_name=run_name,
                )
                run_args = {"run_id": child_run.info.run_id}
            else:
                run_args = {"run_name": run_name, "nested": True}
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

                fitted_estimator, _ = self._fitted_estimator(
                    estimator, X_train_sampled, y_train_sampled
                )

                logged_estimator = self._log_estimator_to_mlflow(
                    fitted_estimator, X_train_sampled, on_worker=on_worker
                )

                eval_result = mlflow.evaluate(
                    model=logged_estimator.model_uri,
                    data=validation_df,
                    targets=self.target_col,
                    model_type=_get_model_type_from_template(self.recipe),
                    evaluators="default",
                    custom_metrics=_load_custom_metrics(
                        self.recipe_root,
                        self.evaluation_metrics.values(),
                    ),
                    evaluator_config={
                        "log_model_explainability": False,
                        "pos_label": self.positive_class,
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
                transformed_metrics = transform_multiclass_metrics_dict(
                    eval_result.metrics, self.extended_task
                )
                sign = -1 if self.evaluation_metrics_greater_is_better[self.primary_metric] else 1
                return sign * transformed_metrics[self.primary_metric]

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
            try:
                train_module_name, early_stop_fn_name = tuning_params["early_stop_fn"].rsplit(
                    ".", 1
                )
            except ValueError:
                early_stop_fn_name = tuning_params["early_stop_fn"]
                train_module_name = _USER_DEFINED_TRAIN_STEP_MODULE
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

    def _write_best_parameters_outputs(
        self,
        output_directory,
        best_hp_params=None,
        best_hardcoded_params=None,
        automl_params=None,
        default_params=None,
    ):
        if best_hp_params or best_hardcoded_params or automl_params or default_params:
            best_parameters_path = os.path.join(output_directory, "best_parameters.yaml")
            if os.path.exists(best_parameters_path):
                os.remove(best_parameters_path)
            with open(best_parameters_path, "a") as file:
                self._write_one_param_output(automl_params or {}, file, "automl parameters")
                self._write_one_param_output(best_hp_params or {}, file, "tuned hyperparameters")
                self._write_one_param_output(
                    best_hardcoded_params or {}, file, "hardcoded parameters"
                )
                self._write_one_param_output(default_params or {}, file, "default parameters")
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
            elif isinstance(value, dict):
                processed_data[key] = str(value)
            else:
                processed_data[key] = str(value)

        if len(processed_data) > 0:
            yaml.safe_dump(processed_data, file, **kwargs)

    def _rebalance_classes(self, train_df):
        import pandas as pd

        resampling_minority_percentage = self.step_config.get(
            "resampling_minority_percentage", _REBALANCING_DEFAULT_RATIO
        )

        df_positive_class = train_df[train_df[self.target_col] == self.positive_class]
        df_negative_class = train_df[train_df[self.target_col] != self.positive_class]

        if len(df_positive_class) > len(df_negative_class):
            df_minority_class, df_majority_class = df_negative_class, df_positive_class
        else:
            df_minority_class, df_majority_class = df_positive_class, df_negative_class

        original_minority_percentage = len(df_minority_class) / len(train_df)
        if original_minority_percentage >= resampling_minority_percentage:
            _logger.info(
                f"Class imbalance of {original_minority_percentage:.2f} "
                f"is better than {resampling_minority_percentage}, no need to rebalance"
            )
            return train_df

        _logger.info(
            f"Detected class imbalance: minority class percentage is "
            f"{original_minority_percentage:.2f}"
        )

        majority_class_target = int(
            len(df_minority_class)
            * (1 - resampling_minority_percentage)
            / resampling_minority_percentage
        )
        df_majority_downsampled = df_majority_class.sample(majority_class_target)
        _logger.info(
            f"After downsampling: minority class percentage is {resampling_minority_percentage:.2f}"
        )
        train_df = pd.concat([df_minority_class, df_majority_downsampled], axis=0).sample(frac=1)

        return train_df
