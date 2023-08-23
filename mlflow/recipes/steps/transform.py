import importlib
import logging
import os
import sys
import time

import cloudpickle
from packaging.version import Version

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.artifacts import DataframeArtifact, TransformerArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.recipes.utils.tracking import TrackingConfig, get_recipe_tracking_config

_logger = logging.getLogger(__name__)

_USER_DEFINED_TRANSFORM_STEP_MODULE = "steps.transform"


def _generate_feature_names(num_features):
    max_length = len(str(num_features))
    return ["f_" + str(i).zfill(max_length) for i in range(num_features)]


def _get_output_feature_names(transformer, num_features, input_features):
    import sklearn

    # `get_feature_names_out` was introduced in scikit-learn 1.0.0.
    if Version(sklearn.__version__) < Version("1.0.0"):
        return _generate_feature_names(num_features)

    try:
        # `get_feature_names_out` fails if `transformer` contains a transformer that doesn't
        # implement `get_feature_names_out`. For example, `FunctionTransformer` only implements
        # `get_feature_names_out` when it's instantiated with `feature_names_out`.
        # In scikit-learn >= 1.1.0, all transformers implement `get_feature_names_out`.
        # In scikit-learn == 1.0.*, some transformers implement `get_feature_names_out`.
        return transformer.get_feature_names_out(input_features)
    except Exception as e:
        _logger.warning(
            f"Failed to get output feature names with `get_feature_names_out`: {e}. "
            "Falling back to using auto-generated feature names."
        )
        return _generate_feature_names(num_features)


def _validate_user_code_output(transformer_fn):
    transformer = transformer_fn()
    if transformer is not None and not (hasattr(transformer, "fit") and callable(transformer.fit)):
        raise MlflowException(
            message="The transformer provided doesn't have a fit method."
        ) from None

    if transformer is not None and not (
        hasattr(transformer, "transform") and callable(transformer.transform)
    ):
        raise MlflowException(
            message="The transformer provided doesn't have a transform method."
        ) from None

    return transformer


class TransformStep(BaseStep):
    def __init__(self, step_config, recipe_root):  # pylint: disable=useless-super-delegation
        super().__init__(step_config, recipe_root)
        self.tracking_config = TrackingConfig.from_dict(self.step_config)

    def _validate_and_apply_step_config(self):
        self.target_col = self.step_config.get("target_col")
        self.positive_class = self.step_config.get("positive_class")
        if self.target_col is None:
            raise MlflowException(
                "Missing target_col config in recipe config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if "using" in self.step_config:
            if self.step_config["using"] not in ["custom"]:
                raise MlflowException(
                    f"Invalid transform step configuration value {self.step_config['using']} for "
                    f"key 'using'. Supported values are: ['custom']",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            self.step_config["using"] = "custom"
        self.run_end_time = None
        self.execution_duration = None
        self.skip_data_profiling = self.step_config.get("skip_data_profiling", False)

    def _run(self, output_directory):
        import pandas as pd

        run_start_time = time.time()

        train_data_path = get_step_output_path(
            recipe_root_path=self.recipe_root,
            step_name="split",
            relative_path="train.parquet",
        )
        train_df = pd.read_parquet(train_data_path)
        validate_classification_config(self.task, self.positive_class, train_df, self.target_col)

        validation_data_path = get_step_output_path(
            recipe_root_path=self.recipe_root,
            step_name="split",
            relative_path="validation.parquet",
        )
        validation_df = pd.read_parquet(validation_data_path)

        sys.path.append(self.recipe_root)

        def get_identity_transformer():
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import FunctionTransformer

            return Pipeline(steps=[("identity", FunctionTransformer())])

        if "transformer_method" not in self.step_config and self.step_config["using"] == "custom":
            raise MlflowException(
                "Missing 'transformer_method' configuration in the transform step, "
                "which is using 'custom'.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        method_config = self.step_config.get("transformer_method")
        transformer = None
        if method_config and self.step_config["using"] == "custom":
            transformer_fn = getattr(
                importlib.import_module(_USER_DEFINED_TRANSFORM_STEP_MODULE), method_config
            )
            transformer = _validate_user_code_output(transformer_fn)
        transformer = transformer if transformer else get_identity_transformer()
        transformer.fit(train_df.drop(columns=[self.target_col]), train_df[self.target_col])

        def transform_dataset(dataset):
            features = dataset.drop(columns=[self.target_col])
            transformed_features = transformer.transform(features)
            if not isinstance(transformed_features, pd.DataFrame):
                num_features = transformed_features.shape[1]
                columns = _get_output_feature_names(transformer, num_features, features.columns)
                transformed_features = pd.DataFrame(transformed_features, columns=columns)
            transformed_features[self.target_col] = dataset[self.target_col].values
            return transformed_features

        train_transformed = transform_dataset(train_df)
        validation_transformed = transform_dataset(validation_df)

        with open(os.path.join(output_directory, "transformer.pkl"), "wb") as f:
            cloudpickle.dump(transformer, f)

        train_transformed.to_parquet(
            os.path.join(output_directory, "transformed_training_data.parquet")
        )
        validation_transformed.to_parquet(
            os.path.join(output_directory, "transformed_validation_data.parquet")
        )

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time

        return self._build_profiles_and_card(train_df, train_transformed, transformer)

    def _build_profiles_and_card(self, train_df, train_transformed, transformer) -> BaseCard:
        # Build card
        card = BaseCard(self.recipe_name, self.name)

        if not self.skip_data_profiling:
            # Tab 1: build profiles for train_transformed
            train_transformed_profile = get_pandas_data_profiles(
                [["Profile of Train Transformed Dataset", train_transformed]]
            )
            card.add_tab("Data Profile (Train Transformed)", "{{PROFILE}}").add_pandas_profile(
                "PROFILE", train_transformed_profile
            )

        # Tab 3: transformer diagram
        from sklearn import set_config
        from sklearn.utils import estimator_html_repr

        set_config(display="diagram")
        transformer_repr = estimator_html_repr(transformer)
        card.add_tab("Transformer", "{{TRANSFORMER}}").add_html("TRANSFORMER", transformer_repr)

        # Tab 4: transformer input schema
        card.add_tab("Input Schema", "{{INPUT_SCHEMA}}").add_html(
            "INPUT_SCHEMA",
            BaseCard.render_table({"Name": n, "Type": t} for n, t in train_df.dtypes.items()),
        )

        # Tab 5: transformer output schema
        try:
            card.add_tab("Output Schema", "{{OUTPUT_SCHEMA}}").add_html(
                "OUTPUT_SCHEMA",
                BaseCard.render_table(
                    {"Name": n, "Type": t} for n, t in train_transformed.dtypes.items()
                ),
            )
        except Exception as e:
            card.add_tab("Output Schema", "{{OUTPUT_SCHEMA}}").add_html(
                "OUTPUT_SCHEMA", f"Failed to extract transformer schema. Error: {e}"
            )

        # Tab 6: transformer output data preview
        card.add_tab("Data Preview", "{{DATA_PREVIEW}}").add_html(
            "DATA_PREVIEW", BaseCard.render_table(train_transformed.head())
        )

        # Tab 7: run summary
        (
            card.add_tab(
                "Run Summary",
                """
                {{ EXE_DURATION }}
                {{ LAST_UPDATE_TIME }}
                """,
            )
        )

        return card

    @classmethod
    def from_recipe_config(cls, recipe_config, recipe_root):
        step_config = {}
        if recipe_config.get("steps", {}).get("transform", {}) is not None:
            step_config.update(recipe_config.get("steps", {}).get("transform", {}))
        step_config["target_col"] = recipe_config.get("target_col")
        step_config["recipe"] = recipe_config.get("recipe", "regression/v1")
        if "positive_class" in recipe_config:
            step_config["positive_class"] = recipe_config.get("positive_class")
        step_config.update(
            get_recipe_tracking_config(
                recipe_root_path=recipe_root,
                recipe_config=recipe_config,
            ).to_dict()
        )
        return cls(step_config, recipe_root)

    @property
    def name(self):
        return "transform"

    def get_artifacts(self):
        return [
            DataframeArtifact(
                "transformed_training_data",
                self.recipe_root,
                self.name,
                "transformed_training_data.parquet",
            ),
            DataframeArtifact(
                "transformed_validation_data",
                self.recipe_root,
                self.name,
                "transformed_validation_data.parquet",
            ),
            TransformerArtifact(
                "transformer", self.recipe_root, self.name, self.tracking_config.tracking_uri
            ),
        ]

    def step_class(self):
        return StepClass.TRAINING
