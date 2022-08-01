import importlib
import logging
import os
import sys
import time

import cloudpickle
from packaging.version import Version

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.step import get_pandas_data_profile
from mlflow.pipelines.utils.tracking import get_pipeline_tracking_config

_logger = logging.getLogger(__name__)


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


class TransformStep(BaseStep):
    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
        self.run_end_time = None
        self.execution_duration = None
        self.target_col = self.step_config.get("target_col")
        (self.transformer_module_name, self.transformer_method_name,) = self.step_config[
            "transformer_method"
        ].rsplit(".", 1)

    def _run(self, output_directory):
        import pandas as pd

        run_start_time = time.time()

        train_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="split",
            relative_path="train.parquet",
        )
        train_df = pd.read_parquet(train_data_path)

        validation_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="split",
            relative_path="validation.parquet",
        )
        validation_df = pd.read_parquet(validation_data_path)

        sys.path.append(self.pipeline_root)

        def get_identity_transformer():
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import FunctionTransformer

            return Pipeline(steps=[("identity", FunctionTransformer())])

        transformer_fn = getattr(
            importlib.import_module(self.transformer_module_name), self.transformer_method_name
        )
        transformer = transformer_fn()
        transformer = transformer if transformer else get_identity_transformer()
        transformer.fit(train_df.drop(columns=[self.target_col]))

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

        return self._build_profiles_and_card(
            train_df, train_transformed, validation_transformed, transformer
        )

    def _build_profiles_and_card(
        self, train_df, train_transformed, validation_transformed, transformer
    ) -> BaseCard:
        # Build card
        card = BaseCard(self.pipeline_name, self.name)

        # Tab 1 and 2: build profiles for train_transformed, validation_transformed
        train_transformed_profile = get_pandas_data_profile(
            train_transformed,
            "Profile of Train Transformed Dataset",
        )
        validation_transformed_profile = get_pandas_data_profile(
            validation_transformed,
            "Profile of Validation Transformed Dataset",
        )
        card.add_tab("Data Profile (Train Transformed)", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", train_transformed_profile
        )
        card.add_tab("Data Profile (Validation Transformed)", "{{PROFILE}}").add_pandas_profile(
            "PROFILE", validation_transformed_profile
        )

        # Tab 3: transformer diagram
        from sklearn.utils import estimator_html_repr
        from sklearn import set_config

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

        # Tab 6: run summary
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
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["transform"]
            step_config.update(
                get_pipeline_tracking_config(
                    pipeline_root_path=pipeline_root,
                    pipeline_config=pipeline_config,
                ).to_dict()
            )
        except KeyError:
            raise MlflowException(
                "Config for transform step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        step_config["target_col"] = pipeline_config.get("target_col")
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "transform"
