# pylint: disable=line-too-long


def format_help_string(help_string):
    """
    Formats the specified ``help_string`` to obtain a Mermaid-compatible help string. For example,
    this method replaces quotation marks with their HTML representation.

    :param help_string: The raw help string.
    :return: A Mermaid-compatible help string.
    """
    return help_string.replace('"', "&bsol;#quot;").replace("'", "&bsol;&#39;")


RECIPE_YAML = format_help_string(
    """# recipe.yaml is the main configuration file for the recipe. It defines attributes for each step of the recipe, such as the dataset to use (defined in the the 'ingest' step definition) and the metrics to compute during model training & evaluation (defined in the 'custom_metrics' section, which is used by the 'train' and 'evaluate' steps). recipe.yaml files also support value overrides from profiles (located in the 'profiles' subdirectory of the recipe) using Jinja2 templating syntax. An example recipe.yaml file is displayed below.\n
recipe: "regression/v1"
target_col: "fare_amount"
primary_metric: "root_mean_squared_error"
steps:
  ingest: {{INGEST_CONFIG}}
  split:
    split_ratios: {{SPLIT_RATIOS|default([0.75, 0.125, 0.125])}}
    post_split_filter_method: create_dataset_filter
  transform:
    using: custom
    transformer_method: transformer_fn
  train:
    using: custom
    estimator_method: estimator_fn
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 10
      - metric: mean_absolute_error
        threshold: 50
      - metric: weighted_mean_squared_error
        threshold: 50
  register:
    allow_non_validated_model: false
  ingest_scoring: {{INGEST_SCORING_CONFIG}}
  predict:
    output: {{PREDICT_OUTPUT_CONFIG}}

custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False
"""
)

INGEST_STEP_BASE = """The '{0}' step resolves the dataset specified by the '{1}' section in recipe.yaml and converts it to parquet format, leveraging the custom dataset parsing code defined in `steps/ingest.py` (and referred to by the 'loader_method' attribute of the '{1}' section in recipe.yaml) if necessary. {2} An example recipe.yaml '{1}' configuration is shown below.

{1}:
  location: https://nyc-tlc.s3.amazonaws.com/trip+data/yellow_tripdata_2022-01.parquet
  using: {{{{INGEST_DATA_FORMAT|default('parquet')}}}}
  loader_method: load_file_as_dataframe
"""

INGEST_STEP = format_help_string(
    INGEST_STEP_BASE.format(
        "ingest",
        "steps.ingest",
        "Subsequent steps convert this dataset into training, validation, & test sets and use them to develop a model.",
    )
)

INGEST_USER_CODE = format_help_string(
    """\"\"\"\nsteps/ingest.py defines customizable logic for parsing arbitrary dataset formats (i.e. formats that are not natively parsed by MLflow Recipes) via the `load_file_as_dataframe` function. Note that the Parquet, Delta, and Spark SQL dataset formats are natively parsed by MLflow Recipes, and you do not need to define custom logic for parsing them. An example `load_file_as_dataframe` implementation is displayed below (note that a different function name or module can be specified via the 'loader_method' attribute of the 'data' section in recipe.yaml).\n\"\"\"\n
def load_file_as_dataframe(
    file_path: str,
    file_format: str,
) -> pandas.DataFrame:
    \"\"\"
    Load content from the specified dataset file as a Pandas DataFrame.

    This method is used to load dataset types that are not natively  managed by MLflow Recipes (datasets that are not in Parquet, Delta Table, or Spark SQL Table format). This method is called once for each file in the dataset, and MLflow Recipes automatically combines the resulting DataFrames together.

    :param file_path: The path to the dataset file.
    :param file_format: The file format string, such as "csv".
    :return: A Pandas DataFrame representing the content of the specified file.
    \"\"\"
"""
)

INGESTED_DATA = format_help_string(
    "The ingested parquet representation of the dataset defined in the 'steps.ingest' section of recipe.yaml. Subsequent steps convert this dataset into training, validation, & test sets and use them to develop a model."
)

SPLIT_STEP = format_help_string(
    """The 'split' step splits the ingested dataset produced by the 'ingest' step into a training dataset for model training, a validation dataset for model performance evaluation & tuning, and a test dataset for model performance evaluation. The fraction of records allocated to each dataset is defined by the 'split_ratios' attribute of the 'split' step definition in recipe.yaml. The split step also preprocesses the datasets using logic defined in `steps/split.py` (and referred to by the 'post_split_method' attribute of the 'split' step definition in recipe.yaml). Subsequent steps use these datasets to develop a model and measure its performance. An example recipe.yaml 'split' step definition is shown below.

steps:
  split:
    split_ratios: {{SPLIT_RATIOS|default([0.75, 0.125, 0.125])}}
    post_split_filter_method: create_dataset_filter
"""
)

SPLIT_USER_CODE = format_help_string(
    """\"\"\"\nsteps/split.py defines customizable logic for postprocessing the training, validation, and test datasets prior to model creation via the `create_dataset_filter` function, an example of which is displayed below (note that a different function name or module can be specified via the 'post_split_filter_method' attribute of the 'split' step definition in recipe.yaml).\n\"\"\"\n
This module defines the following routines used by the 'split' step of the recipe:
- ``create_dataset_filter``: Defines customizable logic for filtering the training, validation,
  and test datasets produced by the data splitting procedure. Note that arbitrary transformations
  should go into the transform step.

def create_dataset_filter(dataset: DataFrame) -> Series(bool):
    Mark rows of the split datasets to be additionally filtered. This function will be called on
    the training, validation, and test datasets.
    :param dataset: The {train,validation,test} dataset produced by the data splitting procedure.
    :return: A Series indicating whether each row should be filtered
"""
)

TRAINING_DATA = format_help_string(
    "The training dataset used to train the model. Subsequent steps fit a transformer using this training data, create transformed features, and use the transformed features to fit an estimator, producing a model pipeline consisting of the fitted transformer and the fitted estimator."
)

VALIDATION_DATA = format_help_string(
    "The validation dataset used to evaluate model performance and tune the model pipeline in the train step. It is also used in evaluate step to compute model explanations such as feature importances."
)

TEST_DATA = format_help_string(
    "The test dataset used to evaluate the performance of the model. The 'evaluate' step uses the test dataset to compute a variety of performance metrics."
)

TRANSFORM_STEP = format_help_string(
    """The 'transform' step uses the training dataset produced by 'split' to fit a transformer with the transformation operations defined in `steps/transform.py` (and referred to by the 'transformer_method' attribute of the 'transform' step definition in recipe.yaml). The transformer is then applied to the training dataset and the validation dataset, producing transformed datasets that are used by subsequent steps for estimator training and model performance evaluation. An example recipe.yaml 'transform' step definition is shown below.

steps:
  transform:
    using: custom
    transformer_method: transformer_fn
"""
)

TRANSFORM_USER_CODE = format_help_string(
    """\"\"\"\nsteps/transform.py defines customizable logic for transforming input data during model inference. Transformations are specified via the via the `transformer_fn` function, an example of which is displayed below (note that a different function name or module can be specified via the 'transformer_method' attribute of the 'transform' step definition in recipe.yaml).\n\"\"\"\n
def transformer_fn():
    \"\"\"
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods. The transformer's input and output signatures should be compatible with scikit-learn transformers.
    \"\"\"
"""
)

FITTED_TRANSFORMER = format_help_string(
    "The fitted transformer produced by fitting the transformer defined in `steps/transform.py` on the training dataset output from the 'split' step. The fitted transformer is the first component of the model pipeline. The subsequent 'train' step fits an estimator and creates a model pipeline consisting of the fitted transformer and the fitted estimator."
)

TRANSFORMED_TRAINING_AND_VALIDATION_DATA = format_help_string(
    "1. The transformed training dataset used to fit the estimator component of the model pipeline. Note that training produces a model pipeline consisting of a fitted transformer and a fitted estimator.\n\n2. The validation dataset used to evaluate estimator performance and tune the estimator."
)

TRAIN_STEP = format_help_string(
    """The 'train' step uses the transformed training dataset produced by 'transform' to fit an estimator with the type and parameters defined in `steps/train.py` (and referred to by the 'estimator_method' attribute of the 'train' step definition in recipe.yaml). The estimator is then joined with the fitted transformer output from the 'transform' step to create a model pipeline. Finally, this model pipeline is evaluated against the transformed training and validation datasets to produce performance metrics; custom metrics are computed according to definitions in `steps/custom_metrics.py` and the 'function' attributes of entries in the 'custom' subsection of the 'metrics' section in recipe.yaml. The model pipeline and its associated parameters, performance metrics, and lineage information are logged to MLflow Tracking, producing an MLflow Run. An example recipe.yaml 'train' step definition is shown below, as well as an example custom metric definition.

steps:
  train:
    using: custom
    estimator_method: estimator_fn

custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False
"""
)

TRAIN_USER_CODE = format_help_string(
    """\"\"\"\nsteps/train.py defines customizable logic for specifying your estimator's type and parameters that will be used during training. The estimator type and its parameters are specified via the `estimator_fn` function, an example of which is displayed below (note that a different function name or module can be specified via the 'estimator_method' attribute of the 'train' step definition in recipe.yaml).\n\"\"\"\n
def estimator_fn():
    \"\"\"
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods. The estimator's input and output signatures should be compatible with scikit-learn estimators.
    \"\"\"
"""
)

FITTED_MODEL = format_help_string(
    "The model pipeline produced by fitting the estimator defined in `steps/train.py` on the training dataset and preceding it with the fitted transformer output by the 'transform' step."
)

MLFLOW_RUN = format_help_string(
    "The MLflow Tracking Run containing the model pipeline & its parameters, model performance metrics on the training & validation datasets, and lineage information about the current recipe execution. The downstream 'evaluate' step logs performance metrics and model explanations from the test dataset to this MLflow Run."
)

PREDICTED_TRAINING_DATA = format_help_string(
    "The predicted training dataset that is obtained by predicted training data using the fitted model."
)

CUSTOM_METRICS_USER_CODE = format_help_string(
    """\"\"\"\nsteps/custom_metrics.py defines customizable logic for specifying custom metrics to compute during model training and evaluation. Custom metric functions defined in `steps/custom_metrics.py` are referenced by the 'function' attributes of entries in the 'custom' subsection of the 'metrics' section in recipe.yaml. For example:

custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False

An example custom_metrics.py file is displayed below.
\"\"\"\
def weighted_mean_squared_error(
    eval_df: pandas.DataFrame,
    builtin_metrics: Dict[str, int],
) -> Dict[str, int]:
    \"\"\"
    Computes the weighted mean squared error (MSE) metric.

    :param eval_df: A Pandas DataFrame containing the following columns:

                    - ``"prediction"``: Predictions produced by submitting input data to the model.
                    - ``"target"``: Ground truth values corresponding to the input data.

    :param builtin_metrics: A dictionary containing the built-in metrics that are calculated automatically during model evaluation. The keys are the names of the metrics and the values are the scalar values of the metrics. For more information, see https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate.
    :return: A single-entry dictionary containing the MSE metric. The key is the metric names and the value is the scalar metric value. Note that custom metric functions can return dictionaries with multiple metric entries as well.
    \"\"\"
"""
)

EVALUATE_STEP = format_help_string(
    """The 'evaluate' step evaluates the model pipeline produced by the 'train' step on the test dataset output from the 'split' step, producing performance metrics and model explanations. Performance metrics are compared against configured thresholds to compute a 'model_validation_status', which indicates whether or not a model is good enough to be registered to the MLflow Model Registry by the subsequent 'register' step. Custom performance metrics are computed according to definitions in `steps/custom_metrics.py` and the 'function' attributes of entries in the 'custom' subsection of the 'metrics' section in recipe.yaml. Model performance thresholds are defined in the 'validation_criteria' section of the 'evaluate' step definition in recipe.yaml. Model performance metrics and explanations are logged to MLflow Tracking using the same MLflow Run produced by the 'train' step. An example recipe.yaml 'evaluate' step definition is shown below, as well as an example custom metric definition.

evaluate:
  validation_criteria:
    - metric: root_mean_squared_error
      threshold: 10
    - metric: weighted_mean_squared_error
      threshold: 20

custom_metrics:
  - name: weighted_mean_squared_error
    function: weighted_mean_squared_error
    greater_is_better: False
"""
)

MODEL_VALIDATION_STATUS = format_help_string(
    """Boolean status indicating whether or not the model meets the performance criteria for registration to the MLflow Model Registry. Performance criteria are defined in the 'validation_criteria' section of the 'evaluate' step definition in recipe.yaml, as shown in the example below. The subsequent 'register' step checks the model validation status, and, if it is 'VALIDATED', creates a new model version in the Model Registry corresponding to the trained model pipeline.

evaluate:
  validation_criteria:
    - metric: root_mean_squared_error
      threshold: 10
    - metric: mean_absolute_error
      threshold: 50
    - metric: weighted_mean_squared_error
      threshold: 20
"""
)

REGISTER_STEP = format_help_string(
    """The 'register' step checks the 'model_validation_status' output of the preceding 'evaluate' step and, if model validation was successful (as indicated by the 'VALIDATED' status), registers the model pipeline produced by the 'train' step to the MLflow Model Registry. If the 'model_validation_status' does not indicate that the model passed validation checks (i.e. its value is 'REJECTED'), the model pipeline is not registered to the MLflow Model Registry. This validation status check can be disabled by specifying 'allow_non_validated_model: true' in the 'register' step definition of recipe.yaml, in which case the model pipeline is always registered with the MLflow Model Registry when the 'register' step is executed. If the model pipeline is registered to the MLflow Model Registry, a 'registered_model_version' is produced containing the model name (as configured by the 'model_name' attribute of the 'register' step definition in recipe.yaml) and the model version. An example recipe.yaml 'register' step definition is shown below.

register:
  allow_non_validated_model: true
"""
)

INGEST_SCORING_STEP = format_help_string(
    INGEST_STEP_BASE.format(
        "ingest_scoring", "data_scoring", "Subsequent steps score this dataset for batch scoring."
    )
)

INGESTED_SCORING_DATA = format_help_string(
    "The ingested parquet representation of the dataset defined in the 'data_scoring' section of recipe.yaml. Subsequent steps score this dataset for batch scoring."
)

PREDICT_STEP = format_help_string(
    """The 'predict' step uses the model registered by the 'register' step to score the ingested dataset produced by the 'ingest_scoring' step and writes the resulting dataset to the specified output format and location. To get model for scoring, it reads the register step model version artifact. If the register step was cleared, it uses the latest version of the registered model specified by the `model_name` attribute of the recipe.yaml 'register' step definition. To fix a specific model for use in the 'predict' step, provide its model URI as the 'model_uri' attribute of the recipe.yaml 'predict' step definition. An example recipe.yaml 'predict' step definition is shown below.
steps:
  predict:
    model_uri: "models:/taxi_fare_regressor/Production" # optional
    using: {{OUTPUT_DATA_FORMAT|default('parquet')}}
    location: "{{OUTPUT_DATA_LOCATION}}"
"""
)

SCORED_DATA = format_help_string(
    "The dataset produced by scoring the ingested dataset generated by the 'ingest_scoring' step with the model specified by the the 'predict' step."
)

REGISTERED_MODEL_VERSION = format_help_string(
    "The Model Version in the MLflow Model Registry corresponding to the trained model. A Model Version is produced if the trained model meets the defined performance criteria for model registration or if `allow_non_validated_model: true` is specified in the 'register' step definition of recipe.yaml"
)
