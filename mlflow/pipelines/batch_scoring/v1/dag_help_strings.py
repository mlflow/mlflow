# pylint: disable=line-too-long


def format_help_string(help_string):
    """
    Formats the specified ``help_string`` to obtain a Mermaid-compatible help string. For example,
    this method replaces quotation marks with their HTML representation.

    :param help_string: The raw help string.
    :return: A Mermaid-compatible help string.
    """
    return help_string.replace('"', "&bsol;#quot;").replace("'", "&bsol;&#39;")


PIPELINE_YAML = format_help_string(
    """# pipeline.yaml is the main configuration file for the pipeline. It defines attributes for each step of the batch scoring pipeline, such as the dataset to use (defined in the 'data' section of the 'ingest' step definition) and the metrics to compute during model training & evaluation (defined in the 'metrics' section, which is used by the 'train' and 'evaluate' steps). pipeline.yaml files also support value overrides from profiles (located in the 'profiles' subdirectory of the pipeline) using Jinja2 templating syntax. An example pipeline.yaml file is displayed below.\n
template: "batch_scoring/v1"
# Specifies the dataset to use for model development
data:
  location: {{INGEST_DATA_LOCATION}}
  format: {{INGEST_DATA_FORMAT|default('parquet')}}
  custom_loader_method: steps.ingest.load_file_as_dataframe
steps:
  split:
    split_ratios: {{SPLIT_RATIOS|default([0.75, 0.125, 0.125])}}
    post_split_method: steps.split.process_splits
  transform:
    transformer_method: steps.transform.transformer_fn
  train:
    estimator_method: steps.train.estimator_fn
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 10
  register:
    model_name: "taxi_fare_regressor"
    allow_non_validated_model: true
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
  primary: "root_mean_squared_error"
"""
)

INGEST_STEP = format_help_string(
    """The 'ingest' step resolves the dataset specified by the 'data' section in pipeline.yaml and converts it to parquet format, leveraging the custom dataset parsing code defined in `steps/ingest.py` (and referred to by the 'custom_loader_method' attribute of the 'data' section in pipeline.yaml) if necessary. Subsequent steps convert this dataset into training, validation, & test sets and use them to develop a model. An example pipeline.yaml 'data' configuration is shown below.

data:
  location: https://nyc-tlc.s3.amazonaws.com/trip+data/yellow_tripdata_2022-01.parquet
  format: {{INGEST_DATA_FORMAT|default('parquet')}}
  custom_loader_method: steps.ingest.load_file_as_dataframe
"""
)

INGEST_USER_CODE = format_help_string(
    """\"\"\"\nsteps/ingest.py defines customizable logic for parsing arbitrary dataset formats (i.e. formats that are not natively parsed by MLflow Pipelines) via the `load_file_as_dataframe` function. Note that the Parquet, Delta, and Spark SQL dataset formats are natively parsed by MLflow Pipelines, and you do not need to define custom logic for parsing them. An example `load_file_as_dataframe` implementation is displayed below (note that a different function name or module can be specified via the 'custom_loader_method' attribute of the 'data' section in pipeline.yaml).\n\"\"\"\n
import pandas

def load_file_as_dataframe(
    file_path: str,
    file_format: str,
) -> pandas.DataFrame:
    \"\"\"
    Load content from the specified dataset file as a Pandas DataFrame.

    This method is used to load dataset types that are not natively  managed by MLflow Pipelines (datasets that are not in Parquet, Delta Table, or Spark SQL Table format). This method is called once for each file in the dataset, and MLflow Pipelines automatically combines the resulting DataFrames together.

    :param file_path: The path to the dataset file.
    :param file_format: The file format string, such as "csv".
    :return: A Pandas DataFrame representing the content of the specified file.
    \"\"\"

    if file_format == "csv":
        return pandas.read_csv(file_path, index_col=0)
    else:
        raise NotImplementedError
"""
)

INGESTED_DATA = format_help_string(
    "The ingested parquet representation of the dataset defined in the 'data' section of pipeline.yaml. Subsequent steps convert this dataset into training, validation, & test sets and use them to develop a model."
)

DATA_CLEAN_STEP = format_help_string(
    """The 'data_clean' step applies the user generated data preprocessing method against the ingested dataset produced by the 'ingest' step into a cleaned dataset for batch scoring. An example pipeline.yaml 'data_clean' step definition is shown below.

steps:
  data_clean:
    post_split_method: steps.data_clean.clean
"""
)

DATA_CLEAN_USER_CODE = format_help_string(
    """\"\"\"\nsteps/data_clean.py defines customizable logic for preprocessing the ingested dataset prior to batch scoring.\n\"\"\"\n
import pandas

def clean(
    ingested_df: pandas.DataFrame
) -> pandas.DataFrame:
    \"\"\"
    Perform additional processing on the ingested dataset.

    :param ingested_df: The ingested dataset.
    :return: The processed dataset.
    \"\"\"

    return ingested_df.dropna()
"""
)

PREDICT_STEP = format_help_string(
    """The 'predict' step applies the specified model against the cleaned dataset produced by the 'data_clean' step. An example pipeline.yaml 'predict' step definition is shown below.

steps:
  predict:
    model_name: "taxi_fare_regressor"
    version: 11
    output_path: "{{OUTPUT_DATA_PATH}}"
    output_format: {{OUTPUT_DATA_FORMAT|default('parquet')}}
"""
)
