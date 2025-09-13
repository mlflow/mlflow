export const COLUMN_TYPES = {
  ATTRIBUTES: 'attributes',
  PARAMS: 'params',
  METRICS: 'metrics',
  TAGS: 'tags',
};
export const MLMODEL_FILE_NAME = 'MLmodel';
export const SERVING_INPUT_FILE_NAME = 'serving_input_payload.json';
export const ONE_MB = 1024 * 1024;

export const ATTRIBUTE_COLUMN_LABELS = {
  DATE: 'Created',
  EXPERIMENT_NAME: 'Experiment Name',
  DURATION: 'Duration',
  USER: 'User',
  RUN_NAME: 'Run Name',
  SOURCE: 'Source',
  VERSION: 'Version',
  MODELS: 'Models',
  DATASET: 'Dataset',
  DESCRIPTION: 'Description',
};

export const ATTRIBUTE_COLUMN_SORT_LABEL = {
  DATE: 'Created',
  USER: 'User',
  RUN_NAME: 'Run Name',
  SOURCE: 'Source',
  VERSION: 'Version',
  DESCRIPTION: 'Description',
};

export const ATTRIBUTE_COLUMN_SORT_KEY = {
  DATE: 'attributes.start_time',
  USER: 'tags.`mlflow.user`',
  RUN_NAME: 'tags.`mlflow.runName`',
  SOURCE: 'tags.`mlflow.source.name`',
  VERSION: 'tags.`mlflow.source.git.commit`',
  DESCRIPTION: 'tags.`mlflow.note.content`',
};

export const COLUMN_SORT_BY_ASC = 'ASCENDING';
export const COLUMN_SORT_BY_DESC = 'DESCENDING';
export const SORT_DELIMITER_SYMBOL = '***';

export enum LIFECYCLE_FILTER {
  ACTIVE = 'Active',
  DELETED = 'Deleted',
}

export enum MODEL_VERSION_FILTER {
  WITH_MODEL_VERSIONS = 'With Model Versions',
  WTIHOUT_MODEL_VERSIONS = 'Without Model Versions',
  ALL_RUNS = 'All Runs',
}

export const DEFAULT_ORDER_BY_KEY = ATTRIBUTE_COLUMN_SORT_KEY.DATE;
export const DEFAULT_ORDER_BY_ASC = false;
export const DEFAULT_START_TIME = 'ALL';
export const DEFAULT_CATEGORIZED_UNCHECKED_KEYS = {
  [COLUMN_TYPES.ATTRIBUTES]: [],
  [COLUMN_TYPES.PARAMS]: [],
  [COLUMN_TYPES.METRICS]: [],
  [COLUMN_TYPES.TAGS]: [],
};
export const DEFAULT_DIFF_SWITCH_SELECTED = false;
export const DEFAULT_LIFECYCLE_FILTER = LIFECYCLE_FILTER.ACTIVE;
export const DEFAULT_MODEL_VERSION_FILTER = MODEL_VERSION_FILTER.ALL_RUNS;

export const AUTOML_TAG_PREFIX = '_databricks_automl';
export const AUTOML_EVALUATION_METRIC_TAG = `${AUTOML_TAG_PREFIX}.evaluation_metric`;
export const AUTOML_PROBLEM_TYPE_TAG = `${AUTOML_TAG_PREFIX}.problem_type`;

export const AUTOML_TEST_EVALUATION_METRIC_PREFIX = 'test_';

export const MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME = 'mlflow.experiment.primaryMetric.name';
export const MLFLOW_RUN_DATASET_CONTEXT_TAG = 'mlflow.data.context';
export const MLFLOW_LOGGED_ARTIFACTS_TAG = 'mlflow.loggedArtifacts';
export const MLFLOW_LINKED_PROMPTS_TAG = 'mlflow.linkedPrompts';
export const MLFLOW_LOGGED_MODEL_USER_TAG = 'mlflow.user';
export const EXPERIMENT_PAGE_FEEDBACK_URL = 'https://github.com/mlflow/mlflow/issues/6348';

export const MLFLOW_RUN_TYPE_TAG = 'mlflow.runType';
export const MLFLOW_RUN_COLOR_TAG = 'mlflow.runColor';
export const MLFLOW_RUN_SOURCE_TYPE_TAG = 'mlflow.runSourceType';
export const MLFLOW_RUN_TYPE_VALUE_EVALUATION = 'evaluation';

export const MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG = 'mlflow.source.git.branch';
export const MLFLOW_PROMPT_VERSION_COUNT_TAG = 'PromptVersionCount';

export const MONITORING_BETA_EXPIRATION_DATE = new Date('2030-06-24T00:00:00');

export enum MLflowRunSourceType {
  PROMPT_ENGINEERING = 'PROMPT_ENGINEERING',
}

export const MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME = 'eval_results_table.json';

export enum RunPageTabName {
  OVERVIEW = 'overview',
  TRACES = 'traces',
  MODEL_METRIC_CHARTS = 'model-metrics',
  SYSTEM_METRIC_CHARTS = 'system-metrics',
  ARTIFACTS = 'artifacts',
  EVALUATIONS = 'evaluations',
}

export const MLFLOW_SYSTEM_METRIC_PREFIX = 'system/';

export const MLFLOW_SYSTEM_METRIC_NAME = 'System metrics';

export const MLFLOW_MODEL_METRIC_PREFIX = '';

export const MLFLOW_MODEL_METRIC_NAME = 'Model metrics';

export const EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY = 'viewStateShareKey';
export const EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX = 'mlflow.sharedViewState.';

export const MLFLOW_LOGGED_IMAGE_ARTIFACTS_PATH = 'images';
export const IMAGE_FILE_EXTENSION = 'png';
export const IMAGE_COMPRESSED_FILE_EXTENSION = 'webp';
export const EXPERIMENT_RUNS_IMAGE_AUTO_REFRESH_INTERVAL = 30000;
export const DEFAULT_IMAGE_GRID_CHART_NAME = 'Image grid';

export const LOG_TABLE_IMAGE_COLUMN_TYPE = 'image';
export const LOG_IMAGE_TAG_INDICATOR = 'mlflow.loggedImages';
export const NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE = 10;

/**
 * This is a timestamp that is used as the base for relative time calculations.
 * It is the number of milliseconds since the Unix epoch. It is used to
 * create a timestamp that is 00:00:00.000 and works with plotly. The date
 * doesn't matter because it will be hidden in relative time displays.
 */
export const EPOCH_RELATIVE_TIME = 28800000;
export const LINE_CHART_RELATIVE_TIME_THRESHOLD = 1000 * 60 * 60 * 24; // 1 day
export const HOUR_IN_MILLISECONDS = 1000 * 60 * 60; // 1 hour

export enum ExperimentPageTabName {
  Runs = 'runs',
  Traces = 'traces',
  Models = 'models',
  EvaluationMonitoring = 'evaluation-monitoring',
  Scorers = 'scorers',
  EvaluationRuns = 'evaluation-runs',
  Datasets = 'datasets',
  LabelingSessions = 'labeling-sessions',
  LabelingSchemas = 'label-schemas',
  Prompts = 'prompts',
}

export const getMlflow3DocsLink = () => {
  return 'https://docs.databricks.com/aws/en/mlflow/mlflow-3-install';
};

export enum ExperimentKind {
  GENAI_DEVELOPMENT = 'genai_development',
  CUSTOM_MODEL_DEVELOPMENT = 'custom_model_development',
  GENAI_DEVELOPMENT_INFERRED = 'genai_development_inferred',
  CUSTOM_MODEL_DEVELOPMENT_INFERRED = 'custom_model_development_inferred',
  NO_INFERRED_TYPE = 'no_inferred_type',
  FINETUNING = 'finetuning',
  FORECASTING = 'forecasting',
  CLASSIFICATION = 'classification',
  REGRESSION = 'regression',
  AUTOML = 'automl',
  EMPTY = '',
}

export const CREATE_NEW_VERSION_QUERY_PARAM = 'create_new_version';
