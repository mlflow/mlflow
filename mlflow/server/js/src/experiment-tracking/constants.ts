export const COLUMN_TYPES = {
  ATTRIBUTES: 'attributes',
  PARAMS: 'params',
  METRICS: 'metrics',
  TAGS: 'tags',
};
export const MLMODEL_FILE_NAME = 'MLmodel';
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
};

export const ATTRIBUTE_COLUMN_SORT_LABEL = {
  DATE: 'Created',
  USER: 'User',
  RUN_NAME: 'Run Name',
  SOURCE: 'Source',
  VERSION: 'Version',
};

export const ATTRIBUTE_COLUMN_SORT_KEY = {
  DATE: 'attributes.start_time',
  USER: 'tags.`mlflow.user`',
  RUN_NAME: 'tags.`mlflow.runName`',
  SOURCE: 'tags.`mlflow.source.name`',
  VERSION: 'tags.`mlflow.source.git.commit`',
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
export const DEFAULT_EXPANDED_VALUE = false;
export const DEFAULT_CATEGORIZED_UNCHECKED_KEYS = {
  [COLUMN_TYPES.ATTRIBUTES]: [],
  [COLUMN_TYPES.PARAMS]: [],
  [COLUMN_TYPES.METRICS]: [],
  [COLUMN_TYPES.TAGS]: [],
};
export const DEFAULT_DIFF_SWITCH_SELECTED = false;
export const DEFAULT_LIFECYCLE_FILTER = LIFECYCLE_FILTER.ACTIVE;
export const DEFAULT_MODEL_VERSION_FILTER = MODEL_VERSION_FILTER.ALL_RUNS;

export const PAGINATION_DEFAULT_STATE = {
  nextPageToken: null,
  numRunsFromLatestSearch: null, // number of runs returned from the most recent search request
  loadingMore: false,
};

export const MAX_DETECT_NEW_RUNS_RESULTS = 26; // so the refresh button badge can be 25+
export const POLL_INTERVAL = 15000;

export const AUTOML_TAG_PREFIX = '_databricks_automl';
export const AUTOML_EVALUATION_METRIC_TAG = `${AUTOML_TAG_PREFIX}.evaluation_metric`;

export const MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME = 'mlflow.experiment.primaryMetric.name';
export const MLFLOW_EXPERIMENT_PRIMARY_METRIC_GREATER_IS_BETTER = 'mlflow.experiment.primaryMetric.greaterIsBetter';
export const MLFLOW_RUN_DATASET_CONTEXT_TAG = 'mlflow.data.context';
export const MLFLOW_LOGGED_ARTIFACTS_TAG = 'mlflow.loggedArtifacts';
export const EXPERIMENT_PAGE_FEEDBACK_URL = 'https://github.com/mlflow/mlflow/issues/6348';

export const MLFLOW_RUN_TYPE_TAG = 'mlflow.runType';
export const MLFLOW_RUN_SOURCE_TYPE_TAG = 'mlflow.runSourceType';
export const MLFLOW_RUN_TYPE_VALUE_EVALUATION = 'evaluation';

export const MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG = 'mlflow.source.git.branch';

export enum MLflowRunSourceType {
  PROMPT_ENGINEERING = 'PROMPT_ENGINEERING',
}

export const MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME = 'eval_results_table.json';

export enum RunPageTabName {
  OVERVIEW = 'overview',
  MODEL_METRIC_CHARTS = 'model-metrics',
  SYSTEM_METRIC_CHARTS = 'system-metrics',
  ARTIFACTS = 'artifacts',
}

export const MLFLOW_SYSTEM_METRIC_PREFIX = 'system/';

export const MLFLOW_SYSTEM_METRIC_NAME = 'System metrics';

export const MLFLOW_MODEL_METRIC_PREFIX = '';

export const MLFLOW_MODEL_METRIC_NAME = 'Model metrics';

export const EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY = 'viewStateShareKey';
export const EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX = 'mlflow.sharedViewState.';
