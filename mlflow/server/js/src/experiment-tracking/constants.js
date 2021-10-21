export const COLUMN_TYPES = {
  ATTRIBUTES: 'attributes',
  PARAMS: 'params',
  METRICS: 'metrics',
  TAGS: 'tags',
};
export const MLMODEL_FILE_NAME = 'MLmodel';

export const ATTRIBUTE_COLUMN_LABELS = {
  DATE: 'Start Time',
  DURATION: 'Duration',
  USER: 'User',
  RUN_NAME: 'Run Name',
  SOURCE: 'Source',
  VERSION: 'Version',
  MODELS: 'Models',
};

export const ATTRIBUTE_COLUMN_SORT_LABEL = {
  DATE: 'Start Time',
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

export const LIFECYCLE_FILTER = { ACTIVE: 'Active', DELETED: 'Deleted' };
export const MODEL_VERSION_FILTER = {
  WITH_MODEL_VERSIONS: 'With Model Versions',
  WTIHOUT_MODEL_VERSIONS: 'Without Model Versions',
  ALL_RUNS: 'All Runs',
};

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

export const PAGINATION_DEFAULT_STATE = {
  nextPageToken: null,
  numRunsFromLatestSearch: null, // number of runs returned from the most recent search request
  loadingMore: false,
};

export const MAX_DETECT_NEW_RUNS_RESULTS = 26; // so the refresh button badge can be 25+
export const DETECT_NEW_RUNS_INTERVAL = 15000;
