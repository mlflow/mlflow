import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { type MessageDescriptor, defineMessage } from 'react-intl';
import { isNil } from 'lodash';

const TRACE_METADATA_FIELD_RUN_ID = 'mlflow.sourceRun';
const TRACE_METADATA_FIELD_TOTAL_TOKENS = 'total_tokens';
const TRACE_METADATA_FIELD_INPUTS = 'mlflow.traceInputs';
const TRACE_METADATA_FIELD_OUTPUTS = 'mlflow.traceOutputs';
export const TRACE_TAG_NAME_TRACE_NAME = 'mlflow.traceName';

// Truncation limit for tracing metadata, taken from:
// https://github.com/mlflow/mlflow/blob/2b457f2b46fc135a3fba77aefafe2319a899fc08/mlflow/tracing/constant.py#L23
const MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS = 250;

const getTraceMetadataField = (traceInfo: ModelTraceInfo, field: string) => {
  return traceInfo.request_metadata?.find(({ key }) => key === field)?.value;
};

export const isTraceMetadataPossiblyTruncated = (traceMetadata: string) => {
  return traceMetadata.length >= MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS;
};

export const getTraceInfoRunId = (traceInfo: ModelTraceInfo) =>
  getTraceMetadataField(traceInfo, TRACE_METADATA_FIELD_RUN_ID);

export const getTraceInfoTotalTokens = (traceInfo: ModelTraceInfo) =>
  getTraceMetadataField(traceInfo, TRACE_METADATA_FIELD_TOTAL_TOKENS);

export const getTraceInfoInputs = (traceInfo: ModelTraceInfo) => {
  const inputs = getTraceMetadataField(traceInfo, TRACE_METADATA_FIELD_INPUTS);
  if (isNil(inputs)) {
    return undefined;
  }
  try {
    return JSON.stringify(JSON.parse(inputs)); // unescape non-ascii characters
  } catch (e) {
    return inputs;
  }
};

export const getTraceInfoOutputs = (traceInfo: ModelTraceInfo) => {
  const outputs = getTraceMetadataField(traceInfo, TRACE_METADATA_FIELD_OUTPUTS);
  if (isNil(outputs)) {
    return undefined;
  }
  try {
    return JSON.stringify(JSON.parse(outputs)); // unescape non-ascii characters
  } catch (e) {
    return outputs;
  }
};

export const getTraceTagValue = (traceInfo: ModelTraceInfo, tagName: string) => {
  if (Array.isArray(traceInfo.tags)) {
    return traceInfo.tags?.find(({ key }) => key === tagName)?.value;
  }

  return traceInfo.tags?.[tagName];
};

export const getTraceDisplayName = (traceInfo: ModelTraceInfo) => {
  return getTraceTagValue(traceInfo, TRACE_TAG_NAME_TRACE_NAME) || traceInfo.request_id;
};

export const EXPERIMENT_TRACES_SORTABLE_COLUMNS = ['timestamp_ms'];

// defining a separate const for this column as
// we don't users to be able to control its visibility
export const TRACE_TABLE_CHECKBOX_COLUMN_ID = 'select';

export enum ExperimentViewTracesTableColumns {
  requestId = 'request_id',
  traceName = 'traceName',
  timestampMs = 'timestamp_ms',
  inputs = 'inputs',
  outputs = 'outputs',
  runName = 'runName',
  totalTokens = 'total_tokens',
  source = 'source',
  latency = 'latency',
  tags = 'tags',
  status = 'status',
}

export const ExperimentViewTracesTableColumnLabels: Record<ExperimentViewTracesTableColumns, MessageDescriptor> = {
  [ExperimentViewTracesTableColumns.requestId]: defineMessage({
    defaultMessage: 'Request ID',
    description: 'Experiment page > traces table > request id column header',
  }),
  [ExperimentViewTracesTableColumns.traceName]: defineMessage({
    defaultMessage: 'Trace name',
    description: 'Experiment page > traces table > trace name column header',
  }),
  [ExperimentViewTracesTableColumns.timestampMs]: defineMessage({
    defaultMessage: 'Time created',
    description: 'Experiment page > traces table > time created column header',
  }),
  [ExperimentViewTracesTableColumns.status]: defineMessage({
    defaultMessage: 'Status',
    description: 'Experiment page > traces table > status column header',
  }),
  [ExperimentViewTracesTableColumns.inputs]: defineMessage({
    defaultMessage: 'Request',
    description: 'Experiment page > traces table > input column header',
  }),
  [ExperimentViewTracesTableColumns.outputs]: defineMessage({
    defaultMessage: 'Response',
    description: 'Experiment page > traces table > output column header',
  }),
  [ExperimentViewTracesTableColumns.runName]: defineMessage({
    defaultMessage: 'Run name',
    description: 'Experiment page > traces table > run name column header',
  }),
  [ExperimentViewTracesTableColumns.totalTokens]: defineMessage({
    defaultMessage: 'Tokens',
    description: 'Experiment page > traces table > tokens column header',
  }),
  [ExperimentViewTracesTableColumns.source]: defineMessage({
    defaultMessage: 'Source',
    description: 'Experiment page > traces table > source column header',
  }),
  [ExperimentViewTracesTableColumns.latency]: defineMessage({
    defaultMessage: 'Execution time',
    description: 'Experiment page > traces table > latency column header',
  }),
  [ExperimentViewTracesTableColumns.tags]: defineMessage({
    defaultMessage: 'Tags',
    description: 'Experiment page > traces table > tags column header',
  }),
};

export const ExperimentViewTracesStatusLabels = {
  UNSET: null,
  IN_PROGRESS: defineMessage({
    defaultMessage: 'In progress',
    description: 'Experiment page > traces table > status label > in progress',
  }),
  OK: defineMessage({
    defaultMessage: 'OK',
    description: 'Experiment page > traces table > status label > ok',
  }),
  ERROR: defineMessage({
    defaultMessage: 'Error',
    description: 'Experiment page > traces table > status label > error',
  }),
};
