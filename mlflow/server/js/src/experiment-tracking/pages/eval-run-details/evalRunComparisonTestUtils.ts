import type { ModelTraceInfoV3, ModelTraceSpanV3 } from '@databricks/web-shared/model-trace-explorer';
import type { UseGetRunQueryDataResponse } from '../../components/run-page/hooks/useGetRunQuery';

export const createComparisonRootSpan = (
  input: string,
  overrides: Partial<ModelTraceSpanV3> = {},
): ModelTraceSpanV3 => ({
  trace_id: 'trace-id',
  span_id: '0123456789abcdef',
  parent_span_id: '',
  trace_state: '',
  name: 'root',
  start_time_unix_nano: '1',
  end_time_unix_nano: '2',
  status: { code: 'STATUS_CODE_OK' },
  attributes: { 'mlflow.spanInputs': input },
  ...overrides,
});

export const createComparisonRun = (runUuid: string): NonNullable<UseGetRunQueryDataResponse> => ({
  __typename: 'MlflowRunExtension',
  info: {
    __typename: 'MlflowRunInfo',
    runName: `Run ${runUuid}`,
    runUuid,
    experimentId: 'experiment-id',
    status: null,
    artifactUri: null,
    lifecycleStage: 'active',
    startTime: '1767225600000',
    endTime: null,
    userId: `creator-${runUuid}`,
  },
  experiment: null,
  modelVersions: null,
  outputs: null,
  data: {
    __typename: 'MlflowRunData',
    metrics: [],
    params: [{ __typename: 'MlflowParam', key: 'endpoint', value: `endpoint-${runUuid}` }],
    tags: [],
  },
  inputs: {
    __typename: 'MlflowRunInputs',
    modelInputs: [],
    datasetInputs: [
      {
        __typename: 'MlflowDatasetInput',
        tags: [],
        dataset: {
          __typename: 'MlflowDataset',
          name: `dataset-${runUuid}`,
          digest: null,
          profile: null,
          schema: null,
          source: null,
          sourceType: null,
        },
      },
    ],
  },
});

export const createComparisonTrace = (
  runUuid: string,
  scores: Record<string, boolean | number | string>,
  index = 0,
  rationale = 'Sampled feedback',
): ModelTraceInfoV3 => ({
  trace_id: `trace-${runUuid}-${index}`,
  trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-id' } },
  request_time: '2026-01-01T00:00:00Z',
  state: 'OK',
  tags: {},
  request_preview: 'PRIVATE INPUT MUST NOT BE SENT',
  response_preview: 'PRIVATE OUTPUT MUST NOT BE SENT',
  trace_metadata: { 'mlflow.sourceRun': runUuid },
  assessments: Object.entries(scores).map(([name, value]) => ({
    assessment_id: `${runUuid}-${index}-${name}`,
    assessment_name: name,
    trace_id: `trace-${runUuid}-${index}`,
    create_time: '2026-01-01T00:00:00Z',
    last_update_time: '2026-01-01T00:00:00Z',
    feedback: { value },
    source: { source_type: 'CODE', source_id: 'scorer' },
    rationale,
  })),
});
