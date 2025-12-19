export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
export type MakeEmpty<T extends { [key: string]: unknown }, K extends keyof T> = { [_ in K]?: never };
export type Incremental<T> = T | { [P in keyof T]?: P extends ' $fragmentName' | '__typename' ? T[P] : never };
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: { input: string; output: string; }
  String: { input: string; output: string; }
  Boolean: { input: boolean; output: boolean; }
  Int: { input: number; output: number; }
  Float: { input: number; output: number; }
  /** LongString Scalar type to prevent truncation to max integer in JavaScript. */
  LongString: { input: string; output: string; }
};

export enum MlflowDeploymentJobConnectionState {
  CONNECTED = 'CONNECTED',
  DEPLOYMENT_JOB_CONNECTION_STATE_UNSPECIFIED = 'DEPLOYMENT_JOB_CONNECTION_STATE_UNSPECIFIED',
  NOT_FOUND = 'NOT_FOUND',
  NOT_SET_UP = 'NOT_SET_UP',
  REQUIRED_PARAMETERS_CHANGED = 'REQUIRED_PARAMETERS_CHANGED'
}

export type MlflowGetExperimentInput = {
  experimentId?: InputMaybe<Scalars['String']['input']>;
};

export type MlflowGetMetricHistoryBulkIntervalInput = {
  endStep?: InputMaybe<Scalars['Int']['input']>;
  maxResults?: InputMaybe<Scalars['Int']['input']>;
  metricKey?: InputMaybe<Scalars['String']['input']>;
  runIds?: InputMaybe<Array<InputMaybe<Scalars['String']['input']>>>;
  startStep?: InputMaybe<Scalars['Int']['input']>;
};

export type MlflowGetRunInput = {
  runId?: InputMaybe<Scalars['String']['input']>;
  runUuid?: InputMaybe<Scalars['String']['input']>;
};

export enum MlflowModelVersionDeploymentJobStateDeploymentJobRunState {
  APPROVAL = 'APPROVAL',
  DEPLOYMENT_JOB_RUN_STATE_UNSPECIFIED = 'DEPLOYMENT_JOB_RUN_STATE_UNSPECIFIED',
  FAILED = 'FAILED',
  NO_VALID_DEPLOYMENT_JOB_FOUND = 'NO_VALID_DEPLOYMENT_JOB_FOUND',
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  SUCCEEDED = 'SUCCEEDED'
}

export enum MlflowModelVersionStatus {
  FAILED_REGISTRATION = 'FAILED_REGISTRATION',
  PENDING_REGISTRATION = 'PENDING_REGISTRATION',
  READY = 'READY'
}

export enum MlflowRunStatus {
  FAILED = 'FAILED',
  FINISHED = 'FINISHED',
  KILLED = 'KILLED',
  RUNNING = 'RUNNING',
  SCHEDULED = 'SCHEDULED'
}

export type MlflowSearchRunsInput = {
  experimentIds?: InputMaybe<Array<InputMaybe<Scalars['String']['input']>>>;
  filter?: InputMaybe<Scalars['String']['input']>;
  maxResults?: InputMaybe<Scalars['Int']['input']>;
  orderBy?: InputMaybe<Array<InputMaybe<Scalars['String']['input']>>>;
  pageToken?: InputMaybe<Scalars['String']['input']>;
  runViewType?: InputMaybe<MlflowViewType>;
};

export enum MlflowViewType {
  ACTIVE_ONLY = 'ACTIVE_ONLY',
  ALL = 'ALL',
  DELETED_ONLY = 'DELETED_ONLY'
}

export type GetRunVariables = Exact<{
  data: MlflowGetRunInput;
}>;


export type GetRun = { mlflowGetRun: { __typename: 'MlflowGetRunResponse', apiError: { __typename: 'ApiError', helpUrl: string | null, code: string | null, message: string | null } | null, run: { __typename: 'MlflowRunExtension', info: { __typename: 'MlflowRunInfo', runName: string | null, status: MlflowRunStatus | null, runUuid: string | null, experimentId: string | null, artifactUri: string | null, endTime: string | null, lifecycleStage: string | null, startTime: string | null, userId: string | null } | null, experiment: { __typename: 'MlflowExperiment', experimentId: string | null, name: string | null, artifactLocation: string | null, lifecycleStage: string | null, lastUpdateTime: string | null, tags: Array<{ __typename: 'MlflowExperimentTag', key: string | null, value: string | null }> | null } | null, modelVersions: Array<{ __typename: 'MlflowModelVersion', status: MlflowModelVersionStatus | null, version: string | null, name: string | null, source: string | null }> | null, data: { __typename: 'MlflowRunData', metrics: Array<{ __typename: 'MlflowMetricExtension', key: string | null, value: number | null, step: string | null, timestamp: string | null }> | null, params: Array<{ __typename: 'MlflowParam', key: string | null, value: string | null }> | null, tags: Array<{ __typename: 'MlflowRunTag', key: string | null, value: string | null }> | null } | null, inputs: { __typename: 'MlflowRunInputs', datasetInputs: Array<{ __typename: 'MlflowDatasetInput', dataset: { __typename: 'MlflowDataset', digest: string | null, name: string | null, profile: string | null, schema: string | null, source: string | null, sourceType: string | null } | null, tags: Array<{ __typename: 'MlflowInputTag', key: string | null, value: string | null }> | null }> | null, modelInputs: Array<{ __typename: 'MlflowModelInput', modelId: string | null }> | null } | null, outputs: { __typename: 'MlflowRunOutputs', modelOutputs: Array<{ __typename: 'MlflowModelOutput', modelId: string | null, step: string | null }> | null } | null } | null } | null };

export type SearchRunsVariables = Exact<{
  data: MlflowSearchRunsInput;
}>;


export type SearchRuns = { mlflowSearchRuns: { __typename: 'MlflowSearchRunsResponse', apiError: { __typename: 'ApiError', helpUrl: string | null, code: string | null, message: string | null } | null, runs: Array<{ __typename: 'MlflowRunExtension', info: { __typename: 'MlflowRunInfo', runName: string | null, status: MlflowRunStatus | null, runUuid: string | null, experimentId: string | null, artifactUri: string | null, endTime: string | null, lifecycleStage: string | null, startTime: string | null, userId: string | null } | null, experiment: { __typename: 'MlflowExperiment', experimentId: string | null, name: string | null, artifactLocation: string | null, lifecycleStage: string | null, lastUpdateTime: string | null, tags: Array<{ __typename: 'MlflowExperimentTag', key: string | null, value: string | null }> | null } | null, data: { __typename: 'MlflowRunData', metrics: Array<{ __typename: 'MlflowMetricExtension', key: string | null, value: number | null, step: string | null, timestamp: string | null }> | null, params: Array<{ __typename: 'MlflowParam', key: string | null, value: string | null }> | null, tags: Array<{ __typename: 'MlflowRunTag', key: string | null, value: string | null }> | null } | null, inputs: { __typename: 'MlflowRunInputs', datasetInputs: Array<{ __typename: 'MlflowDatasetInput', dataset: { __typename: 'MlflowDataset', digest: string | null, name: string | null, profile: string | null, schema: string | null, source: string | null, sourceType: string | null } | null, tags: Array<{ __typename: 'MlflowInputTag', key: string | null, value: string | null }> | null }> | null, modelInputs: Array<{ __typename: 'MlflowModelInput', modelId: string | null }> | null } | null, outputs: { __typename: 'MlflowRunOutputs', modelOutputs: Array<{ __typename: 'MlflowModelOutput', modelId: string | null, step: string | null }> | null } | null, modelVersions: Array<{ __typename: 'MlflowModelVersion', version: string | null, name: string | null, creationTimestamp: string | null, status: MlflowModelVersionStatus | null, source: string | null }> | null }> | null } | null };

export type GetMetricHistoryBulkIntervalVariables = Exact<{
  data: MlflowGetMetricHistoryBulkIntervalInput;
}>;


export type GetMetricHistoryBulkInterval = { mlflowGetMetricHistoryBulkInterval: { __typename: 'MlflowGetMetricHistoryBulkIntervalResponse', metrics: Array<{ __typename: 'MlflowMetricWithRunId', timestamp: string | null, step: string | null, runId: string | null, key: string | null, value: number | null }> | null, apiError: { __typename: 'ApiError', code: string | null, message: string | null } | null } | null };

export type MlflowGetExperimentQueryVariables = Exact<{
  input: MlflowGetExperimentInput;
}>;


export type MlflowGetExperimentQuery = { mlflowGetExperiment: { __typename: 'MlflowGetExperimentResponse', apiError: { __typename: 'ApiError', code: string | null, message: string | null } | null, experiment: { __typename: 'MlflowExperiment', artifactLocation: string | null, creationTime: string | null, experimentId: string | null, lastUpdateTime: string | null, lifecycleStage: string | null, name: string | null, tags: Array<{ __typename: 'MlflowExperimentTag', key: string | null, value: string | null }> | null } | null } | null };
