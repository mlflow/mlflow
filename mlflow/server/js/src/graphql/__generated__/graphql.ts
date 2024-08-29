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
  LongString: { input: GraphQLLongString; output: GraphQLLongString; }
};

export type MlflowGetRunInput = {
  runId?: InputMaybe<Scalars['String']['input']>;
  runUuid?: InputMaybe<Scalars['String']['input']>;
};

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

export type GetRunVariables = Exact<{
  data: MlflowGetRunInput;
}>;


export type GetRun = { mlflowGetRun: { __typename: 'MlflowGetRunResponse', apiError: { __typename: 'ApiError', helpUrl: string | null, code: string | null, message: string | null } | null, run: { __typename: 'MlflowRunExtension', info: { __typename: 'MlflowRunInfo', runName: string | null, status: MlflowRunStatus | null, runUuid: string | null, experimentId: string | null, artifactUri: string | null, endTime: GraphQLLongString | null, lifecycleStage: string | null, startTime: GraphQLLongString | null, userId: string | null } | null, experiment: { __typename: 'MlflowExperiment', experimentId: string | null, name: string | null, artifactLocation: string | null, lifecycleStage: string | null, lastUpdateTime: GraphQLLongString | null, tags: Array<{ __typename: 'MlflowExperimentTag', key: string | null, value: string | null }> | null } | null, modelVersions: Array<{ __typename: 'MlflowModelVersion', status: MlflowModelVersionStatus | null, version: string | null, name: string | null, source: string | null }> | null, data: { __typename: 'MlflowRunData', metrics: Array<{ __typename: 'MlflowMetric', key: string | null, value: number | null, step: GraphQLLongString | null, timestamp: GraphQLLongString | null }> | null, params: Array<{ __typename: 'MlflowParam', key: string | null, value: string | null }> | null, tags: Array<{ __typename: 'MlflowRunTag', key: string | null, value: string | null }> | null } | null, inputs: { __typename: 'MlflowRunInputs', datasetInputs: Array<{ __typename: 'MlflowDatasetInput', dataset: { __typename: 'MlflowDataset', digest: string | null, name: string | null, profile: string | null, schema: string | null, source: string | null, sourceType: string | null } | null, tags: Array<{ __typename: 'MlflowInputTag', key: string | null, value: string | null }> | null }> | null } | null } | null } | null };
