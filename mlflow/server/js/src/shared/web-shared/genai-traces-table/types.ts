import type { ModelTrace, ModelTraceInfo, ModelTraceSpan } from '../model-trace-explorer';

export type AssessmentDType = 'string' | 'numeric' | 'boolean' | 'pass-fail' | 'unknown';
export type AssessmentType = 'AI_JUDGE' | 'HUMAN' | 'CODE';

// Reflects structure logged by mlflow.log_table()
export interface RawGenaiEvaluationArtifactResponse {
  columns?: (string | null)[];
  data?: (string | number | null | boolean | Record<string, any>)[][];
  filename: string;
}

export interface AssessmentInfo {
  name: string;
  displayName: string;
  // True when the assessment comes from a built-in judge.
  isKnown: boolean;
  isOverall: boolean;
  // The metric that produced the assessment. Defined as the name of the judge or custom metric function that produced the assessment.
  metricName: string;
  source?: RunEvaluationResultAssessmentSource;
  isCustomMetric: boolean;
  isEditable: boolean;
  isRetrievalAssessment: boolean;
  // The type of the assessment value.
  dtype: AssessmentDType;
  uniqueValues: Set<AssessmentValueType>;

  // Display metadata.
  docsLink: string;
  missingTooltip: string;
  description: string;

  // True if if the assesment contains at least one error
  containsErrors?: boolean;
}

interface RootCauseAssessmentInfo {
  assessmentName: string;
  suggestedActions?: string;
}

export interface EvaluationArtifactTableEntryAssessment {
  evaluation_id: string;

  name: string;

  boolean_value: boolean | null;
  numeric_value: number | null;
  string_value: string | null;
  rationale: string | null;

  source: {
    source_type: AssessmentType;
    source_id: string;
    metadata: any;
  };

  metadata?: Record<string, any>;

  timestamp: number;

  error_code?: string;
  error_message?: string;
}

export interface EvaluationArtifactTableEntryMetric {
  evaluation_id: string;
  key: string;
  value: number;
  timestamp: number;
}

export interface EvaluationArtifactTableEntryEvaluation {
  evaluation_id: string;
  inputs_id: string;
  request_id: string;
  run_id?: string;

  inputs: Record<string, any>;
  outputs: Record<string, any>;
  targets: Record<string, any>;
}

export type RunEvaluationResultAssessmentSource = {
  sourceType: AssessmentType;
  sourceId: string;
  metadata: Record<string, string>;
};

export type RunEvaluationResultAssessment = {
  name: string;
  rationale?: string | null;
  source?: RunEvaluationResultAssessmentSource;
  metadata?: Record<string, string | boolean | number>;
  errorCode?: string;
  errorMessage?: string;
  numericValue?: number | null;
  booleanValue?: boolean | null;
  stringValue?: string | null;
  // Root cause assessment points to the assessment name causing the failure.
  rootCauseAssessment?: RootCauseAssessmentInfo | null;
  timestamp?: number | null;
};

export type AssessmentValueType = string | boolean | number | undefined;

export type AssessmentRunCounts = Map<AssessmentValueType, number>;

export interface AssessmentAggregates {
  assessmentInfo: AssessmentInfo;

  // Counts for the current run and other run.
  currentCounts?: AssessmentRunCounts;
  otherCounts?: AssessmentRunCounts;

  // Numeric values for the current run and other run.
  currentNumericValues?: number[];
  otherNumericValues?: number[];

  currentNumRootCause: number;
  otherNumRootCause: number;

  // Numeric aggregate counts for the current run.
  currentNumericAggregate?: NumericAggregate;

  assessmentFilters: AssessmentFilter[];
}

export interface EvaluationsOverviewTableSort {
  key: string;
  type: TracesTableColumnType;
  asc: boolean;
}

export interface TraceActions {
  exportToEvals?: {
    showExportTracesToDatasetsModal: boolean;
    setShowExportTracesToDatasetsModal: (visible: boolean) => void;
    renderExportTracesToDatasetsModal: ({
      selectedTraceInfos,
    }: {
      selectedTraceInfos: ModelTrace['info'][];
    }) => React.ReactNode;
  };
  deleteTracesAction?: {
    deleteTraces: (experimentId: string, traceIds: string[]) => Promise<any>;
  };
  editTags?: {
    showEditTagsModalForTrace: (trace: ModelTraceInfo) => void;
    EditTagsModal: React.ReactNode;
  };
}

// @deprecated, use TableFilter instead
export interface AssessmentFilter {
  assessmentName: string;
  filterValue: AssessmentValueType;
  // Only defined when filtering on an assessment for RCA values.
  filterType?: 'rca' | undefined;
  run: string;
}
export type TableFilter = {
  // The column group (e.g. "Assessments") or a specific column (e.g. "execution_duration")
  column: TracesTableColumnGroup | string;
  // Should be defined if a column group is used.
  key?: string;
  operator: FilterOperator;
  value: TableFilterValue;
};

export type TableFilterValue = string | boolean | number | undefined;

export interface TableFilterOption {
  value: string;
  renderValue: () => string | React.ReactNode;
}

export interface TableFilterOptions {
  source: TableFilterOption[];
}

export enum FilterOperator {
  EQUALS = '=',
  GREATER_THAN = '>',
  LESS_THAN = '<',
  GREATER_THAN_OR_EQUALS = '>=',
  LESS_THAN_OR_EQUALS = '<=',
}

export interface AssessmentDropdownSuggestionItem {
  label: string;
  key: string;
  rootAssessmentName?: string;
  disabled?: boolean;
}

export interface RunEvaluationResultAssessmentDraft extends RunEvaluationResultAssessment {
  isDraft: true;
}

export type RunEvaluationResultMetric = {
  key: string;
  value: number;
  timestamp: number;
};

export type RunEvaluationTracesRetrievalChunk = {
  docUrl: string;
  content: string;
  retrievalAssessmentsByName?: Record<string, RunEvaluationResultAssessment[]>;
  target?: string;
};

// TODO(nsthorat): Move these to the shared types location:
// https://src.dev.databricks.com/databricks-eng/universe/-/blob/webapp/web/js/genai/shared/types.ts
// The shared type does not yet support TraceInfoV3.
// I had to add these here because the types in genai/shared/types are TraceV2.
// The types in trace-explorer are also TraceV2.
export type AssessmentV3 = {
  assessment_id: string;
  assessment_name: string;
  trace_id: string;
  span_id?: string;
  create_time: string;
  last_update_time: string;
  feedback?: {
    value: string | number | boolean;
    error?: {
      error_code?: string;
      error_message?: string;
    };
  };
  expectation?: {
    value: string | string[];
    serialized_value?: {
      serialization_format?: string;
      value: string | string[];
    };
    error?: {
      error_code?: string;
      error_message?: string;
    };
  };
  metadata?: Record<string, string>;
  rationale?: string;
  error?: {
    error_code?: string;
    error_message?: string;
  };
  source?: {
    source_type?: 'HUMAN' | 'LLM_JUDGE' | 'CODE';
    source_id?: string;
  };
};

export type TraceInfoV3 = {
  trace_id: string;
  client_request_id?: string;
  trace_location: {
    type: 'MLFLOW_EXPERIMENT' | 'INFERENCE_TABLE';
    mlflow_experiment?: { experiment_id: string };
    inference_table?: { full_table_name: string };
  };
  request?: string;
  request_preview?: string;
  response?: string;
  response_preview?: string;
  request_time: string;
  execution_duration?: string;
  state: 'STATE_UNSPECIFIED' | 'OK' | 'ERROR' | 'IN_PROGRESS';
  trace_metadata?: Record<string, string>;
  tags?: Record<string, string>;
  assessments?: AssessmentV3[];
};

export type TraceV3 = {
  info: TraceInfoV3;
  data: {
    spans: ModelTraceSpan[];
  };
};

/**
 * An entity encompassing single review evaluation data.
 */
export type RunEvaluationTracesDataEntry = {
  evaluationId: string;
  requestId: string;
  inputsTitle?: string;
  inputs: Record<string, any>;
  inputsId: string;
  outputs: Record<string, any>;
  targets: Record<string, any>;
  errorCode?: string;
  errorMessage?: string;
  requestTime?: string;
  overallAssessments: RunEvaluationResultAssessment[];
  responseAssessmentsByName: Record<
    // Keyed by assessment name (e.g. "overall_judgement", "readability_score" etc.)
    string,
    RunEvaluationResultAssessment[]
  >;
  metrics: Record<string, RunEvaluationResultMetric>;
  retrievalChunks?: RunEvaluationTracesRetrievalChunk[];

  // NOTE(nsthorat): We will slowly migrate to this type.
  traceInfo?: TraceInfoV3;
};

export interface EvalTraceComparisonEntry {
  currentRunValue?: RunEvaluationTracesDataEntry;
  otherRunValue?: RunEvaluationTracesDataEntry;
}

export interface SaveAssessmentsQuery {
  savePendingAssessments: (
    runUuid: string,
    evaluationId: string,
    pendingAssessmentEntries: RunEvaluationResultAssessmentDraft[],
  ) => void;
  isSaving: boolean;
}

// Internal type used to determine behavior of different types of columns.
// We should try to move away from this and start to use TracesTableColumnGroup instead.
export enum TracesTableColumnType {
  ASSESSMENT = 'ASSESSMENT',
  EXPECTATION = 'EXPECTATION',
  TRACE_INFO = 'TRACE_INFO',
  INPUT = 'INPUT',
  // This is a hack so that internal agent monitoring can display request time.
  INTERNAL_MONITOR_REQUEST_TIME = 'INTERNAL_MONITOR_REQUEST_TIME',
}

// This represents columns that are grouped together.
// For example, each assessment is its own column, but they are all grouped under the "Assessments" column group.
export enum TracesTableColumnGroup {
  ASSESSMENT = 'ASSESSMENT',
  EXPECTATION = 'EXPECTATION',
  TAG = 'TAG',
  INFO = 'INFO',
}

export const TracesTableColumnGroupToLabelMap = {
  [TracesTableColumnGroup.ASSESSMENT]: 'Assessments',
  [TracesTableColumnGroup.EXPECTATION]: 'Expectations',
  [TracesTableColumnGroup.TAG]: 'Tags',
  // We don't show a label for the info column group
  [TracesTableColumnGroup.INFO]: '\u00A0',
};

export interface TracesTableColumn {
  // This is the assessment name for assessments, and a static string for trace info and input columns
  id: string;
  label: string;
  type: TracesTableColumnType;
  group?: TracesTableColumnGroup;

  // TODO: Remove this field once migration to trace info v3 is complete
  assessmentInfo?: AssessmentInfo;
  expectationName?: string;
}

export interface TableFilterFormState {
  filters: TableFilter[];
}

// A bucket of a numeric aggregate.
export type NumericAggregateCount = {
  // The lower bound of the bucket, inclusive.
  lower: number;
  // The upper bound of the bucket, exclusive except for the last bucket.
  upper: number;
  // The number of values in the bucket.
  count: number;
};

// A numeric aggregate with the min, mid, and max values, and the counts of values in each bucket.
export type NumericAggregate = {
  min: number;
  max: number;
  maxCount: number;
  counts: NumericAggregateCount[];
};
