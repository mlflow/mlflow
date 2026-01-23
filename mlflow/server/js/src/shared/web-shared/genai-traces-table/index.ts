// TBD: this module will provide a space for trace UI components shared
// between new Tiles UI in webapp and evaluation review UI in MLflow.
// Eventually everything will be moved to monolith codebase.

export { mergeMetricsAndAssessmentsWithEvaluations } from './utils/EvaluationDataParseUtils';

export { COMPARE_TO_RUN_COLOR, CURRENT_RUN_COLOR } from './utils/Colors';

export { useAssessmentFilters } from './hooks/useAssessmentFilters';

export {
  isEvaluationResultOverallAssessment,
  isEvaluationResultPerRetrievalChunkAssessment,
  isRetrievedContext,
  KnownEvaluationResultAssessmentStringValue,
  KnownEvaluationResultAssessmentValueLabel,
} from './components/GenAiEvaluationTracesReview.utils';

export { useTableSort } from './hooks/useTableSort';

export { GenAiTracesTable } from './GenAITracesTable';
export { useGenAiExperimentRunsForComparison } from './hooks/useGenAiExperimentRunsForComparison';
export { useGenAiTraceEvaluationArtifacts } from './hooks/useGenAiTraceEvaluationArtifacts';
export {
  useSearchMlflowTraces,
  useMlflowTracesTableMetadata,
  invalidateMlflowSearchTracesCache,
  searchMlflowTracesQueryFn,
  SEARCH_MLFLOW_TRACES_QUERY_KEY,
} from './hooks/useMlflowTraces';
export { getEvalTabTotalTracesLimit } from './utils/FeatureUtils';
export { GenAITracesTableToolbar } from './GenAITracesTableToolbar';
export { GenAiTracesTableSearchInput } from './GenAiTracesTableSearchInput';
export { GenAITracesTableBodyContainer } from './GenAITracesTableBodyContainer';
export { useTableColumns } from './hooks/useTableColumns';
export { getAssessmentInfos } from './utils/AggregationUtils';

export { useFilters } from './hooks/useFilters';

export { GenAITracesTableContext, GenAITracesTableProvider } from './GenAITracesTableContext';

export { MarkdownConverterProvider as GenAiTracesMarkdownConverterProvider } from './utils/MarkdownUtils';

export { RunColorCircle } from './components/RunColorCircle';

export { useSelectedColumns } from './hooks/useGenAITracesUIState';
export { useTableSortURL } from './hooks/useTableSortURL';
export { useColumnsURL } from './hooks/useColumnsURL';

export { GenAiEvaluationTracesReviewModal } from './components/GenAiEvaluationTracesReviewModal';

export * from './types';

export {
  GenAiTraceEvaluationArtifactFile,
  KnownEvaluationResultAssessmentMetadataFields,
  KnownEvaluationResultAssessmentName,
} from './enum';

export {
  ASSESSMENTS_DOC_LINKS,
  getJudgeMetricsLink,
  KnownEvaluationResultAssessmentValueDescription,
} from './components/GenAiEvaluationTracesReview.utils';

export {
  COMPARE_TO_RUN_DROPDOWN_COMPONENT_ID,
  RUN_EVALUATION_RESULTS_TAB_COMPARE_RUNS,
  RUN_EVALUATION_RESULTS_TAB_SINGLE_RUN,
} from './utils/EvaluationLogging';

export {
  getTracesTagKeys,
  getTraceInfoInputs,
  getTraceInfoOutputs,
  convertTraceInfoV3ToRunEvalEntry,
  getSpanAttribute,
  formatTraceId,
} from './utils/TraceUtils';

export {
  REQUEST_TIME_COLUMN_ID,
  EXECUTION_DURATION_COLUMN_ID,
  STATE_COLUMN_ID,
  TAGS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  TOKENS_COLUMN_ID,
  TRACE_ID_COLUMN_ID,
  CUSTOM_METADATA_COLUMN_ID,
  SESSION_COLUMN_ID,
  INPUTS_COLUMN_ID,
} from './hooks/useTableColumns';

// Test utilities
export {
  createTestTraceInfoV3,
  createTestAssessmentInfo,
  createTestColumns,
} from './test-fixtures/EvaluatedTraceTestUtils';

export { shouldUseTracesV4API } from './utils/FeatureUtils';
export { createTraceLocationForExperiment, createTraceLocationForUCSchema } from './utils/TraceLocationUtils';
export type { GetTraceFunction } from './hooks/useGetTrace';
export { useFetchTraceV4LazyQuery, useFetchTraceV4Query, getTraceV4QueryKey } from './hooks/useFetchTraceV4';
export { doesTraceSupportV4API } from './utils/TraceLocationUtils';
export { GenAIChatSessionsTable } from './sessions-table/GenAIChatSessionsTable';
export { useGetTraces } from './hooks/useGetTraces';
export { useGetTrace } from './hooks/useGetTrace';
