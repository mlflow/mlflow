import { useMemo } from 'react';
import { MLFLOW_SOURCE_RUN_KEY } from '@databricks/web-shared/genai-traces-table';
import { MLFLOW_TRACE_USER_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { useMlflowTraceFilterFields } from '../../components/experiment-page/components/traces-v4/utils/filterModel';
import {
  COMPARISON_FILTER_FIELDS,
  getComparisonFilterAssessments,
  getComparisonTraceInput,
  getComparisonTraceOutput,
} from './evalRunComparisonTraceFilters';
import type { EvalComparisonRunWithTraces } from './evalRunComparisonData';

export const useEvalRunComparisonFilterFields = (
  runs: EvalComparisonRunWithTraces[],
  recordedInputs: ReadonlyMap<string, string>,
) => {
  // Suggest only values the local predicates can match, without modifying shared query-cache objects.
  const suggestionTraces = useMemo<ModelTraceInfoV3[]>(
    () =>
      runs.flatMap((run) =>
        run.traceInfos.map((trace) => ({
          ...trace,
          trace_metadata: {
            ...trace.trace_metadata,
            [MLFLOW_SOURCE_RUN_KEY]: run.runUuid,
            ...((trace.trace_metadata?.[MLFLOW_TRACE_USER_KEY] ?? trace.tags?.[MLFLOW_TRACE_USER_KEY]) !== undefined
              ? {
                  [MLFLOW_TRACE_USER_KEY]:
                    trace.trace_metadata?.[MLFLOW_TRACE_USER_KEY] ?? trace.tags[MLFLOW_TRACE_USER_KEY],
                }
              : {}),
          },
          request: getComparisonTraceInput(trace, recordedInputs),
          request_preview: undefined,
          response: getComparisonTraceOutput(trace),
          response_preview: undefined,
          assessments: getComparisonFilterAssessments(trace),
        })),
      ),
    [runs, recordedInputs],
  );
  const fields = useMlflowTraceFilterFields(undefined, suggestionTraces);
  return useMemo(() => fields.filter(({ id }) => COMPARISON_FILTER_FIELDS.has(id)), [fields]);
};
