import { useCallback, useMemo } from 'react';
import { useIntl } from 'react-intl';
import {
  convertTraceInfoV3ToRunEvalEntry,
  createTraceLocationForExperiment,
  getAssessmentInfos,
  SEARCH_MLFLOW_TRACES_QUERY_KEY,
  searchMlflowTracesQueryFn,
  type AssessmentAggregates,
  type AssessmentDType,
  type AssessmentInfo,
  type EvalTraceComparisonEntry,
  type RunEvaluationTracesDataEntry,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from their modules.
import { getAssessmentAggregates } from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import { filterTracesByAssessmentSourceRunId } from '@databricks/web-shared/genai-traces-table/utils/TraceUtils';
import type { ModelTraceInfoV3, ModelTraceSearchLocation } from '@databricks/web-shared/model-trace-explorer';
import { useQuery } from '@databricks/web-shared/query-client';
import { buildEvalRunTraceFilter } from './useEvalRunTracesV4Controller';

/** The header summaries cover at most this many of the run's traces. */
export const AGGREGATE_TRACE_LIMIT = 1000;

export interface EvalRunAssessmentAggregate {
  name: string;
  displayName: string;
  dtype: AssessmentDType;
  assessmentInfo: AssessmentInfo;
  aggregates: AssessmentAggregates;
}

export interface EvalRunAssessmentAggregatesResult {
  scorers: EvalRunAssessmentAggregate[];
  refetch: () => void;
  error: Error | null;
}

/** The run's traces for the header summaries: the whole run (not the table's search), capped. */
export const fetchEvalRunTracesForAggregates = async ({
  locations,
  runUuid,
  signal,
}: {
  locations: ModelTraceSearchLocation[];
  runUuid: string;
  signal?: AbortSignal;
}): Promise<ModelTraceInfoV3[]> => {
  const traces = await searchMlflowTracesQueryFn({
    signal,
    locations,
    filter: buildEvalRunTraceFilter({ runUuid }),
    limit: AGGREGATE_TRACE_LIMIT,
  });
  // The search pages until it reaches the limit, so the last page can overshoot it.
  return traces.slice(0, AGGREGATE_TRACE_LIMIT);
};

/** Loads the run's assessment summaries independently of the table's row search. */
export const useEvalRunAssessmentAggregates = ({
  experimentId,
  runUuid,
}: {
  experimentId: string;
  runUuid: string;
}): EvalRunAssessmentAggregatesResult => {
  const intl = useIntl();
  const locations = useMemo(() => [createTraceLocationForExperiment(experimentId)], [experimentId]);
  const aggregateQuery = useQuery<ModelTraceInfoV3[], Error>({
    queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY, 'eval-run-assessment-aggregates', experimentId, runUuid],
    queryFn: ({ signal }) => fetchEvalRunTracesForAggregates({ locations, runUuid, signal }),
    refetchOnWindowFocus: false,
  });
  // Only this run's assessments: feedback other runs logged on a linked trace stays out.
  const evaluationResults = useMemo<RunEvaluationTracesDataEntry[]>(
    () =>
      (filterTracesByAssessmentSourceRunId(aggregateQuery.data, runUuid) ?? []).map((trace) =>
        convertTraceInfoV3ToRunEvalEntry(trace),
      ),
    [aggregateQuery.data, runUuid],
  );
  const evaluatedTraces = useMemo<EvalTraceComparisonEntry[]>(
    () => evaluationResults.map((currentRunValue) => ({ currentRunValue })),
    [evaluationResults],
  );
  const assessmentInfos = useMemo(
    () => getAssessmentInfos(intl, evaluationResults, undefined),
    [intl, evaluationResults],
  );

  const scorers = useMemo(
    () =>
      assessmentInfos.map((assessmentInfo) => ({
        name: assessmentInfo.name,
        displayName: assessmentInfo.displayName,
        dtype: assessmentInfo.dtype,
        assessmentInfo,
        aggregates: getAssessmentAggregates(assessmentInfo, evaluatedTraces, []),
      })),
    [assessmentInfos, evaluatedTraces],
  );
  const { refetch: refetchAggregateQuery } = aggregateQuery;
  const refetch = useCallback(() => {
    void refetchAggregateQuery();
  }, [refetchAggregateQuery]);

  return { scorers, refetch, error: aggregateQuery.error ?? null };
};
