import { getExperimentTraceV3 } from '@databricks/web-shared/model-trace-explorer';
import { useQuery } from '@databricks/web-shared/query-client';

import { extractPriorAnswers, type PriorAnswer, type RawTraceAssessment } from '../reviewAnswers';

export const REVIEW_QUEUE_TRACE_ASSESSMENTS_QUERY_KEY = 'REVIEW_QUEUE_TRACE_ASSESSMENTS';

interface TraceGetResponse {
  trace?: { trace_info?: { assessments?: RawTraceAssessment[] } };
  trace_info?: { assessments?: RawTraceAssessment[] };
}

/**
 * Fetch a trace's existing assessments, normalized into prior answers used to
 * prefill the focused-review widgets. Reuses the trace-get endpoint via the
 * exported `getExperimentTraceV3`; the v3 response nests the info under
 * `trace.trace_info`, with a defensive fallback to a flatter shape.
 */
export const useTraceAssessmentsQuery = ({
  traceId,
  sourceId,
  enabled = true,
}: {
  traceId: string;
  /** When set, scope prior answers to this reviewer's own source. */
  sourceId?: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, isFetching } = useQuery<PriorAnswer[], Error>({
    queryKey: [REVIEW_QUEUE_TRACE_ASSESSMENTS_QUERY_KEY, traceId, sourceId],
    queryFn: async () => {
      const response = (await getExperimentTraceV3({ traceId })) as TraceGetResponse;
      const assessments = response?.trace?.trace_info?.assessments ?? response?.trace_info?.assessments ?? [];
      return extractPriorAnswers(assessments, sourceId);
    },
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(traceId),
  });

  return { priorAnswers: data ?? [], isLoading, isFetching };
};
