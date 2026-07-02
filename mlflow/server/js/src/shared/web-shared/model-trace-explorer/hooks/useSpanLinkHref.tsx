import { useMemo } from 'react';

import { useQuery } from '../../query-client/queryClient';

import { isV3ModelTraceInfo, parseV4TraceIdToObject } from '../ModelTraceExplorer.utils';
import { fetchTraceInfoV3 } from '../api';
import { getExperimentPageTracesTabRoute } from '../routes';

const SPAN_LINK_TRACE_INFO_QUERY_KEY = 'SPAN_LINK_TRACE_INFO';

export const useSpanLinkHref = (traceId: string | undefined): string | undefined => {
  const resolvedTraceId = useMemo(() => {
    if (!traceId) return undefined;
    return parseV4TraceIdToObject(traceId)?.trace_id ?? traceId;
  }, [traceId]);

  const { data } = useQuery({
    queryKey: [SPAN_LINK_TRACE_INFO_QUERY_KEY, resolvedTraceId],
    queryFn: () => fetchTraceInfoV3({ traceId: resolvedTraceId ?? '' }),
    enabled: Boolean(resolvedTraceId),
    refetchOnWindowFocus: false,
    retry: false,
    staleTime: Infinity,
  });

  return useMemo(() => {
    if (!traceId || !data?.trace) {
      return undefined;
    }

    let experimentId: string | undefined;

    if (isV3ModelTraceInfo(data?.trace?.trace_info)) {
      if (data?.trace?.trace_info?.trace_location?.type === 'MLFLOW_EXPERIMENT') {
        experimentId = data?.trace?.trace_info?.trace_location.mlflow_experiment?.experiment_id;
      }
    } else {
      experimentId = data?.trace?.trace_info?.experiment_id;
    }

    if (!experimentId) {
      return undefined;
    }

    return `${getExperimentPageTracesTabRoute(experimentId)}?selectedEvaluationId=${resolvedTraceId}`;
  }, [traceId, resolvedTraceId, data?.trace]);
};
