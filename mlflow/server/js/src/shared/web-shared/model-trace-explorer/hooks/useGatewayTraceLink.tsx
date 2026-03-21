import { useMemo } from 'react';

import { useQuery } from '../../query-client/queryClient';

import { isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { fetchTraceInfoV3 } from '../api';
import { getExperimentPageTracesTabRoute } from '../routes';

const GATEWAY_TRACE_INFO_QUERY_KEY = 'GATEWAY_TRACE_INFO';

/**
 * Fetches trace info for a linked gateway trace and returns a navigable href.
 * Returns undefined while loading or if the experiment ID cannot be resolved.
 */
export const useGatewayTraceLink = (linkedTraceId: string | undefined): string | undefined => {
  const { data } = useQuery({
    queryKey: [GATEWAY_TRACE_INFO_QUERY_KEY, linkedTraceId],
    queryFn: () => fetchTraceInfoV3({ traceId: linkedTraceId ?? '' }),
    enabled: Boolean(linkedTraceId),
    refetchOnWindowFocus: false,
    retry: false,
  });

  return useMemo(() => {
    if (!linkedTraceId || !data?.trace) {
      return undefined;
    }

    let experimentId: string | undefined;

    if (isV3ModelTraceInfo(data?.trace?.trace_info)) {
      // V3 format: trace info at root with trace_location
      if (data?.trace?.trace_info?.trace_location?.type === 'MLFLOW_EXPERIMENT') {
        experimentId = data?.trace?.trace_info?.trace_location.mlflow_experiment?.experiment_id;
      }
    } else {
      // V2 format: nested under data.trace.trace_info
      experimentId = data?.trace?.trace_info?.experiment_id;
    }

    if (!experimentId) {
      return undefined;
    }

    return `${getExperimentPageTracesTabRoute(experimentId)}?selectedEvaluationId=${linkedTraceId}`;
  }, [linkedTraceId, data?.trace]);
};
