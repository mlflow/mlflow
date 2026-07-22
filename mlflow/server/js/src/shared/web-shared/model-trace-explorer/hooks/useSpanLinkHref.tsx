import { useMemo } from 'react';

import { useQuery } from '../../query-client/queryClient';

import { isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { fetchTraceInfoV3 } from '../api';
import { getExperimentPageTracesTabRoute } from '../routes';

const SPAN_LINK_TRACE_INFO_QUERY_KEY = 'SPAN_LINK_TRACE_INFO';

// Only supports V3 trace IDs (tr-...). V4/UC traces skip link materialization
// in the Python layer, so trace:/<location>/<hex> IDs won't appear here.
export const useSpanLinkHref = (traceId: string | undefined): string | undefined => {
  const { data } = useQuery({
    queryKey: [SPAN_LINK_TRACE_INFO_QUERY_KEY, traceId],
    queryFn: () => fetchTraceInfoV3({ traceId: traceId ?? '' }),
    enabled: Boolean(traceId),
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

    return `${getExperimentPageTracesTabRoute(experimentId)}?selectedEvaluationId=${traceId}`;
  }, [traceId, data?.trace]);
};
