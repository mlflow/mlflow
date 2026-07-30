import { useQuery } from '../../query-client/queryClient';

import { getTraceHref } from '../ModelTraceExplorer.utils';
import { fetchTraceInfoV3 } from '../api';

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

  if (!traceId || !data?.trace?.trace_info) return undefined;
  return getTraceHref(traceId, data.trace.trace_info);
};
