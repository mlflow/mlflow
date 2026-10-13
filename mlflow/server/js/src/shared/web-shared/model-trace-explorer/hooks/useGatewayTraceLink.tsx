import { useQuery } from '../../query-client/queryClient';

import { getTraceHref } from '../ModelTraceExplorer.utils';
import { fetchTraceInfoV3 } from '../api';

const GATEWAY_TRACE_INFO_QUERY_KEY = 'GATEWAY_TRACE_INFO';

export const useGatewayTraceLink = (linkedTraceId: string | undefined): string | undefined => {
  const { data } = useQuery({
    queryKey: [GATEWAY_TRACE_INFO_QUERY_KEY, linkedTraceId],
    queryFn: () => fetchTraceInfoV3({ traceId: linkedTraceId ?? '' }),
    enabled: Boolean(linkedTraceId),
    refetchOnWindowFocus: false,
    retry: false,
  });

  if (!linkedTraceId || !data?.trace?.trace_info) return undefined;
  return getTraceHref(linkedTraceId, data.trace.trace_info);
};
