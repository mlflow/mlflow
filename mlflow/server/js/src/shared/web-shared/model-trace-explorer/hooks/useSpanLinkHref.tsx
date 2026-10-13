import { useMemo } from 'react';

import { useQuery } from '../../query-client/queryClient';

import { getTraceHref } from '../ModelTraceExplorer.utils';
import { fetchBatchTraceInfosV3 } from '../api';

const SPAN_LINK_TRACE_INFOS_QUERY_KEY = 'SPAN_LINK_TRACE_INFOS';

// Only supports V3 trace IDs (tr-...). V4/UC traces skip link materialization
// in the Python layer, so trace:/<location>/<hex> IDs won't appear here.
export const useSpanLinkHrefs = (traceIds: string[]): Record<string, string | undefined> => {
  const sortedIds = useMemo(() => [...new Set(traceIds)].sort(), [traceIds]);

  const { data } = useQuery({
    queryKey: [SPAN_LINK_TRACE_INFOS_QUERY_KEY, sortedIds],
    queryFn: () => fetchBatchTraceInfosV3({ traceIds: sortedIds }),
    enabled: sortedIds.length > 0,
    refetchOnWindowFocus: false,
    retry: false,
    staleTime: Infinity,
  });

  return useMemo(() => {
    if (!data?.trace_infos) return {};

    const hrefMap: Record<string, string | undefined> = {};
    for (const info of data.trace_infos) {
      hrefMap[info.trace_id] = getTraceHref(info.trace_id, info);
    }
    return hrefMap;
  }, [data?.trace_infos]);
};
