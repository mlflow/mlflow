import { isNil } from 'lodash';

import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { useQuery } from '@databricks/web-shared/query-client';

export function useGetTrace(
  getTrace: ((requestId?: string, traceId?: string) => Promise<ModelTrace | undefined>) | undefined,
  requestId?: string,
  traceId?: string,
) {
  return useQuery({
    queryKey: ['getTrace', requestId, traceId],
    queryFn: () => (getTrace ? getTrace(requestId, traceId) : Promise.resolve(undefined)),
    enabled: !isNil(getTrace) && (!isNil(requestId) || !isNil(traceId)),
    staleTime: Infinity, // Keep data fresh as long as the component is mounted
    refetchOnWindowFocus: false, // Disable refetching on window focus
    retry: 1,
    keepPreviousData: true,
  });
}
