import { isNil } from 'lodash';

import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { useQuery } from '@databricks/web-shared/query-client';

export function useGetTrace(
  getTrace: ((traceId: string) => Promise<ModelTrace | undefined>) | undefined,
  traceId?: string,
) {
  return useQuery({
    queryKey: ['getTrace', traceId],
    queryFn: () => (getTrace && traceId ? getTrace(traceId) : Promise.resolve(undefined)),
    enabled: !isNil(getTrace) && !isNil(traceId),
    staleTime: Infinity, // Keep data fresh as long as the component is mounted
    refetchOnWindowFocus: false, // Disable refetching on window focus
    retry: 1,
    keepPreviousData: true,
  });
}
