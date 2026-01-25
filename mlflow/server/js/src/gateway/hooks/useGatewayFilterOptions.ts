import { useMemo } from 'react';
import { useQuery } from '../../common/utils/reactQueryHooks';
import { fetchOrFail, getAjaxUrl } from '../../common/utils/FetchUtils';
import { catchNetworkErrorIfExists } from '../../experiment-tracking/utils/NetworkUtils';
import { GatewayTraceTagKey } from '../../shared/web-shared/model-trace-explorer';

interface TraceInfo {
  trace_id: string;
  tags?: Record<string, string>;
}

interface SearchTracesResponse {
  traces?: TraceInfo[];
}

/**
 * Fetches traces from an experiment and extracts filter options
 */
async function fetchTracesForFilterOptions(experimentId: string): Promise<SearchTracesResponse> {
  const payload = {
    experiment_ids: [experimentId],
    max_results: 1000,
  };

  return fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}

export interface UseGatewayFilterOptionsResult {
  /** Available provider names */
  providers: string[];
  /** Available model names */
  models: string[];
  /** Whether data is loading */
  isLoading: boolean;
  /** Error if fetching failed */
  error: unknown;
}

/**
 * Hook that fetches gateway trace data and extracts unique provider and model values
 * for filter dropdowns.
 *
 * @param experimentId - The experiment ID to fetch traces from
 * @returns Filter options including providers and models
 */
export function useGatewayFilterOptions(experimentId: string | null): UseGatewayFilterOptionsResult {
  const { data, isLoading, error } = useQuery({
    queryKey: ['gatewayFilterOptions', experimentId],
    queryFn: async () => {
      if (!experimentId) return { traces: [] };
      return fetchTracesForFilterOptions(experimentId);
    },
    enabled: !!experimentId,
    refetchOnWindowFocus: false,
    staleTime: 60000, // Cache for 1 minute
  });

  const { providers, models } = useMemo(() => {
    const providerSet = new Set<string>();
    const modelSet = new Set<string>();

    if (data?.traces) {
      for (const trace of data.traces) {
        if (trace.tags) {
          const provider = trace.tags[GatewayTraceTagKey.PROVIDER];
          const model = trace.tags[GatewayTraceTagKey.MODEL];

          if (provider) {
            providerSet.add(provider);
          }
          if (model) {
            modelSet.add(model);
          }
        }
      }
    }

    return {
      providers: Array.from(providerSet),
      models: Array.from(modelSet),
    };
  }, [data?.traces]);

  return {
    providers,
    models,
    isLoading,
    error,
  };
}
