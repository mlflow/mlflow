import { useMemo } from 'react';
import { useQuery } from '../../common/utils/reactQueryHooks';
import { getAjaxUrl } from '../../common/utils/FetchUtils';
import { GatewayTraceTagKey } from '../../shared/web-shared/model-trace-explorer';

interface TraceInfo {
  trace_id: string;
  tags?: Record<string, string>;
}

interface SearchTracesResponse {
  traces?: TraceInfo[];
}

interface MlflowExperimentLocation {
  type: 'MLFLOW_EXPERIMENT';
  mlflow_experiment: {
    experiment_id: string;
  };
}

/**
 * Fetches traces from experiments and extracts filter options
 */
async function fetchTracesForFilterOptions(experimentIds: string[]): Promise<SearchTracesResponse> {
  // Convert experiment IDs to locations format required by the API
  const locations: MlflowExperimentLocation[] = experimentIds.map((id) => ({
    type: 'MLFLOW_EXPERIMENT',
    mlflow_experiment: {
      experiment_id: id,
    },
  }));

  const payload = {
    locations,
    max_results: 100,
  };

  const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch traces: ${response.statusText}`);
  }

  return response.json();
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
 * @param experimentIds - The experiment IDs to fetch traces from
 * @returns Filter options including providers and models
 */
export function useGatewayFilterOptions(experimentIds: string[]): UseGatewayFilterOptionsResult {
  const { data, isLoading, error } = useQuery({
    queryKey: ['gatewayFilterOptions', experimentIds],
    queryFn: async () => {
      if (experimentIds.length === 0) return { traces: [] };
      return fetchTracesForFilterOptions(experimentIds);
    },
    enabled: experimentIds.length > 0,
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

  console.log('providers', providers);
  console.log('models', models);

  return {
    providers,
    models,
    isLoading,
    error,
  };
}
