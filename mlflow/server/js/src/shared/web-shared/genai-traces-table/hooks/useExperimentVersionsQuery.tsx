import { useQuery } from '@databricks/web-shared/query-client';

import { getAjaxUrl, makeRequest } from '../utils/FetchUtils';

interface LoggedModel {
  info: {
    model_id: string;
    experiment_id: string;
    name: string;
    creation_timestamp_ms?: number;
    last_updated_timestamp_ms?: number;
    artifact_uri?: string;
    status?: string;
    creator_id?: number;
    tags?: Array<{
      key: string;
      value: string;
    }>;
  };
  data: any;
}

interface UseExperimentVersionsQueryResponseType {
  models: LoggedModel[];
  next_page_token?: string;
}

export const useExperimentVersionsQuery = (
  experimentId: string,
  disabled = false,
): {
  data: LoggedModel[] | undefined;
  isLoading: boolean;
  error?: Error;
} => {
  const queryKey = ['EXPERIMENT_MODEL_VERSIONS', experimentId];

  const { data, isLoading, error } = useQuery<UseExperimentVersionsQueryResponseType, Error>({
    queryKey,
    queryFn: async () => {
      // Search for model versions related to this experiment
      const requestBody = {
        experiment_ids: [experimentId],
      };

      return makeRequest(getAjaxUrl('ajax-api/2.0/mlflow/logged-models/search'), 'POST', requestBody);
    },
    staleTime: Infinity,
    cacheTime: Infinity,
    enabled: !disabled,
    refetchOnMount: false,
    retry: false,
  });

  return {
    data: data?.models,
    isLoading,
    error: error || undefined,
  };
};
