import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useMemo } from 'react';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

type EvaluationDataset = {
  dataset_id: string;
  name: string;
};

type SearchEvaluationDatasetsResponse = {
  datasets?: EvaluationDataset[];
  next_page_token?: string;
};

/**
 * Hook that fetches evaluation datasets for an experiment and provides a lookup
 * from dataset_id to dataset name. Results are cached for 5 minutes.
 */
export const useDatasetNamesLookup = ({ experimentId }: { experimentId: string }) => {
  const { data, isLoading } = useQuery<SearchEvaluationDatasetsResponse, Error>({
    queryKey: ['dataset_names_lookup', experimentId],
    queryFn: async () => {
      // Fetch all datasets for this experiment (no pagination needed for name lookup)
      return (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/datasets/search'), 'POST', {
        experiment_ids: [experimentId],
        max_results: 1000, // Get all datasets
      })) as SearchEvaluationDatasetsResponse;
    },
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    cacheTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    retry: false,
    refetchOnWindowFocus: false,
  });

  // Build a lookup map from dataset_id to name
  const datasetNameMap = useMemo(() => {
    const map = new Map<string, string>();
    if (data?.datasets) {
      for (const dataset of data.datasets) {
        map.set(dataset.dataset_id, dataset.name);
      }
    }
    return map;
  }, [data?.datasets]);

  const getDatasetName = (datasetId: string | undefined): string | undefined => {
    if (!datasetId) return undefined;
    return datasetNameMap.get(datasetId);
  };

  return {
    getDatasetName,
    isLoading,
    datasetNameMap,
  };
};
