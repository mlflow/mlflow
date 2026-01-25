import { useQueries } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { CHECK_MULTITURN_DATASETS_QUERY_KEY } from '../constants';

/**
 * Hook to check if any of the selected datasets is a multiturn dataset.
 */
export const useCheckMultiturnDatasets = ({ datasetIds }: { datasetIds: string[] }) => {
  const results = useQueries({
    queries: datasetIds.map((datasetId) => ({
      queryKey: [CHECK_MULTITURN_DATASETS_QUERY_KEY, datasetId],
      queryFn: async () => {
        const queryParams = new URLSearchParams();
        queryParams.set('dataset_id', datasetId);
        queryParams.set('max_results', '1');

        const response = await fetchAPI(
          getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records?${queryParams.toString()}`),
          'GET',
        ).catch(() => null);
        if (!response) return false;

        const records = parseJSONSafe(response.records);
        if (records && records.length > 0) {
          const firstRecordInputs = records[0].inputs;
          if (firstRecordInputs && firstRecordInputs.goal) {
            return true;
          }
        }

        return false;
      },
      refetchOnWindowFocus: false,
    })),
  });
  const isLoading = results.some((r) => r.isLoading);
  const data = results.some((r) => r.data === true);

  return { isLoading, data };
};
