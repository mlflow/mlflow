import { useQuery } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { CHECK_MULTITURN_DATASETS_QUERY_KEY } from '../constants';

/**
 * Hook to check if any of the selected datasets is a multiturn dataset.
 */
export const useCheckMultiturnDatasets = ({ datasetIds }: { datasetIds: string[] }) => {
  return useQuery({
    queryKey: [CHECK_MULTITURN_DATASETS_QUERY_KEY, datasetIds],
    queryFn: async () => {
      let hasMultiturnDataset = false;
      if (datasetIds.length === 0) {
        return hasMultiturnDataset;
      }

      await Promise.all(
        datasetIds.map(async (datasetId) => {
          const queryParams = new URLSearchParams();
          queryParams.set('dataset_id', datasetId);
          queryParams.set('max_results', '1');

          const response = await fetchAPI(
            getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records?${queryParams.toString()}`),
            'GET',
          ).catch(() => null);
          if (!response) return;

          const records = parseJSONSafe(response.records);
          if (records && records.length > 0) {
            const firstRecordInputs = records[0].inputs;
            if (firstRecordInputs && firstRecordInputs.goal) {
              hasMultiturnDataset = true;
            }
          }
        }),
      );

      return hasMultiturnDataset;
    },
    enabled: datasetIds.length > 0,
    refetchOnWindowFocus: false,
  });
};
