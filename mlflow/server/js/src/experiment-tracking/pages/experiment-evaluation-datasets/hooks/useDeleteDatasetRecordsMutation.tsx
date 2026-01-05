import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GET_DATASET_RECORDS_QUERY_KEY, SEARCH_EVALUATION_DATASETS_QUERY_KEY } from '../constants';
import type { DeleteDatasetRecordsPayload, DeleteDatasetRecordsResponse } from '../types';

export const useDeleteDatasetRecordsMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: (data: DeleteDatasetRecordsResponse) => void;
  onError?: (error: any) => void;
}) => {
  const queryClient = useQueryClient();

  const { mutate: deleteDatasetRecordsMutation, isLoading } = useMutation({
    mutationFn: async ({ datasetId, datasetRecordIds }: DeleteDatasetRecordsPayload) => {
      const response = await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`), 'DELETE', {
        dataset_record_ids: datasetRecordIds,
      });
      return response as DeleteDatasetRecordsResponse;
    },
    onSuccess: (data, variables) => {
      // Invalidate the dataset records query to refresh the records list
      queryClient.invalidateQueries({ queryKey: [GET_DATASET_RECORDS_QUERY_KEY, variables.datasetId] });
      // Invalidate the datasets query to refresh the profile (record count)
      queryClient.invalidateQueries({ queryKey: [SEARCH_EVALUATION_DATASETS_QUERY_KEY] });
      onSuccess?.(data);
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    deleteDatasetRecordsMutation,
    isLoading,
  };
};
