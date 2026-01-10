import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { GET_DATASET_RECORDS_QUERY_KEY } from '../constants';

type DeleteDatasetRecordsPayload = {
  datasetId: string;
  datasetRecordIds: string[];
};

type DeleteDatasetRecordsResponse = {
  deleted_count: number;
};

export const useDeleteDatasetRecordsMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: (deletedCount: number) => void;
  onError?: (error: any) => void;
}) => {
  const queryClient = useQueryClient();

  const { mutate: deleteDatasetRecordsMutation, isLoading } = useMutation({
    mutationFn: async ({ datasetId, datasetRecordIds }: DeleteDatasetRecordsPayload) => {
      const requestBody = {
        dataset_record_ids: datasetRecordIds,
      };

      const response = (await fetchAPI(
        getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`),
        'DELETE',
        requestBody,
      )) as DeleteDatasetRecordsResponse;

      return response;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: [GET_DATASET_RECORDS_QUERY_KEY] });
      onSuccess?.(data.deleted_count);
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