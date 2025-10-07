import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMutation } from '@tanstack/react-query';

type UpsertDatasetRecordsPayload = {
  datasetId: string;
  // JSON serialized list of dataset records
  records: string;
};

type UpsertDatasetRecordsResponse = {
  insertedCount: number;
  updatedCount: number;
};

export const useUpsertDatasetRecordsMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: any) => void;
}) => {
  const { mutate: upsertDatasetRecordsMutation, isLoading } = useMutation({
    mutationFn: async ({ datasetId, records }: UpsertDatasetRecordsPayload) => {
      const requestBody = {
        dataset_id: datasetId,
        records: records,
      };

      const response = (await fetchAPI(
        getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`),
        'POST',
        requestBody,
      )) as UpsertDatasetRecordsResponse;

      return response;
    },
    onSuccess: () => {
      onSuccess?.();
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    upsertDatasetRecordsMutation,
    isLoading,
  };
};
