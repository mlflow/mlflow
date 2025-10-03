import { postJson } from '@mlflow/mlflow/src/common/utils/FetchUtils';
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
      const response = (await postJson({
        relativeUrl: `ajax-api/3.0/mlflow/datasets/${datasetId}/records`,
        data: {
          dataset_id: datasetId,
          records: records,
        },
      })) as UpsertDatasetRecordsResponse;

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
