import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useCallback } from 'react';
import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';
import { GET_DATASET_RECORDS_QUERY_KEY, SEARCH_EVALUATION_DATASETS_QUERY_KEY } from '../constants';

type UpsertDatasetRecordsPayload = {
  datasetId: string;
  // JSON serialized list of dataset records
  records: string;
};

// Records are matched on their inputs: a record whose inputs already exist in the dataset
// updates that record and counts as updated rather than inserted.
type UpsertDatasetRecordsResponse = {
  inserted_count?: number;
  updated_count?: number;
};

export const useUpsertDatasetRecordsMutation = () => {
  const queryClient = useQueryClient();
  const { mutateAsync: upsertDatasetRecordsMutationAsync, isLoading } = useMutation({
    mutationFn: async ({ datasetId, records }: UpsertDatasetRecordsPayload) => {
      const requestBody = {
        dataset_id: datasetId,
        records: records,
      };

      const response = (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`), {
        method: 'POST',
        body: requestBody,
      })) as UpsertDatasetRecordsResponse;

      return response;
    },
  });

  const invalidateAfterUpsert = useCallback(
    (datasetIds: string[]) => {
      if (datasetIds.length === 0) {
        return;
      }
      // Search holds dataset.profile (record count); records is the open table.
      queryClient.invalidateQueries({ queryKey: [SEARCH_EVALUATION_DATASETS_QUERY_KEY] });
      for (const datasetId of datasetIds) {
        queryClient.invalidateQueries({ queryKey: [GET_DATASET_RECORDS_QUERY_KEY, datasetId] });
      }
    },
    [queryClient],
  );

  return {
    upsertDatasetRecordsMutationAsync,
    invalidateAfterUpsert,
    isLoading,
  };
};
