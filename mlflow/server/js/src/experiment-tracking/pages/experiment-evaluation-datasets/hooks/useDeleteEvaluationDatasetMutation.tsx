import { deleteJson, postJson } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { EvaluationDataset } from '../types';
import { SEARCH_EVALUATION_DATASETS_QUERY_KEY } from '../constants';

export const useDeleteEvaluationDatasetMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: any) => void;
}) => {
  const queryClient = useQueryClient();

  const { mutate: deleteEvaluationDatasetMutation, isLoading } = useMutation({
    mutationFn: async ({ datasetId }: { datasetId: string }) => {
      await deleteJson({
        relativeUrl: `ajax-api/3.0/mlflow/datasets/${datasetId}`,
        data: {},
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [SEARCH_EVALUATION_DATASETS_QUERY_KEY] });
      onSuccess?.();
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    deleteEvaluationDatasetMutation,
    isLoading,
  };
};
