import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { EvaluationDataset } from '../types';
import { SEARCH_EVALUATION_DATASETS_QUERY_KEY } from '../constants';

type CreateDatasetResponse = {
  dataset: EvaluationDataset;
};

type CreateDatasetPayload = {
  datasetName: string;
  experimentIds?: string[];
};

export const useCreateEvaluationDatasetMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: any) => void;
}) => {
  const queryClient = useQueryClient();

  const { mutate: createEvaluationDatasetMutation, isLoading } = useMutation({
    mutationFn: async ({ datasetName, experimentIds }: CreateDatasetPayload) => {
      const requestBody = {
        name: datasetName,
        experiment_ids: experimentIds,
      };

      const response = (await fetchAPI(
        getAjaxUrl('ajax-api/3.0/mlflow/datasets/create'),
        'POST',
        requestBody,
      )) as CreateDatasetResponse;

      return response.dataset;
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
    createEvaluationDatasetMutation,
    isLoading,
  };
};
