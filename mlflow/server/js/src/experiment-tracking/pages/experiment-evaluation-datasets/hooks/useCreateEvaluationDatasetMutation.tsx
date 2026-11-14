import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { EvaluationDataset } from '../types';
import { SEARCH_EVALUATION_DATASETS_QUERY_KEY } from '../constants';
import { V2_DATASETS_PAGE_QUERY_KEY } from '../../experiment-evaluation-datasets-v2/hooks/useDatasetsPageQuery';

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

      const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/datasets/create'), {
        method: 'POST',
        body: requestBody,
      })) as CreateDatasetResponse;

      return response.dataset;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [SEARCH_EVALUATION_DATASETS_QUERY_KEY] });
      queryClient.invalidateQueries({ queryKey: [V2_DATASETS_PAGE_QUERY_KEY] });
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
