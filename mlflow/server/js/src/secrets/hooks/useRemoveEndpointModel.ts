import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_ROUTES_QUERY_KEY } from '../constants';
import type { RemoveEndpointModelRequest, RemoveEndpointModelResponse } from '../types';

export const useRemoveEndpointModel = ({
  onSuccess,
  onError,
}: {
  onSuccess?: (data: RemoveEndpointModelResponse) => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: removeModel, isLoading } = useMutation<
    RemoveEndpointModelResponse,
    Error,
    RemoveEndpointModelRequest
  >({
    mutationFn: async (request: RemoveEndpointModelRequest) => {
      return await secretsApi.removeEndpointModel(request);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: [LIST_ROUTES_QUERY_KEY] });
      onSuccess?.(data);
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    removeModel,
    isLoading,
  };
};
