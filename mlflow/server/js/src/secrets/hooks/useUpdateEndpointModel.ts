import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_ROUTES_QUERY_KEY } from '../constants';
import type { UpdateEndpointModelRequest, UpdateEndpointModelResponse } from '../types';

export const useUpdateEndpointModel = ({
  onSuccess,
  onError,
}: {
  onSuccess?: (data: UpdateEndpointModelResponse) => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: updateModel, isLoading } = useMutation<
    UpdateEndpointModelResponse,
    Error,
    UpdateEndpointModelRequest
  >({
    mutationFn: async (request: UpdateEndpointModelRequest) => {
      return await secretsApi.updateEndpointModel(request);
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
    updateModel,
    isLoading,
  };
};
