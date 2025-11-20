import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_ROUTES_QUERY_KEY } from '../constants';
import type { UpdateEndpointRequest, UpdateEndpointResponse } from '../types';

export const useUpdateEndpoint = ({
  onSuccess,
  onError,
}: {
  onSuccess?: (data: UpdateEndpointResponse) => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: updateEndpoint, isLoading } = useMutation<
    UpdateEndpointResponse,
    Error,
    UpdateEndpointRequest
  >({
    mutationFn: async (request: UpdateEndpointRequest) => {
      return await secretsApi.updateEndpoint(request);
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
    updateEndpoint,
    isLoading,
  };
};
