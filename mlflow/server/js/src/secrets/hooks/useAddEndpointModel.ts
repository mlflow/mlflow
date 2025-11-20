import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_ROUTES_QUERY_KEY } from '../constants';
import type { AddEndpointModelRequest, AddEndpointModelResponse } from '../types';

export const useAddEndpointModel = ({
  onSuccess,
  onError,
}: {
  onSuccess?: (data: AddEndpointModelResponse) => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: addModel, isLoading } = useMutation<
    AddEndpointModelResponse,
    Error,
    AddEndpointModelRequest
  >({
    mutationFn: async (request: AddEndpointModelRequest) => {
      return await secretsApi.addEndpointModel(request);
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
    addModel,
    isLoading,
  };
};
