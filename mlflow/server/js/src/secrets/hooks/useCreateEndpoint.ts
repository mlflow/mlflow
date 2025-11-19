import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { endpointsApi_Legacy } from '../api/routesApi';
import type { CreateEndpointRequest_Legacy, CreateEndpointResponse_Legacy } from '../types';
import { LIST_ROUTES_QUERY_KEY } from '../constants';

export const useCreateEndpoint = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<CreateEndpointResponse_Legacy, Error, CreateEndpointRequest_Legacy>({
    mutationFn: async (request: CreateEndpointRequest_Legacy) => {
      return await endpointsApi_Legacy.createEndpoint(request);
    },
    onSuccess: () => {
      // Invalidate and refetch endpoints list after successful creation
      queryClient.invalidateQueries({ queryKey: [LIST_ROUTES_QUERY_KEY] });
    },
  });

  return {
    createEndpoint: mutate,
    createEndpointAsync: mutateAsync,
    isLoading,
    error,
  };
};
