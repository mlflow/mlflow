import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { endpointsApi_Legacy } from '../api/routesApi';
import type { UpdateEndpointRequest_Legacy } from '../types';
import { LIST_ROUTES_QUERY_KEY, LIST_SECRETS_QUERY_KEY } from '../constants';

export const useUpdateEndpoint = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation({
    mutationFn: async (request: UpdateEndpointRequest_Legacy) => {
      return await endpointsApi_Legacy.updateEndpoint(request);
    },
    onSuccess: () => {
      // Invalidate endpoints list to refetch
      queryClient.invalidateQueries({ queryKey: [LIST_ROUTES_QUERY_KEY] });
      // Also invalidate secrets list since we might have created a new secret
      queryClient.invalidateQueries({ queryKey: [LIST_SECRETS_QUERY_KEY] });
    },
  });

  return {
    updateEndpoint: mutate,
    updateEndpointAsync: mutateAsync,
    isLoading,
    error,
  };
};
