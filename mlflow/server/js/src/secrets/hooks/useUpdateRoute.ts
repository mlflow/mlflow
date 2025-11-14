import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { routesApi } from '../api/routesApi';
import type { UpdateRouteRequest } from '../types';

const LIST_ROUTES_QUERY_KEY = 'listRoutes';
const LIST_SECRETS_QUERY_KEY = 'listSecrets';

export const useUpdateRoute = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation({
    mutationFn: async (request: UpdateRouteRequest) => {
      return await routesApi.updateRoute(request);
    },
    onSuccess: () => {
      // Invalidate routes list to refetch
      queryClient.invalidateQueries({ queryKey: [LIST_ROUTES_QUERY_KEY] });
      // Also invalidate secrets list since we might have created a new secret
      queryClient.invalidateQueries({ queryKey: [LIST_SECRETS_QUERY_KEY] });
    },
  });

  return {
    updateRoute: mutate,
    updateRouteAsync: mutateAsync,
    isLoading,
    error,
  };
};
