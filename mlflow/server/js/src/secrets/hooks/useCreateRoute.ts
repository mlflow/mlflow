import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { routesApi } from '../api/routesApi';
import type { CreateRouteRequest, CreateRouteResponse } from '../types';

const LIST_ROUTES_QUERY_KEY = 'listRoutes';

export const useCreateRoute = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<
    CreateRouteResponse,
    Error,
    CreateRouteRequest
  >({
    mutationFn: async (request: CreateRouteRequest) => {
      return await routesApi.createRoute(request);
    },
    onSuccess: () => {
      // Invalidate and refetch routes list after successful creation
      queryClient.invalidateQueries({ queryKey: [LIST_ROUTES_QUERY_KEY] });
    },
  });

  return {
    createRoute: mutate,
    createRouteAsync: mutateAsync,
    isLoading,
    error,
  };
};
