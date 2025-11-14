import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { routesApi } from '../api/routesApi';
import { LIST_ROUTES_QUERY_KEY } from '../constants';

export const useDeleteRouteMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: deleteRoute, isLoading } = useMutation<void, Error, string>({
    mutationFn: async (routeId: string) => {
      await routesApi.deleteRoute(routeId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [LIST_ROUTES_QUERY_KEY] });
      onSuccess?.();
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    deleteRoute,
    isLoading,
  };
};
