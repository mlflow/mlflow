import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { endpointsApi_Legacy } from '../api/routesApi';
import { LIST_ROUTES_QUERY_KEY } from '../constants';

export const useDeleteEndpointMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: deleteEndpoint, isLoading } = useMutation<void, Error, string>({
    mutationFn: async (endpointId: string) => {
      await endpointsApi_Legacy.deleteEndpoint(endpointId);
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
    deleteEndpoint,
    isLoading,
  };
};
