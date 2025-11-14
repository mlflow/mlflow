import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_SECRETS_QUERY_KEY, LIST_ROUTES_QUERY_KEY } from '../constants';
import type { DeleteSecretRequest } from '../types';

export const useDeleteSecretMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: deleteSecret, isLoading } = useMutation<void, Error, DeleteSecretRequest>({
    mutationFn: async (request: DeleteSecretRequest) => {
      await secretsApi.deleteSecret(request);
    },
    onSuccess: () => {
      // Invalidate both secrets and routes since delete cascades
      queryClient.invalidateQueries({ queryKey: [LIST_SECRETS_QUERY_KEY] });
      queryClient.invalidateQueries({ queryKey: [LIST_ROUTES_QUERY_KEY] });
      onSuccess?.();
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    deleteSecret,
    isLoading,
  };
};
