import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_BINDINGS_QUERY_KEY, LIST_SECRETS_QUERY_KEY } from '../constants';
import type { BindSecretRequest } from '../types';

export const useBindSecretMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: bindSecret, isLoading } = useMutation<void, Error, BindSecretRequest>({
    mutationFn: async (request: BindSecretRequest) => {
      await secretsApi.bindSecret(request);
    },
    onSuccess: () => {
      // Invalidate both bindings and secrets list to update binding counts
      queryClient.invalidateQueries({ queryKey: [LIST_BINDINGS_QUERY_KEY] });
      queryClient.invalidateQueries({ queryKey: [LIST_SECRETS_QUERY_KEY] });
      onSuccess?.();
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    bindSecret,
    isLoading,
  };
};
