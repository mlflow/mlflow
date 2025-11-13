import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_BINDINGS_QUERY_KEY } from '../constants';
import type { UnbindSecretRequest } from '../types';

export const useUnbindSecretMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: unbindSecret, isLoading } = useMutation<void, Error, UnbindSecretRequest>({
    mutationFn: async (request: UnbindSecretRequest) => {
      await secretsApi.unbindSecret(request);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [LIST_BINDINGS_QUERY_KEY] });
      onSuccess?.();
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    unbindSecret,
    isLoading,
  };
};
