import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_BINDINGS_QUERY_KEY, LIST_SECRETS_QUERY_KEY } from '../constants';
import type { SecretBinding } from '../types';

export const useUnbindSecretMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: unbindSecret, isLoading } = useMutation<void, Error, SecretBinding>({
    mutationFn: async (binding: SecretBinding) => {
      await secretsApi.unbindSecret({
        resource_type: binding.resource_type,
        resource_id: binding.resource_id,
        field_name: binding.field_name,
      });
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
    unbindSecret,
    isLoading,
  };
};
