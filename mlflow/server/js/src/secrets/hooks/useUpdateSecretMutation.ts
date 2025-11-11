import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_SECRETS_QUERY_KEY } from '../constants';
import type { UpdateSecretRequest, Secret } from '../types';

export const useUpdateSecretMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: (secret: Secret) => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: updateSecret, isLoading } = useMutation<Secret, Error, UpdateSecretRequest>({
    mutationFn: async (request: UpdateSecretRequest) => {
      return await secretsApi.updateSecret(request);
    },
    onSuccess: (secret) => {
      queryClient.invalidateQueries({ queryKey: [LIST_SECRETS_QUERY_KEY] });
      onSuccess?.(secret);
    },
    onError: (error) => {
      onError?.(error);
    },
  });

  return {
    updateSecret,
    isLoading,
  };
};
