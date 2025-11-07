import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_SECRETS_QUERY_KEY } from '../constants';
import type { CreateSecretRequest, CreateSecretResponse } from '../types';

export const useCreateSecretMutation = ({
  onSuccess,
  onError,
}: {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
} = {}) => {
  const queryClient = useQueryClient();

  const { mutate: createSecret, isLoading } = useMutation({
    mutationFn: (request: CreateSecretRequest) => secretsApi.createSecret(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [LIST_SECRETS_QUERY_KEY] });
      onSuccess?.();
    },
    onError: (error: Error) => {
      onError?.(error);
    },
  });

  return {
    createSecret,
    isLoading,
  };
};
