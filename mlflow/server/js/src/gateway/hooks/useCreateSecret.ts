import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { CreateSecretRequest, CreateSecretInfoResponse } from '../types';

export const useCreateSecret = () => {
  const queryClient = useQueryClient();

  return useMutation<CreateSecretInfoResponse, Error, CreateSecretRequest>({
    mutationFn: (request) => GatewayApi.createSecret(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_secrets']);
    },
  });
};
