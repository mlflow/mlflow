import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { UpdateSecretRequest, UpdateSecretInfoResponse } from '../types';

export const useUpdateSecret = () => {
  const queryClient = useQueryClient();
  return useMutation<UpdateSecretInfoResponse, Error, UpdateSecretRequest>({
    mutationFn: (request) => GatewayApi.updateSecret(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_secrets']);
    },
  });
};
