import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { UpdateSecretRequest, UpdateSecretInfoResponse } from '../types';

export const useUpdateSecret = () => {
  return useMutation<UpdateSecretInfoResponse, Error, UpdateSecretRequest>({
    mutationFn: (request) => GatewayApi.updateSecret(request),
  });
};
