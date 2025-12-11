import { useMutation } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { UpdateSecretRequest, UpdateSecretResponse } from '../types';

export const useUpdateSecretMutation = () => {
  return useMutation<UpdateSecretResponse, Error, UpdateSecretRequest>({
    mutationFn: (request) => GatewayApi.updateSecret(request),
  });
};
