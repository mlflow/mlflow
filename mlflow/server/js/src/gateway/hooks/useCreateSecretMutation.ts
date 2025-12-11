import { useMutation, useQueryClient } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { CreateSecretRequest, CreateSecretResponse } from '../types';

export const useCreateSecretMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CreateSecretRequest) => GatewayApi.createSecret(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_secrets']);
    },
  });
};
