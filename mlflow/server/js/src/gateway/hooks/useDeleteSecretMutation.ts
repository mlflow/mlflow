import { useMutation, useQueryClient } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';

export const useDeleteSecretMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (secretId: string) => GatewayApi.deleteSecret(secretId),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_secrets']);
    },
  });
};
