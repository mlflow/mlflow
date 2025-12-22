import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';

export const useDeleteSecret = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (secretId: string) => GatewayApi.deleteSecret(secretId),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_secrets']);
    },
  });
};
