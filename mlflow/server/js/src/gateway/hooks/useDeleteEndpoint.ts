import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';

export const useDeleteEndpoint = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (endpointId: string) => GatewayApi.deleteEndpoint(endpointId),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
      queryClient.invalidateQueries(['gateway_bindings']);
    },
  });
};
