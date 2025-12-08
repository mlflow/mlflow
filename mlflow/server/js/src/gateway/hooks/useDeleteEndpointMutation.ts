import { useMutation, useQueryClient } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';

export const useDeleteEndpointMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (endpointId: string) => GatewayApi.deleteEndpoint(endpointId),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
    },
  });
};
