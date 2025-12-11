import { useMutation, useQueryClient } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { CreateEndpointRequest, CreateEndpointResponse } from '../types';

export const useCreateEndpointMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CreateEndpointRequest) => GatewayApi.createEndpoint(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
    },
  });
};
