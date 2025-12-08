import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { DetachModelFromEndpointRequest } from '../types';

/**
 * Mutation to detach a model definition from an endpoint (removes the mapping/linkage).
 * This does NOT delete the model definition itself.
 */
export const useDetachModelFromEndpointMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: DetachModelFromEndpointRequest) => GatewayApi.detachModelFromEndpoint(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
    },
  });
};

// Legacy export - now detaches the mapping rather than deleting the model
export const useDeleteEndpointModelMutation = useDetachModelFromEndpointMutation;
