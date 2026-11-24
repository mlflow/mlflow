import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { SetEndpointTagRequest } from '../types';

export const useSetEndpointTagMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: SetEndpointTagRequest) => GatewayApi.setEndpointTag(request),
    onSuccess: () => {
      queryClient.invalidateQueries(['gateway_endpoints']);
    },
  });
};
