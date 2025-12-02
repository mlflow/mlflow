import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';
import type { UpdateModelDefinitionRequest } from '../types';

export const useUpdateModelDefinitionMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: UpdateModelDefinitionRequest) => GatewayApi.updateModelDefinition(request),
    onSuccess: () => {
      queryClient.invalidateQueries([GatewayQueryKeys.modelDefinitions]);
      queryClient.invalidateQueries([GatewayQueryKeys.endpoints]);
    },
  });
};
