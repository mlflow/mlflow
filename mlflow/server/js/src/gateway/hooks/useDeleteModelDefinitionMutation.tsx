import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';

export const useDeleteModelDefinitionMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelDefinitionId: string) => GatewayApi.deleteModelDefinition(modelDefinitionId),
    onSuccess: () => {
      queryClient.invalidateQueries([GatewayQueryKeys.modelDefinitions]);
      queryClient.invalidateQueries([GatewayQueryKeys.endpoints]);
    },
  });
};
