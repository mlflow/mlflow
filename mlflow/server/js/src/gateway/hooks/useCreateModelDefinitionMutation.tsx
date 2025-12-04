import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';
import type { CreateModelDefinitionRequest } from '../types';

export const useCreateModelDefinitionMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CreateModelDefinitionRequest) => GatewayApi.createModelDefinition(request),
    onSuccess: () => {
      queryClient.invalidateQueries([GatewayQueryKeys.modelDefinitions]);
    },
  });
};
