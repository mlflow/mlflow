import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';

export const useModelDefinitionQuery = (modelDefinitionId: string) => {
  return useQuery([GatewayQueryKeys.modelDefinitions, modelDefinitionId], {
    queryFn: () => GatewayApi.getModelDefinition(modelDefinitionId),
    retry: false,
    enabled: Boolean(modelDefinitionId),
  });
};
