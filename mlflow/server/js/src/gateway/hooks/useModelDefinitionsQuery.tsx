import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';
import type { ListModelDefinitionsResponse } from '../types';

type ModelDefinitionsQueryKey = ReturnType<typeof GatewayQueryKeys.modelDefinitionsList>;

export const useModelDefinitionsQuery = () => {
  const queryResult = useQuery<
    ListModelDefinitionsResponse,
    Error,
    ListModelDefinitionsResponse,
    ModelDefinitionsQueryKey
  >(GatewayQueryKeys.modelDefinitionsList(), {
    queryFn: () => GatewayApi.listModelDefinitions(),
    retry: false,
  });

  return {
    data: queryResult.data?.model_definitions,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
