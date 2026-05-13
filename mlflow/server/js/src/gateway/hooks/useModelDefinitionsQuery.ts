import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListModelDefinitionsResponse } from '../types';

type ModelDefinitionsQueryKey = ['gateway_model_definitions'];

export const useModelDefinitionsQuery = () => {
  const queryResult = useQuery<
    ListModelDefinitionsResponse,
    Error,
    ListModelDefinitionsResponse,
    ModelDefinitionsQueryKey
  >(['gateway_model_definitions'], {
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
