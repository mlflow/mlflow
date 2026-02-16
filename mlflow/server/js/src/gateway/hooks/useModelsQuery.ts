import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ModelsResponse } from '../types';

const queryFn = ({ queryKey }: QueryFunctionContext<ModelsQueryKey>) => {
  const [, { provider }] = queryKey;
  return GatewayApi.listModels(provider);
};

type ModelsQueryKey = ['gateway_models', { provider?: string }];

export const useModelsQuery = ({ provider }: { provider?: string } = {}) => {
  const queryResult = useQuery<ModelsResponse, Error, ModelsResponse, ModelsQueryKey>(
    ['gateway_models', { provider }],
    {
      queryFn,
      retry: false,
      enabled: provider !== undefined,
    },
  );

  return {
    data: queryResult.data?.models,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
