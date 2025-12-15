import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ProviderConfig } from '../types';

const queryFn = ({ queryKey }: QueryFunctionContext<ProviderConfigQueryKey>) => {
  const [, { provider }] = queryKey;
  return GatewayApi.getProviderConfig(provider);
};

type ProviderConfigQueryKey = ['gateway_provider_config', { provider: string }];

export const useProviderConfigQuery = ({ provider }: { provider: string }) => {
  const queryResult = useQuery<ProviderConfig, Error, ProviderConfig, ProviderConfigQueryKey>(
    ['gateway_provider_config', { provider }],
    {
      queryFn,
      retry: false,
      enabled: Boolean(provider),
    },
  );

  return {
    data: queryResult.data,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
