import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';
import type { SecretsConfigResponse } from '../types';

type SecretsConfigQueryKey = ReturnType<typeof GatewayQueryKeys.secretsConfigQuery>;

const queryFn = () => {
  return GatewayApi.getSecretsConfig();
};

export const useSecretsConfigQuery = () => {
  const queryResult = useQuery<SecretsConfigResponse, Error, SecretsConfigResponse, SecretsConfigQueryKey>(
    GatewayQueryKeys.secretsConfigQuery(),
    {
      queryFn,
      retry: false,
      staleTime: Infinity,
    },
  );

  return {
    secretsAvailable: queryResult.data?.secrets_available,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
  };
};
