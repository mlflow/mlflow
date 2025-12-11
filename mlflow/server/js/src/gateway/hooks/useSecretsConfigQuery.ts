import { useQuery } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';
import type { SecretsConfigResponse } from '../types';

type SecretsConfigQueryKey = ReturnType<typeof GatewayQueryKeys.secretsConfigQuery>;

/**
 * Hook to check if secrets/encryption is configured on the backend.
 * Returns whether the MLFLOW_CRYPTO_KEK_PASSPHRASE environment variable is set.
 */
export const useSecretsConfigQuery = () => {
  const queryResult = useQuery<SecretsConfigResponse, Error, SecretsConfigResponse, SecretsConfigQueryKey>(
    GatewayQueryKeys.secretsConfigQuery(),
    {
      queryFn: () => GatewayApi.getSecretsConfig(),
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
