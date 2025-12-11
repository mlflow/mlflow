import type { QueryFunctionContext } from '../../common/utils/reactQueryHooks';
import { useQuery } from '../../common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListSecretsResponse } from '../types';

const queryFn = ({ queryKey }: QueryFunctionContext<SecretsQueryKey>) => {
  const [, { provider }] = queryKey;
  return GatewayApi.listSecrets(provider);
};

type SecretsQueryKey = ['gateway_secrets', { provider?: string }];

export const useSecretsQuery = ({ provider }: { provider?: string } = {}) => {
  const queryResult = useQuery<ListSecretsResponse, Error, ListSecretsResponse, SecretsQueryKey>(
    ['gateway_secrets', { provider }],
    {
      queryFn,
      retry: false,
    },
  );

  return {
    data: queryResult.data?.secrets,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
