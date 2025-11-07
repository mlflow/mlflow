import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_SECRETS_QUERY_KEY } from '../constants';
import type { ListSecretsResponse } from '../types';

export const useListSecrets = ({ enabled = true }: { enabled?: boolean } = {}) => {
  const { data, isLoading, error, refetch } = useQuery<ListSecretsResponse, Error>({
    queryKey: [LIST_SECRETS_QUERY_KEY],
    queryFn: async () => {
      return await secretsApi.listSecrets();
    },
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
  });

  return {
    secrets: data?.secrets ?? [],
    isLoading,
    error,
    refetch,
  };
};
