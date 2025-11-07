import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { secretsApi } from '../api/secretsApi';
import { LIST_BINDINGS_QUERY_KEY } from '../constants';
import type { ListBindingsResponse } from '../types';

export const useListBindings = ({ secretId, enabled = true }: { secretId: string; enabled?: boolean }) => {
  const { data, isLoading, error, refetch } = useQuery<ListBindingsResponse, Error>({
    queryKey: [LIST_BINDINGS_QUERY_KEY, secretId],
    queryFn: async () => {
      return await secretsApi.listBindings({ secret_id: secretId });
    },
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && !!secretId,
  });

  return {
    bindings: data?.bindings ?? [],
    isLoading,
    error,
    refetch,
  };
};
