import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListUsersResponse } from '../types';

export const useUsersQuery = () => {
  const queryResult = useQuery<ListUsersResponse, Error>(['auth_users'], {
    queryFn: GatewayApi.listUsers,
    retry: false,
    refetchOnWindowFocus: false,
  });

  return {
    data: queryResult.data?.users ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
  };
};
