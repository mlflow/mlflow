import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchEndpoint } from '../../common/utils/FetchUtils';

interface UserInfo {
  id: number;
  username: string;
}

interface ListUsersResponse {
  users: UserInfo[];
}

const queryFn = () => {
  return fetchEndpoint({
    relativeUrl: 'ajax-api/2.0/mlflow/users/list',
    error: () => {},
  }) as Promise<ListUsersResponse>;
};

export const useUsersQuery = () => {
  const queryResult = useQuery<ListUsersResponse, Error>(['auth_users'], {
    queryFn,
    retry: false,
    refetchOnWindowFocus: false,
  });

  return {
    data: queryResult.data?.users ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
  };
};
