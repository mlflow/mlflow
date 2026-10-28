import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListUsersResponse } from '../types';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

export const useUsersQuery = () => {
  // The roster is workspace-scoped, so cache it per workspace.
  const workspace = useActiveWorkspace();
  const queryResult = useQuery<ListUsersResponse, Error>(['auth_users', workspace ?? ''], {
    queryFn: () => GatewayApi.listUsers(workspace ?? undefined),
    retry: false,
    refetchOnWindowFocus: false,
  });

  return {
    data: queryResult.data?.users ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
  };
};
