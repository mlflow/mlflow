import { useQuery } from '@databricks/web-shared/query-client';

import { AdminApi } from '../../../../admin/api';
import type { ListUsersResponse } from '../../../../admin/types';
import { useActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';

export const ASSIGNABLE_USERS_QUERY_KEY = 'REVIEW_QUEUE_ASSIGNABLE_USERS';

// Stable fallback so a disabled/loading query returns the same array identity
// every render, avoiding render loops in consumers keyed on the list.
const EMPTY_USERS: ListUsersResponse['users'] = [];

/**
 * Usernames assignable as review-queue targets, via the admin `users/list`
 * endpoint. Any authenticated user may list users; that grants no access on its
 * own. Callers pass `enabled` to skip the request on a no-auth server.
 */
export const useAssignableUsersQuery = ({ enabled }: { enabled: boolean }) => {
  // The roster is workspace-scoped, so cache it per workspace.
  const workspace = useActiveWorkspace();
  const { data, isLoading, error } = useQuery<ListUsersResponse, Error>({
    queryKey: [ASSIGNABLE_USERS_QUERY_KEY, workspace ?? ''],
    queryFn: () => AdminApi.listUsers(workspace ?? undefined),
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
  });

  return { users: data?.users ?? EMPTY_USERS, isLoading, error };
};
