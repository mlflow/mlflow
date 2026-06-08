import { useQuery } from '@databricks/web-shared/query-client';

import { AdminApi } from '../../../../admin/api';
import type { ListUsersResponse } from '../../../../admin/types';

export const ASSIGNABLE_USERS_QUERY_KEY = 'REVIEW_QUEUE_ASSIGNABLE_USERS';

/**
 * Usernames assignable as review-queue targets, for the "Flag for review" user
 * search. Backed by the admin `users/list` endpoint, which is workspace-admin
 * gated server-side — so callers must pass `enabled` only when the current user
 * can list users (the request 403s otherwise). We intend to relax that ACL to
 * any workspace user; until then user search is admin-only.
 */
export const useAssignableUsersQuery = ({ enabled }: { enabled: boolean }) => {
  const { data, isLoading, error } = useQuery<ListUsersResponse, Error>({
    queryKey: [ASSIGNABLE_USERS_QUERY_KEY],
    queryFn: () => AdminApi.listUsers(),
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
  });

  return { users: data?.users ?? [], isLoading, error };
};
