import { useQuery } from '@databricks/web-shared/query-client';

import { AdminApi } from '../../../../admin/api';
import type { ListUsersResponse } from '../../../../admin/types';

export const ASSIGNABLE_USERS_QUERY_KEY = 'REVIEW_QUEUE_ASSIGNABLE_USERS';

// Stable fallback so a disabled/loading query (e.g. a no-auth server, or before
// the roster loads) returns the same array identity every render. A fresh `[]`
// here would give consumers a new reference each render, looping any
// `useMemo`/`useEffect` keyed on the list (see QueueSettingsModal's members typeahead).
const EMPTY_USERS: ListUsersResponse['users'] = [];

/**
 * Usernames assignable as review-queue targets, for the "Flag for review" user
 * search. Backed by the admin `users/list` endpoint. Any authenticated user may
 * list users (the basic-auth ACL allows it); listing grants no access on its own
 * — assigning a user to a queue still requires experiment MANAGE. Callers still
 * pass `enabled` to skip the request on a no-auth server.
 */
export const useAssignableUsersQuery = ({ enabled }: { enabled: boolean }) => {
  const { data, isLoading, error } = useQuery<ListUsersResponse, Error>({
    queryKey: [ASSIGNABLE_USERS_QUERY_KEY],
    queryFn: () => AdminApi.listUsers(),
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
  });

  return { users: data?.users ?? EMPTY_USERS, isLoading, error };
};
