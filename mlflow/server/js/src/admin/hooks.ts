import { useMutation, useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { AdminApi } from './api';
import type { UpdatePasswordRequest } from './types';

/**
 * Fetches the currently authenticated user's identity (id, username, is_admin)
 * from the backend. Works with HTTP Basic Auth since the backend reads
 * request.authorization.username; no identifying cookie needed.
 */
export const useCurrentUserQuery = () => {
  return useQuery({
    queryKey: ['admin_current_user'],
    queryFn: AdminApi.getCurrentUser,
    retry: false,
    refetchOnWindowFocus: false,
  });
};

export const useCurrentUserIsAdmin = () => {
  const { data } = useCurrentUserQuery();
  return Boolean(data?.user?.is_admin);
};

/**
 * Returns `true` when auth is available or while ``/users/current`` is still
 * loading. The endpoint is registered only when MLflow is started with the
 * basic-auth app, so a 404/401 from the query indicates auth isn't configured
 * (or the user isn't logged in). Treat the initial loading state as
 * "available" so auth-gated UI doesn't briefly flicker hidden before the
 * current-user request resolves.
 */
export const useIsAuthAvailable = () => {
  const { data, isError, isLoading } = useCurrentUserQuery();
  return !isError && (isLoading || Boolean(data?.user));
};

export const AdminQueryKeys = {
  userRoles: (username: string) => ['admin_user_roles', username] as const,
};

export const useUpdatePassword = () => {
  return useMutation({
    mutationFn: (request: UpdatePasswordRequest) => AdminApi.updatePassword(request),
  });
};

export const useUserRolesQuery = (username: string) => {
  return useQuery({
    queryKey: AdminQueryKeys.userRoles(username),
    queryFn: () => AdminApi.listUserRoles(username),
    retry: false,
    refetchOnWindowFocus: false,
    enabled: Boolean(username),
  });
};
