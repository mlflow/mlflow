import { useEffect, useMemo, useRef } from 'react';
import { useMutation, useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { AccountApi } from './api';
import { isWorkspaceAdminRole } from './types';
import type { UpdatePasswordRequest } from './types';

export const useCurrentUserQuery = () => {
  return useQuery({
    queryKey: ['account_current_user'],
    queryFn: AccountApi.getCurrentUser,
    retry: false,
    refetchOnWindowFocus: false,
  });
};

export const useCurrentUserIsAdmin = () => {
  const { data } = useCurrentUserQuery();
  return Boolean(data?.user?.is_admin);
};

/**
 * The set of workspaces in which the current user holds a workspace-admin
 * role (``(workspace, *, MANAGE)``). Mirrors the backend
 * ``store.list_workspace_admin_workspaces``. Composes self-authorized
 * queries — no admin-only endpoint required.
 *
 * ``WorkspaceSelector`` blanket-invalidates the query cache after a
 * workspace switch, so this hook can briefly observe ``rolesData ===
 * undefined`` while the refetch is in-flight. Without protection that
 * would flip ``canManage`` to false in the sidebar and flicker the
 * Manage entry off and back on. We hold the last computed set in a ref
 * and reuse it whenever ``rolesData`` is missing, so the UI gate stays
 * stable across refetches.
 */
export const useCurrentUserAdminWorkspaces = (): Set<string> => {
  const { data: currentUser } = useCurrentUserQuery();
  const username = currentUser?.user?.username ?? '';
  const { data: rolesData } = useUserRolesQuery(username);

  const computed = useMemo(() => {
    if (!rolesData) {
      return null;
    }
    const result = new Set<string>();
    for (const role of rolesData.roles ?? []) {
      if (isWorkspaceAdminRole(role)) {
        result.add(role.workspace);
      }
    }
    return result;
  }, [rolesData]);

  const previous = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (computed) {
      previous.current = computed;
    }
  }, [computed]);

  return computed ?? previous.current;
};

/**
 * True when the current user is a workspace admin in at least one
 * workspace. Thin wrapper over ``useCurrentUserAdminWorkspaces`` for
 * call sites that only need the boolean.
 */
export const useCurrentUserIsWorkspaceAdmin = (): boolean => {
  return useCurrentUserAdminWorkspaces().size > 0;
};

/**
 * True while ``/users/current`` is loading or returns a user; false on
 * 404/401 (auth not configured). Loading counts as available so
 * auth-gated UI doesn't flicker hidden while the request resolves.
 */
export const useIsAuthAvailable = () => {
  const { data, isError, isLoading } = useCurrentUserQuery();
  return !isError && (isLoading || Boolean(data?.user));
};

/**
 * True when the server uses the built-in Basic Auth handler. Custom
 * ``authorization_function`` deployments return false so callers can
 * hide Basic-Auth-only affordances (the logout XHR trick and the
 * change-password modal). Defaults to false until the response lands -
 * we'd rather hide a button briefly than expose a non-functional one.
 */
export const useIsBasicAuth = () => {
  const { data } = useCurrentUserQuery();
  return Boolean(data?.is_basic_auth);
};

export const AccountQueryKeys = {
  userRoles: (username: string) => ['account_user_roles', username] as const,
  myPermissions: ['account_my_permissions'] as const,
};

export const useUpdatePassword = () => {
  return useMutation({
    mutationFn: (request: UpdatePasswordRequest) => AccountApi.updatePassword(request),
  });
};

export const useUserRolesQuery = (username: string) => {
  return useQuery({
    queryKey: AccountQueryKeys.userRoles(username),
    queryFn: () => AccountApi.listUserRoles(username),
    retry: false,
    refetchOnWindowFocus: false,
    enabled: Boolean(username),
  });
};

/** Direct (non-role-derived) grants for the current user; no admin gate. */
export const useMyPermissionsQuery = () => {
  // Gate on a known username so we don't fire a guaranteed 401 on
  // unauthenticated / auth-disabled deployments.
  const { data } = useCurrentUserQuery();
  const username = data?.user?.username;
  return useQuery({
    queryKey: AccountQueryKeys.myPermissions,
    queryFn: AccountApi.listMyPermissions,
    retry: false,
    refetchOnWindowFocus: false,
    enabled: Boolean(username),
  });
};
