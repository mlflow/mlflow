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
 * Workspaces in which the current user holds ``(workspace, *, MANAGE)``.
 *
 * Holds the last computed set in a ref so the gate stays stable across
 * the cache invalidation ``WorkspaceSelector`` triggers on workspace
 * switch (without it the Manage entry flickers off and back on). Resets
 * on username change so a dev-user-switcher cookie swap can't briefly
 * serve the previous identity's set during the new user's first refetch.
 */
export const useCurrentUserAdminWorkspaces = (): Set<string> => {
  const { data: currentUser } = useCurrentUserQuery();
  const username = currentUser?.user?.username ?? '';
  // Platform admins short-circuit on ``is_admin`` everywhere; their
  // workspace-admin set is unused. Skip the fetch to save a request per page load.
  const { data: rolesData } = useUserRolesQuery(username, {
    enabled: !currentUser?.user?.is_admin,
  });

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
  const lastUsername = useRef(username);
  useEffect(() => {
    if (lastUsername.current !== username) {
      previous.current = new Set();
      lastUsername.current = username;
    }
    if (computed) {
      previous.current = computed;
    }
  }, [computed, username]);

  return computed ?? previous.current;
};

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

export const useUserRolesQuery = (username: string, options: { enabled?: boolean } = {}) => {
  return useQuery({
    queryKey: AccountQueryKeys.userRoles(username),
    queryFn: () => AccountApi.listUserRoles(username),
    retry: false,
    refetchOnWindowFocus: false,
    // Callers pass ``enabled: false`` to skip a fetch they know will 403.
    enabled: Boolean(username) && options.enabled !== false,
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
