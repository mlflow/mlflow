import { useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getLocalStorageItem } from '../shared/web-shared/hooks/useLocalStorage';
import { useSearchParams } from '../common/utils/RoutingUtils';
import { SETTINGS_RETURN_TO_PARAM } from '../settings/settingsSectionConstants';
import { AdminApi } from './api';
import type {
  CreateRoleRequest,
  UpdateRoleRequest,
  AddPermissionRequest,
  UpdatePermissionRequest,
  CreateUserRequest,
  UpdatePasswordRequest,
  UpdateAdminRequest,
} from './types';

/**
 * Opt-in dev flag to render the DevUserSwitcher (bottom-right floating toolbar).
 *
 * Two gates must both be true:
 *   1. Build is development (``process.env['NODE_ENV'] === 'development'``).
 *      The DevUserSwitcher stores plaintext passwords in localStorage and sets
 *      an ``Authorization`` cookie — gating at build time prevents it from
 *      being shipped in production bundles even if the localStorage flag is
 *      somehow set.
 *   2. The localStorage flag is set:
 *        localStorage.setItem('mlflow.settings.admin.enable-dev-user-switcher_v1', 'true')
 *      Mirrors the ``mlflow.settings.telemetry.enable-dev-logging`` pattern.
 */
export const ADMIN_ENABLE_DEV_USER_SWITCHER_STORAGE_KEY = 'mlflow.settings.admin.enable-dev-user-switcher';

export const DEV_USER_SWITCHER_ENABLED: boolean =
  process.env['NODE_ENV'] === 'development' &&
  getLocalStorageItem(ADMIN_ENABLE_DEV_USER_SWITCHER_STORAGE_KEY, 1, false, false);

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
 * Returns a helper that appends the current ``returnTo`` query param (used by
 * the Settings sub-sidebar exit link) to a destination path. When users
 * navigate into Admin from Settings, internal Admin links must preserve
 * ``returnTo`` so the Settings exit link still goes back to the originating
 * page after they drill into a user or role detail.
 */
export const useWithSettingsReturnTo = () => {
  const [searchParams] = useSearchParams();
  const returnTo = searchParams.get(SETTINGS_RETURN_TO_PARAM);
  return useCallback(
    (path: string) => {
      if (!returnTo) {
        return path;
      }
      const separator = path.includes('?') ? '&' : '?';
      return `${path}${separator}${SETTINGS_RETURN_TO_PARAM}=${encodeURIComponent(returnTo)}`;
    },
    [returnTo],
  );
};

/**
 * Returns `true` when auth is available or while ``/users/current`` is still
 * loading. The endpoint is registered only when MLflow is started with the
 * basic-auth app, so a 404/401 from the query indicates auth isn't configured
 * (or the user isn't logged in). Treat the initial loading state as
 * "available" so auth-gated UI (sidebar Admin / Account links, settings
 * sub-items) doesn't briefly flicker hidden before the current-user request
 * resolves.
 */
export const useIsAuthAvailable = () => {
  const { data, isError, isLoading } = useCurrentUserQuery();
  return !isError && (isLoading || Boolean(data?.user));
};

export const AdminQueryKeys = {
  users: ['admin_users'] as const,
  roles: ['admin_roles'] as const,
  roleDetail: (roleId: number) => ['admin_role', roleId] as const,
  rolePermissions: (roleId: number) => ['admin_role_permissions', roleId] as const,
  roleUsers: (roleId: number) => ['admin_role_users', roleId] as const,
  userRoles: (username: string) => ['admin_user_roles', username] as const,
};

// User queries and mutations
export const useUsersQuery = () => {
  return useQuery({
    queryKey: AdminQueryKeys.users,
    queryFn: AdminApi.listUsers,
    retry: false,
    refetchOnWindowFocus: false,
  });
};

export const useCreateUser = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateUserRequest) => AdminApi.createUser(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
    },
  });
};

export const useDeleteUser = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (username: string) => AdminApi.deleteUser(username),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
    },
  });
};

export const useUpdateAdmin = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: UpdateAdminRequest) => AdminApi.updateAdmin(request),
    // Toggling ``is_admin`` doesn't change a user's role assignments —
    // only the user list (which renders the admin badge) needs refreshing.
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
    },
  });
};

export const useUpdatePassword = () => {
  return useMutation({
    mutationFn: (request: UpdatePasswordRequest) => AdminApi.updatePassword(request),
  });
};

// Role queries and mutations
export const useRolesQuery = (workspace?: string) => {
  return useQuery({
    queryKey: [...AdminQueryKeys.roles, workspace],
    queryFn: () => AdminApi.listRoles(workspace),
    retry: false,
    refetchOnWindowFocus: false,
  });
};

export const useRoleDetailQuery = (roleId: number) => {
  return useQuery({
    queryKey: AdminQueryKeys.roleDetail(roleId),
    queryFn: () => AdminApi.getRole(roleId),
    retry: false,
    refetchOnWindowFocus: false,
    enabled: Number.isFinite(roleId),
  });
};

export const useCreateRole = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateRoleRequest) => AdminApi.createRole(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roles });
    },
  });
};

export const useUpdateRole = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: UpdateRoleRequest) => AdminApi.updateRole(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roles });
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleDetail(roleId) });
    },
  });
};

export const useDeleteRole = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (roleId: number) => AdminApi.deleteRole(roleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roles });
    },
  });
};

// Permission mutations
export const useAddPermission = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: AddPermissionRequest) => AdminApi.addPermission(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleDetail(roleId) });
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.rolePermissions(roleId) });
    },
  });
};

export const useRemovePermission = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (permissionId: number) => AdminApi.removePermission(permissionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleDetail(roleId) });
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.rolePermissions(roleId) });
    },
  });
};

export const useUpdatePermission = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: UpdatePermissionRequest) => AdminApi.updatePermission(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleDetail(roleId) });
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.rolePermissions(roleId) });
    },
  });
};

// Role assignment mutations
export const useAssignRole = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (username: string) => AdminApi.assignRole(username, roleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleUsers(roleId) });
    },
  });
};

export const useUnassignRole = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (username: string) => AdminApi.unassignRole(username, roleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleUsers(roleId) });
    },
  });
};

export const useRoleUsersQuery = (roleId: number) => {
  return useQuery({
    queryKey: AdminQueryKeys.roleUsers(roleId),
    queryFn: () => AdminApi.listRoleUsers(roleId),
    retry: false,
    refetchOnWindowFocus: false,
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

/**
 * Grant a per-user permission via the legacy per-resource CRUD endpoints.
 * Single mutation hook that dispatches based on `resource_type`. Bridges to
 * the synthetic-role mechanism after Phase 2 (#22855) — same API call,
 * different storage on the server side.
 */
export const useGrantUserPermission = () => {
  return useMutation({
    mutationFn: (request: { resource_type: string; resource_id: string; username: string; permission: string }) =>
      AdminApi.grantUserPermission(request.resource_type, request.resource_id, request.username, request.permission),
  });
};
