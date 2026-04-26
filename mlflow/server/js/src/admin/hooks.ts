import { useMutation, useQuery, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getLocalStorageItem } from '../shared/web-shared/hooks/useLocalStorage';
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
    onSuccess: (_, request) => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.userRoles(request.username) });
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
