import { useMutation, useQuery, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
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
