import { useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useSearchParams } from '../common/utils/RoutingUtils';
import { SETTINGS_RETURN_TO_PARAM } from '../settings/settingsSectionConstants';
import { AccountQueryKeys } from '../account/hooks';
import { AdminApi } from './api';
import type {
  AddPermissionRequest,
  CreateRoleRequest,
  CreateUserRequest,
  ResourceOption,
  UpdateAdminRequest,
  UpdateRoleRequest,
} from './types';

// Re-export account-side hooks so admin pages can keep importing them via
// ``./hooks``. The canonical home is account/hooks.
export {
  useCurrentUserIsAdmin,
  useCurrentUserQuery,
  useIsAuthAvailable,
  useIsBasicAuth,
  useMyPermissionsQuery,
  useUpdatePassword,
  useUserRolesQuery,
} from '../account/hooks';

/**
 * Appends the current ``returnTo`` query param to a destination path so
 * Admin links navigated to from Settings preserve the Settings exit
 * target after drilling into a user or role detail.
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

export const AdminQueryKeys = {
  users: ['admin_users'] as const,
  roles: ['admin_roles'] as const,
  roleDetail: (roleId: number) => ['admin_role', roleId] as const,
  roleUsers: (roleId: number) => ['admin_role_users', roleId] as const,
  resourceOptions: (resourceType: string) => ['admin_resource_options', resourceType] as const,
  userPermissions: (username: string) => ['admin_user_permissions', username] as const,
};

/** Direct (non-role-derived) grants for an arbitrary user. Admin / self / WP-admin-of-target. */
export const useUserPermissionsQuery = (username: string) => {
  return useQuery({
    queryKey: AdminQueryKeys.userPermissions(username),
    queryFn: () => AdminApi.listUserPermissions(username),
    enabled: Boolean(username),
    retry: false,
    refetchOnWindowFocus: false,
  });
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
    // Toggling ``is_admin`` doesn't change a user's role assignments -
    // only the user list (which renders the admin badge) needs refreshing.
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
    },
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

// Permission mutations. Permissions are returned inline as part of
// ``RoleDetailQuery``, so the role-detail key is the only invalidation
// target.
export const useAddPermission = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: AddPermissionRequest) => AdminApi.addPermission(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleDetail(roleId) });
    },
  });
};

export const useRemovePermission = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (permissionId: number) => AdminApi.removePermission(permissionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleDetail(roleId) });
    },
  });
};

// Role assignment mutations
export const useAssignRole = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (username: string) => AdminApi.assignRole(username, roleId),
    onSuccess: (_data, username) => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleUsers(roleId) });
      // The Admin Users tab renders a per-user roles cell from
      // ``useUserRolesQuery(username)``; refetch it so the new role shows up
      // immediately after the modal closes.
      queryClient.invalidateQueries({ queryKey: AccountQueryKeys.userRoles(username) });
    },
  });
};

export const useUnassignRole = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (username: string) => AdminApi.unassignRole(username, roleId),
    onSuccess: (_data, username) => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleUsers(roleId) });
      queryClient.invalidateQueries({ queryKey: AccountQueryKeys.userRoles(username) });
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

/**
 * Grant a per-user permission via the legacy per-resource CRUD endpoints.
 * Single mutation hook that dispatches based on `resource_type`. Bridges to
 * the synthetic-role mechanism after Phase 2 (#22855) - same API call,
 * different storage on the server side.
 */
export const useGrantUserPermission = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: { resource_type: string; resource_id: string; username: string; permission: string }) =>
      AdminApi.grantUserPermission(request.resource_type, request.resource_id, request.username, request.permission),
    onSuccess: (_data, variables) => {
      // Refresh the per-user roles cell (Admin Users tab + Account page)
      // and the per-user direct-permissions list (UserDetailPage).
      queryClient.invalidateQueries({ queryKey: AccountQueryKeys.userRoles(variables.username) });
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.userPermissions(variables.username) });
    },
  });
};

/** Revoke a per-user direct permission. Mirrors ``useGrantUserPermission``. */
export const useRevokeUserPermission = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: { resource_type: string; resource_id: string; username: string }) =>
      AdminApi.revokeUserPermission(request.resource_type, request.resource_id, request.username),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: AccountQueryKeys.userRoles(variables.username) });
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.userPermissions(variables.username) });
    },
  });
};

/**
 * Lookup options for the resource-id picker in the per-user grant form.
 * Each branch hits a different list endpoint; only the active branch is
 * fetched (others are gated via ``enabled``). Returns a uniform
 * ``ResourceOption[]`` so the consumer doesn't have to switch on shape.
 */
export const useResourceOptionsQuery = (resourceType: string) => {
  const experiments = useQuery({
    queryKey: AdminQueryKeys.resourceOptions('experiment'),
    queryFn: AdminApi.listExperimentsLite,
    enabled: resourceType === 'experiment',
    retry: false,
    refetchOnWindowFocus: false,
  });
  const registeredModels = useQuery({
    queryKey: AdminQueryKeys.resourceOptions('registered_model'),
    queryFn: AdminApi.listRegisteredModelsLite,
    enabled: resourceType === 'registered_model',
    retry: false,
    refetchOnWindowFocus: false,
  });
  const gatewaySecrets = useQuery({
    queryKey: AdminQueryKeys.resourceOptions('gateway_secret'),
    queryFn: AdminApi.listGatewaySecretsLite,
    enabled: resourceType === 'gateway_secret',
    retry: false,
    refetchOnWindowFocus: false,
  });
  const gatewayEndpoints = useQuery({
    queryKey: AdminQueryKeys.resourceOptions('gateway_endpoint'),
    queryFn: AdminApi.listGatewayEndpointsLite,
    enabled: resourceType === 'gateway_endpoint',
    retry: false,
    refetchOnWindowFocus: false,
  });
  const gatewayModelDefinitions = useQuery({
    queryKey: AdminQueryKeys.resourceOptions('gateway_model_definition'),
    queryFn: AdminApi.listGatewayModelDefinitionsLite,
    enabled: resourceType === 'gateway_model_definition',
    retry: false,
    refetchOnWindowFocus: false,
  });

  let options: ResourceOption[] = [];
  let isLoading = false;
  let error: unknown = null;
  switch (resourceType) {
    case 'experiment':
      options = (experiments.data?.experiments ?? []).map((e) => ({ id: e.experiment_id, name: e.name }));
      ({ isLoading, error } = experiments);
      break;
    case 'registered_model':
      // Registered models are identified by name only (no separate id), so
      // ``id`` and ``name`` collapse to the same string.
      options = (registeredModels.data?.registered_models ?? []).map((m) => ({ id: m.name, name: m.name }));
      ({ isLoading, error } = registeredModels);
      break;
    case 'gateway_secret':
      options = (gatewaySecrets.data?.secrets ?? []).map((s) => ({ id: s.secret_id, name: s.secret_name }));
      ({ isLoading, error } = gatewaySecrets);
      break;
    case 'gateway_endpoint':
      options = (gatewayEndpoints.data?.endpoints ?? []).map((e) => ({ id: e.endpoint_id, name: e.name }));
      ({ isLoading, error } = gatewayEndpoints);
      break;
    case 'gateway_model_definition':
      options = (gatewayModelDefinitions.data?.model_definitions ?? []).map((m) => ({
        id: m.model_definition_id,
        name: m.name,
      }));
      ({ isLoading, error } = gatewayModelDefinitions);
      break;
  }
  return { options, isLoading, error };
};
