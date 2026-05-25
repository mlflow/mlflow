import { useCallback, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useSearchParams } from '../common/utils/RoutingUtils';
import { SETTINGS_RETURN_TO_PARAM } from '../settings/settingsSectionConstants';
import { AccountQueryKeys } from '../account/hooks';
import { AdminApi, scorerResourcePattern } from './api';
import { isSyntheticUserRole } from '../account/types';
import type {
  AddPermissionRequest,
  CreateRoleRequest,
  CreateUserRequest,
  ResourceOption,
  UpdateAdminRequest,
  UpdateRoleRequest,
} from './types';
import { ALL_RESOURCE_PATTERN } from './types';

// Re-export account-side hooks so admin pages can keep importing them via
// ``./hooks``. The canonical home is account/hooks.
export {
  useCurrentUserAdminWorkspaces,
  useCurrentUserIsAdmin,
  useCurrentUserIsWorkspaceAdmin,
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
};

// User queries and mutations
export const useUsersQuery = () => {
  return useQuery({
    queryKey: AdminQueryKeys.users,
    queryFn: async () => {
      const data = await AdminApi.listUsers();
      // Drop synthetic ``__user_<id>__`` roles from each user's roles
      // list. They're a backend implementation detail for direct grants
      // and shouldn't appear in human-facing user/role tables.
      return {
        ...data,
        users: data.users.map((u) => ({
          ...u,
          roles: u.roles?.filter((r) => !isSyntheticUserRole(r.name)),
        })),
      };
    },
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
//
// ``workspaces`` accepts a single workspace, a list of workspaces, or
// ``undefined`` (admin-only unscoped listing). The cache key normalizes a list
// to a sorted form so callers passing the same set in different order share
// the same entry.
export const useRolesQuery = (workspaces?: string | readonly string[], options: { enabled?: boolean } = {}) => {
  const normalized = useMemo<string | readonly string[] | undefined>(() => {
    if (workspaces === undefined) return undefined;
    if (typeof workspaces === 'string') return workspaces;
    return [...workspaces].sort();
  }, [workspaces]);
  return useQuery({
    queryKey: [...AdminQueryKeys.roles, normalized],
    queryFn: async () => {
      const data = await AdminApi.listRoles(normalized);
      // Synthetic ``__user_<id>__`` roles back direct grants and are not
      // human-facing. Drop them at the source so every consumer
      // (tables, dropdowns, name lookups) stays in sync.
      return { ...data, roles: data.roles.filter((r) => !isSyntheticUserRole(r.name)) };
    },
    retry: false,
    refetchOnWindowFocus: false,
    // Callers pass ``enabled: false`` to skip a fetch they know will 403.
    enabled: options.enabled !== false,
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
//
// Invalidate both the per-user ``userRoles`` query (UserDetailPage, Account
// page) and the bulk ``users`` query (Admin Users tab eager-loads roles per
// user) so the new assignment is reflected everywhere on the next render.
export const useAssignRole = (roleId: number) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (username: string) => AdminApi.assignRole(username, roleId),
    onSuccess: (_data, username) => {
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.roleUsers(roleId) });
      queryClient.invalidateQueries({ queryKey: AccountQueryKeys.userRoles(username) });
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
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
      queryClient.invalidateQueries({ queryKey: AdminQueryKeys.users });
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
      // Direct grants flow through the synthetic ``__user_<id>__`` role
      // surfaced by ``listUserRoles``, so a single ``userRoles`` invalidation
      // refreshes both the Roles tab and the Direct Permissions view.
      queryClient.invalidateQueries({ queryKey: AccountQueryKeys.userRoles(variables.username) });
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
  // Also fired for ``scorer`` so the picker can join scorers to their
  // experiment names client-side — keeps ``experiment_name`` off the
  // ``Scorer`` proto (it's a UI concern, per Tome's review).
  const experiments = useQuery({
    queryKey: AdminQueryKeys.resourceOptions('experiment'),
    queryFn: AdminApi.listExperimentsLite,
    enabled: resourceType === 'experiment' || resourceType === 'scorer',
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
  const scorers = useQuery({
    queryKey: AdminQueryKeys.resourceOptions('scorer'),
    queryFn: AdminApi.listScorersLite,
    enabled: resourceType === 'scorer',
    retry: false,
    refetchOnWindowFocus: false,
  });
  const prompts = useQuery({
    queryKey: AdminQueryKeys.resourceOptions('prompt'),
    queryFn: AdminApi.listPromptsLite,
    enabled: resourceType === 'prompt',
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
    case 'scorer': {
      // ``id`` is the composite ``<experiment_id>/<urlencoded scorer_name>``
      // computed client-side to match the backend's ``_scorer_pattern`` — the
      // picker sets ``draft.resourceId = id``, which is what the staged grant
      // submits as its ``resource_pattern``. Display label disambiguates two
      // same-named scorers in different experiments by looking up the name
      // in the parallel-fetched experiments list (the ``Scorer`` proto itself
      // doesn't carry it — UI concern, kept off the wire); falls back to the
      // raw experiment id when the lookup misses.
      const experimentNameById = new Map(
        (experiments.data?.experiments ?? []).map((e) => [e.experiment_id, e.name] as const),
      );
      options = (scorers.data?.scorers ?? []).map((s) => ({
        id: scorerResourcePattern(s.experiment_id, s.scorer_name),
        name: `${s.scorer_name} (in ${experimentNameById.get(String(s.experiment_id)) ?? s.experiment_id})`,
      }));
      isLoading = scorers.isLoading || experiments.isLoading;
      error = scorers.error ?? experiments.error;
      break;
    }
    case 'prompt':
      // Prompts are identified by name (same as registered_models); the lite
      // call hits ``registered-models/search`` with a tag filter, so the
      // response shape is ``{registered_models: [{name}]}``.
      options = (prompts.data?.registered_models ?? []).map((p) => ({ id: p.name, name: p.name }));
      ({ isLoading, error } = prompts);
      break;
  }
  // A resource literally named ``*`` would collide with the wildcard scope
  // (backend stores both as ``resource_pattern = '*'``), so drop it from the
  // picker — selecting it would silently grant access to ALL resources of
  // the type. Users that genuinely need an all-resources grant should use
  // the wildcard radio.
  options = options.filter((o) => o.id !== ALL_RESOURCE_PATTERN);
  return { options, isLoading, error };
};
