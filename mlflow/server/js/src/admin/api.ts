import { matchPredefinedError, UnknownError } from '@databricks/web-shared/errors';
import { fetchEndpoint } from '../common/utils/FetchUtils';
import type {
  AddPermissionRequest,
  AssignmentResponse,
  CreateRoleRequest,
  CreateUserRequest,
  ListAssignmentsResponse,
  ListRolesResponse,
  ListUsersResponse,
  RolePermissionResponse,
  RoleResponse,
  UpdateAdminRequest,
  UpdateRoleRequest,
} from './types';
import type { ListMyPermissionsResponse, UpdatePasswordRequest, UserResponse } from '../account/types';

const defaultErrorHandler = async ({
  reject,
  response,
  err: originalError,
}: {
  reject: (cause: any) => void;
  // ``fetchEndpoint`` omits ``response`` on network/CORS/abort failures.
  response: Response | undefined;
  err: Error;
}) => {
  if (!response) {
    reject(originalError);
    return;
  }

  const predefinedError = matchPredefinedError(response);
  const error = predefinedError instanceof UnknownError ? originalError : predefinedError;
  try {
    const messageFromResponse = (await response.json())?.message;
    if (messageFromResponse) {
      error.message = messageFromResponse;
    }
  } catch {
    // Keep original error message if extraction fails
  }

  reject(error);
};

export const AdminApi = {
  // Role CRUD
  createRole: (request: CreateRoleRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/roles/create',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<RoleResponse>;
  },

  getRole: (roleId: number) => {
    const params = new URLSearchParams();
    params.append('role_id', roleId.toString());
    const relativeUrl = ['ajax-api/3.0/mlflow/roles/get', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<RoleResponse>;
  },

  listRoles: (workspaces?: string | readonly string[]) => {
    const params = new URLSearchParams();
    const list = workspaces === undefined ? [] : typeof workspaces === 'string' ? [workspaces] : workspaces;
    for (const w of list) {
      if (w) params.append('workspace', w);
    }
    // Drop the trailing ``?`` when there are no params - the unconditional
    // ``join('?')`` produced ``.../list?`` which clutters logs and breaks
    // some caching / signature checks.
    const query = params.toString();
    const relativeUrl = query ? `ajax-api/3.0/mlflow/roles/list?${query}` : 'ajax-api/3.0/mlflow/roles/list';
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ListRolesResponse>;
  },

  updateRole: (request: UpdateRoleRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/roles/update',
      method: 'PATCH',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<RoleResponse>;
  },

  deleteRole: (roleId: number) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/roles/delete',
      method: 'DELETE',
      body: JSON.stringify({ role_id: roleId }),
      error: defaultErrorHandler,
    });
  },

  // Role Permissions
  addPermission: (request: AddPermissionRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/roles/permissions/add',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<RolePermissionResponse>;
  },

  removePermission: (rolePermissionId: number) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/roles/permissions/remove',
      method: 'DELETE',
      body: JSON.stringify({ role_permission_id: rolePermissionId }),
      error: defaultErrorHandler,
    });
  },

  // User-Role Assignments
  assignRole: (username: string, roleId: number) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/roles/assign',
      method: 'POST',
      body: JSON.stringify({ username, role_id: roleId }),
      error: defaultErrorHandler,
    }) as Promise<AssignmentResponse>;
  },

  unassignRole: (username: string, roleId: number) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/roles/unassign',
      method: 'DELETE',
      body: JSON.stringify({ username, role_id: roleId }),
      error: defaultErrorHandler,
    });
  },

  listRoleUsers: (roleId: number) => {
    const params = new URLSearchParams();
    params.append('role_id', roleId.toString());
    const relativeUrl = ['ajax-api/3.0/mlflow/roles/users/list', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ListAssignmentsResponse>;
  },

  // Direct permissions for an arbitrary user (admin / self / WP-admin-of-target).
  // The response shape mirrors ``/users/current/permissions``.
  listUserPermissions: (username: string) => {
    const params = new URLSearchParams();
    params.append('username', username);
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/users/permissions/list?${params.toString()}`,
      error: defaultErrorHandler,
    }) as Promise<ListMyPermissionsResponse>;
  },

  // User CRUD (admin-only)
  listUsers: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/users/list',
      error: defaultErrorHandler,
    }) as Promise<ListUsersResponse>;
  },

  createUser: (request: CreateUserRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/users/create',
      method: 'POST',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<UserResponse>;
  },

  updatePassword: (request: UpdatePasswordRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/users/update-password',
      method: 'PATCH',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<void>;
  },

  updateAdmin: (request: UpdateAdminRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/users/update-admin',
      method: 'PATCH',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<void>;
  },

  deleteUser: (username: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/users/delete',
      method: 'DELETE',
      body: JSON.stringify({ username }),
      error: defaultErrorHandler,
    });
  },

  // Legacy per-resource permission CRUD endpoints. These are REST (not AJAX) and
  // continue to work pre/post Phase 2: post-migration the auth backend rewires
  // them to write `role_permissions` rows under a synthetic per-user role. Only
  // the create paths are exposed here - the user-permissions page is a write
  // surface today (no per-user listing API exists pre-Phase 2).
  grantUserPermission: (resourceType: string, resourceId: string, username: string, permission: string) => {
    const body = (extra: Record<string, string>) => JSON.stringify({ ...extra, username, permission });
    switch (resourceType) {
      case 'experiment':
        return fetchEndpoint({
          relativeUrl: 'api/2.0/mlflow/experiments/permissions/create',
          method: 'POST',
          body: body({ experiment_id: resourceId }),
          error: defaultErrorHandler,
        });
      case 'registered_model':
        return fetchEndpoint({
          relativeUrl: 'api/2.0/mlflow/registered-models/permissions/create',
          method: 'POST',
          body: body({ name: resourceId }),
          error: defaultErrorHandler,
        });
      case 'gateway_secret':
        return fetchEndpoint({
          relativeUrl: 'api/3.0/mlflow/gateway/secrets/permissions/create',
          method: 'POST',
          body: body({ secret_id: resourceId }),
          error: defaultErrorHandler,
        });
      case 'gateway_endpoint':
        return fetchEndpoint({
          relativeUrl: 'api/3.0/mlflow/gateway/endpoints/permissions/create',
          method: 'POST',
          body: body({ endpoint_id: resourceId }),
          error: defaultErrorHandler,
        });
      case 'gateway_model_definition':
        return fetchEndpoint({
          relativeUrl: 'api/3.0/mlflow/gateway/model-definitions/permissions/create',
          method: 'POST',
          body: body({ model_definition_id: resourceId }),
          error: defaultErrorHandler,
        });
      default:
        return Promise.reject(
          new Error(
            `Granting per-user permission for resource_type=${resourceType} is not supported. ` +
              'Use a role assignment instead.',
          ),
        );
    }
  },

  // Revoke a per-user direct permission. Mirror of ``grantUserPermission`` —
  // dispatches by ``resource_type`` to the matching ``DELETE_*_PERMISSION``
  // endpoint. The backend treats the (resource, user) pair as the primary
  // key, so no permission level is required to delete.
  revokeUserPermission: (resourceType: string, resourceId: string, username: string) => {
    const body = (extra: Record<string, string>) => JSON.stringify({ ...extra, username });
    switch (resourceType) {
      case 'experiment':
        return fetchEndpoint({
          relativeUrl: 'api/2.0/mlflow/experiments/permissions/delete',
          method: 'DELETE',
          body: body({ experiment_id: resourceId }),
          error: defaultErrorHandler,
        });
      case 'registered_model':
        return fetchEndpoint({
          relativeUrl: 'api/2.0/mlflow/registered-models/permissions/delete',
          method: 'DELETE',
          body: body({ name: resourceId }),
          error: defaultErrorHandler,
        });
      case 'gateway_secret':
        return fetchEndpoint({
          relativeUrl: 'api/3.0/mlflow/gateway/secrets/permissions/delete',
          method: 'DELETE',
          body: body({ secret_id: resourceId }),
          error: defaultErrorHandler,
        });
      case 'gateway_endpoint':
        return fetchEndpoint({
          relativeUrl: 'api/3.0/mlflow/gateway/endpoints/permissions/delete',
          method: 'DELETE',
          body: body({ endpoint_id: resourceId }),
          error: defaultErrorHandler,
        });
      case 'gateway_model_definition':
        return fetchEndpoint({
          relativeUrl: 'api/3.0/mlflow/gateway/model-definitions/permissions/delete',
          method: 'DELETE',
          body: body({ model_definition_id: resourceId }),
          error: defaultErrorHandler,
        });
      default:
        return Promise.reject(
          new Error(`Revoking per-user permission for resource_type=${resourceType} is not supported.`),
        );
    }
  },

  // Lightweight resource lookups for populating the per-user grant form.
  // ``max_results=1000`` is enough for typical workspaces; if any deployment
  // outgrows this we'd add pagination + server-side search. Each request
  // honours the active workspace (via the ``X-MLFLOW-WORKSPACE`` header
  // ``fetchEndpoint`` already attaches), so admins see resources in the
  // workspace they're currently scoped to.
  listExperimentsLite: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/experiments/search?max_results=1000',
      error: defaultErrorHandler,
    }) as Promise<{ experiments?: { experiment_id: string; name: string }[] }>;
  },

  listRegisteredModelsLite: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/registered-models/search?max_results=1000',
      error: defaultErrorHandler,
    }) as Promise<{ registered_models?: { name: string }[] }>;
  },

  listGatewaySecretsLite: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/gateway/secrets/list',
      error: defaultErrorHandler,
    }) as Promise<{ secrets?: { secret_id: string; secret_name: string }[] }>;
  },

  listGatewayEndpointsLite: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/gateway/endpoints/list',
      error: defaultErrorHandler,
    }) as Promise<{ endpoints?: { endpoint_id: string; name: string }[] }>;
  },

  listGatewayModelDefinitionsLite: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/gateway/model-definitions/list',
      error: defaultErrorHandler,
    }) as Promise<{ model_definitions?: { model_definition_id: string; name: string }[] }>;
  },
};
