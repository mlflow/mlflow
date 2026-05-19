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

  // Unified per-user permission convenience APIs (replace the 10 legacy
  // per-resource ``create`` / ``delete`` endpoints removed in this PR).
  // The server resolves ``(resource_type, resource_id)`` against the user's
  // synthetic ``__user_<id>__`` role; resource-type validation lives on the
  // store layer.
  grantUserPermission: (resourceType: string, resourceId: string, username: string, permission: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/users/permissions/grant',
      method: 'POST',
      body: JSON.stringify({
        username,
        resource_type: resourceType,
        resource_id: resourceId,
        permission,
      }),
      error: defaultErrorHandler,
    });
  },

  revokeUserPermission: (resourceType: string, resourceId: string, username: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/users/permissions/revoke',
      method: 'POST',
      body: JSON.stringify({
        username,
        resource_type: resourceType,
        resource_id: resourceId,
      }),
      error: defaultErrorHandler,
    });
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
