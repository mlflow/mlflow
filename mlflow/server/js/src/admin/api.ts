import { matchPredefinedError, UnknownError } from '@databricks/web-shared/errors';
import { fetchEndpoint } from '../common/utils/FetchUtils';
import type {
  ListRolesResponse,
  RoleResponse,
  ListRolePermissionsResponse,
  RolePermissionResponse,
  AssignmentResponse,
  ListAssignmentsResponse,
  ListUserRolesResponse,
  ListUsersResponse,
  UserResponse,
  CreateRoleRequest,
  UpdateRoleRequest,
  AddPermissionRequest,
  UpdatePermissionRequest,
  CreateUserRequest,
  UpdatePasswordRequest,
  UpdateAdminRequest,
} from './types';

const defaultErrorHandler = async ({
  reject,
  response,
  err: originalError,
}: {
  reject: (cause: any) => void;
  // ``fetchEndpoint`` invokes the error handler without an HTTP response
  // when the request itself fails (network, CORS, abort), so ``response``
  // is genuinely optional. The implementation guards on ``!response``.
  response: Response | undefined;
  err: Error;
}) => {
  // ``response`` can be undefined when the request fails before getting a
  // response (network error, CORS failure, abort). Defer the predefined-error
  // match until we have one and fall back to the original error otherwise.
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

  listRoles: (workspace?: string) => {
    const params = new URLSearchParams();
    if (workspace) {
      params.append('workspace', workspace);
    }
    // Drop the trailing ``?`` when there are no params — the unconditional
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

  listPermissions: (roleId: number) => {
    const params = new URLSearchParams();
    params.append('role_id', roleId.toString());
    const relativeUrl = ['ajax-api/3.0/mlflow/roles/permissions/list', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ListRolePermissionsResponse>;
  },

  updatePermission: (request: UpdatePermissionRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/roles/permissions/update',
      method: 'PATCH',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<RolePermissionResponse>;
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

  listUserRoles: (username: string) => {
    const params = new URLSearchParams();
    params.append('username', username);
    const relativeUrl = ['ajax-api/3.0/mlflow/users/roles/list', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ListUserRolesResponse>;
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

  // Existing User APIs (v2)
  getCurrentUser: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/users/current',
      error: defaultErrorHandler,
    }) as Promise<UserResponse>;
  },

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
};
