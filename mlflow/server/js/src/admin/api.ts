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
  // The response shape mirrors ``/users/current/permissions``. ``workspace``
  // (when set) overrides the ``X-MLFLOW-WORKSPACE`` header so the modal can
  // read grants from a workspace other than the session-active one — paired
  // with the grant/revoke workspace override so pre-filled rows and the
  // submit-time mutations target the same workspace.
  listUserPermissions: (username: string, workspace?: string) => {
    const params = new URLSearchParams();
    params.append('username', username);
    return fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/users/permissions/list?${params.toString()}`,
      error: defaultErrorHandler,
      ...workspaceHeader(workspace),
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
  grantUserPermission: (
    resourceType: string,
    resourceId: string,
    username: string,
    permission: string,
    workspace?: string,
  ) => {
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
      ...workspaceHeader(workspace),
    });
  },

  revokeUserPermission: (resourceType: string, resourceId: string, username: string, workspace?: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/users/permissions/revoke',
      method: 'POST',
      body: JSON.stringify({
        username,
        resource_type: resourceType,
        resource_id: resourceId,
      }),
      error: defaultErrorHandler,
      ...workspaceHeader(workspace),
    });
  },

  // Lightweight resource lookups for populating the per-user grant form.
  // ``max_results=1000`` is enough for typical workspaces; if any deployment
  // outgrows this we'd add pagination + server-side search. ``workspace``
  // (when set) overrides the ``X-MLFLOW-WORKSPACE`` header for the request
  // so the picker can fetch resources from a workspace other than the
  // user's session-active one (used by the role / direct-grant flows when
  // the admin is granting *into* a different workspace).
  listExperimentsLite: (workspace?: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/experiments/search?max_results=1000',
      error: defaultErrorHandler,
      ...workspaceHeader(workspace),
    }) as Promise<{ experiments?: { experiment_id: string; name: string }[] }>;
  },

  listRegisteredModelsLite: (workspace?: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/registered-models/search?max_results=1000',
      error: defaultErrorHandler,
      ...workspaceHeader(workspace),
    }) as Promise<{ registered_models?: { name: string }[] }>;
  },

  listGatewaySecretsLite: (workspace?: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/gateway/secrets/list',
      error: defaultErrorHandler,
      ...workspaceHeader(workspace),
    }) as Promise<{ secrets?: { secret_id: string; secret_name: string }[] }>;
  },

  listGatewayEndpointsLite: (workspace?: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/gateway/endpoints/list',
      error: defaultErrorHandler,
      ...workspaceHeader(workspace),
    }) as Promise<{ endpoints?: { endpoint_id: string; name: string }[] }>;
  },

  // Cross-experiment ``ListScorers``: omitting ``experiment_id`` returns every
  // scorer in the active workspace. The auth-side ``filter_list_scorers``
  // ``AFTER_REQUEST_PATH_HANDLERS`` entry filters the response by the caller's
  // experiment + scorer read predicates. ``resource_pattern`` is computed
  // client-side via ``scorerResourcePattern`` to match
  // ``SqlAlchemyStore._scorer_pattern``.
  listScorersLite: (workspace?: string) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/scorers/list',
      error: defaultErrorHandler,
      ...workspaceHeader(workspace),
    }) as Promise<{ scorers?: Scorer[] }>;
  },

  // Prompts share the registered-models table (tagged with
  // ``mlflow.prompt.is_prompt = 'true'``). The existing search endpoint with
  // a tag filter is enough — ``filter_search_registered_models`` classifies
  // each row by the same tag and applies ``_role_based_read_predicate('prompt')``,
  // so a dedicated prompt list-lite handler isn't needed.
  listPromptsLite: (workspace?: string) => {
    const filter = encodeURIComponent("tag.mlflow.prompt.is_prompt = 'true'");
    return fetchEndpoint({
      relativeUrl: `ajax-api/2.0/mlflow/registered-models/search?max_results=1000&filter=${filter}`,
      error: defaultErrorHandler,
      ...workspaceHeader(workspace),
    }) as Promise<{ registered_models?: { name: string }[] }>;
  },
};

// Override ``X-MLFLOW-WORKSPACE`` per request; ``fetchEndpoint`` merges
// ``headerOptions`` after the defaults. ``!== undefined`` (not truthy) so a
// bogus ``''`` surfaces as a 4xx instead of inheriting the session workspace.
const workspaceHeader = (workspace?: string): { headerOptions?: Record<string, string> } =>
  workspace !== undefined ? { headerOptions: { 'X-MLFLOW-WORKSPACE': workspace } } : {};

/** Shape of one scorer row from the generic ``ListScorers`` response. */
export interface Scorer {
  experiment_id: number;
  scorer_name: string;
  scorer_version?: number;
  scorer_id?: string;
}

/**
 * Composite RBAC resource pattern for a scorer. Mirrors the server's
 * ``SqlAlchemyStore._scorer_pattern`` exactly so the picker's submitted id
 * lines up byte-for-byte with persisted grants.
 *
 * Python's ``urllib.parse.quote(name, safe='')`` percent-encodes more
 * characters than JS's ``encodeURIComponent`` — notably ``*'!()``, which JS
 * preserves but Python escapes. Patch over the gap so a scorer name with any
 * of those characters still resolves the grant on the server side.
 */
export const scorerResourcePattern = (experimentId: number | string, scorerName: string): string => {
  const encoded = encodeURIComponent(scorerName).replace(
    /[*'!()]/g,
    (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`,
  );
  return `${experimentId}/${encoded}`;
};
