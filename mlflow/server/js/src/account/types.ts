export interface User {
  id: number;
  username: string;
  is_admin: boolean;
  /**
   * Roles the requester is authorized to see for this user. The backend
   * scopes the list per-requester (admin sees every role; workspace
   * managers see only roles in workspaces they administer). Optional so
   * older callers / fixtures that don't populate it stay valid.
   */
  roles?: Role[];
}

export interface RolePermission {
  id: number;
  role_id: number;
  resource_type: string;
  resource_pattern: string;
  permission: string;
}

export interface Role {
  id: number;
  name: string;
  workspace: string;
  description: string | null;
  permissions: RolePermission[];
}

export interface UserResponse {
  user: User;
  /**
   * True when the server's ``authorization_function`` is the built-in
   * Basic Auth handler. Custom auth plugins set this to false so the
   * frontend can hide Basic-Auth-only affordances (logout XHR trick,
   * change-password modal).
   */
  is_basic_auth?: boolean;
}

export interface ListUserRolesResponse {
  roles: Role[];
}

/**
 * One row of ``GET /users/permissions/list``: a single permission grant on
 * one of the user's roles, carrying enough role identity that the frontend
 * can split the "Direct permissions" view (rows whose ``role_name`` matches
 * the synthetic ``__user_<id>__`` pattern) from the "Role permissions"
 * view. ``workspace`` is the role's workspace.
 */
export interface UserRolePermissionRow {
  role_id: number;
  role_name: string;
  workspace: string;
  resource_type: string;
  resource_pattern: string;
  permission: string;
}

export interface ListMyPermissionsResponse {
  is_admin: boolean;
  permissions: UserRolePermissionRow[];
}

export interface UpdatePasswordRequest {
  username: string;
  password: string;
  // Required for self-service; omitted when an admin updates another user.
  current_password?: string;
}

/** A workspace-admin role has a (workspace, *, MANAGE) permission row. */
export const isWorkspaceAdminRole = (role: Role): boolean =>
  role.permissions?.some(
    (p) => p.resource_type === 'workspace' && p.resource_pattern === '*' && p.permission === 'MANAGE',
  ) ?? false;

/**
 * Mirrors ``mlflow.utils.workspace_utils.DEFAULT_WORKSPACE_NAME``. Used
 * to filter out stale non-default rows in single-tenant mode, where the
 * Workspace column is hidden.
 */
export const DEFAULT_WORKSPACE_NAME = 'default';

/** Backend wildcard ``*`` is surfaced to users as "all". */
export const ALL_RESOURCE_PATTERN = '*';
export const ALL_RESOURCE_PATTERN_LABEL = 'all';

export const formatResourcePattern = (pattern: string): string =>
  pattern === ALL_RESOURCE_PATTERN ? ALL_RESOURCE_PATTERN_LABEL : pattern;

/** Inverse of ``formatResourcePattern`` - translates user input back to the canonical wildcard. */
export const parseResourcePattern = (input: string): string => {
  const trimmed = input.trim();
  return trimmed.toLowerCase() === ALL_RESOURCE_PATTERN_LABEL ? ALL_RESOURCE_PATTERN : trimmed;
};
