export interface User {
  id: number;
  username: string;
  is_admin: boolean;
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
 * Direct (non-role-derived) per-resource grant. Shape matches
 * ``RolePermission`` (without role-table joins) so the frontend can
 * union both into one view. ``workspace`` is ``null`` when the
 * resource has been deleted or could not be resolved; the field is
 * always present on the wire (mirrors the backend's
 * ``_UserDirectPermission`` dataclass).
 */
export interface DirectPermission {
  resource_type: string;
  resource_pattern: string;
  permission: string;
  workspace: string | null;
}

export interface ListMyPermissionsResponse {
  permissions: DirectPermission[];
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
