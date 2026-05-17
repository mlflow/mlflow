import type { DirectPermission, Role, RolePermission, User } from '../account/types';

// Re-export identity primitives + helpers from account/ so admin pages can
// keep importing them via ``./types``. The canonical home is account/.
export type { DirectPermission, Role, RolePermission, User };
export {
  ALL_RESOURCE_PATTERN,
  ALL_RESOURCE_PATTERN_LABEL,
  DEFAULT_WORKSPACE_NAME,
  formatResourcePattern,
  isWorkspaceAdminRole,
  parseResourcePattern,
} from '../account/types';

export interface UserRoleAssignment {
  id: number;
  user_id: number;
  role_id: number;
}

// API response types
export interface ListRolesResponse {
  roles: Role[];
}

export interface RoleResponse {
  role: Role;
}

export interface RolePermissionResponse {
  role_permission: RolePermission;
}

export interface AssignmentResponse {
  assignment: UserRoleAssignment;
}

export interface ListAssignmentsResponse {
  assignments: UserRoleAssignment[];
}

export interface ListUsersResponse {
  users: User[];
}

// Request types
export interface CreateRoleRequest {
  name: string;
  workspace: string;
  description?: string;
}

export interface UpdateRoleRequest {
  role_id: number;
  name?: string;
  description?: string;
}

export interface AddPermissionRequest {
  role_id: number;
  resource_type: string;
  resource_pattern: string;
  permission: string;
}

export interface CreateUserRequest {
  username: string;
  password: string;
}

export interface UpdateAdminRequest {
  username: string;
  is_admin: boolean;
}

/**
 * Options for the resource-id selector in the per-user permission grant
 * form. ``id`` is the canonical identifier sent to the backend (e.g.
 * ``experiment_id`` or ``registered_model.name``); ``name`` is the
 * human-readable label shown in the dropdown.
 */
export interface ResourceOption {
  id: string;
  name: string;
}

export const RESOURCE_TYPES = [
  'experiment',
  'registered_model',
  'scorer',
  'gateway_secret',
  'gateway_endpoint',
  'gateway_model_definition',
  'workspace',
] as const;

export const PERMISSIONS = ['READ', 'USE', 'EDIT', 'MANAGE', 'NO_PERMISSIONS'] as const;
