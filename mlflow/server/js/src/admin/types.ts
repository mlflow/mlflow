export interface Role {
  id: number;
  name: string;
  workspace: string;
  description: string | null;
  permissions: RolePermission[];
}

export interface RolePermission {
  id: number;
  role_id: number;
  resource_type: string;
  resource_pattern: string;
  permission: string;
}

export interface UserRoleAssignment {
  id: number;
  user_id: number;
  role_id: number;
}

export interface User {
  id: number;
  username: string;
  is_admin: boolean;
}

// API response types
export interface ListRolesResponse {
  roles: Role[];
}

export interface RoleResponse {
  role: Role;
}

export interface ListRolePermissionsResponse {
  role_permissions: RolePermission[];
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

export interface ListUserRolesResponse {
  roles: Role[];
}

export interface ListUsersResponse {
  users: User[];
}

export interface UserResponse {
  user: User;
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

export interface UpdatePermissionRequest {
  role_permission_id: number;
  permission: string;
}

export interface CreateUserRequest {
  username: string;
  password: string;
}

export interface UpdatePasswordRequest {
  username: string;
  password: string;
  // Required when a user changes their own password; omitted when an admin
  // changes someone else's password. Backend enforces.
  current_password?: string;
}

export interface UpdateAdminRequest {
  username: string;
  is_admin: boolean;
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

/** A workspace-admin role has a (workspace, *, MANAGE) permission row. */
export const isWorkspaceAdminRole = (role: Role): boolean =>
  role.permissions?.some(
    (p) => p.resource_type === 'workspace' && p.resource_pattern === '*' && p.permission === 'MANAGE',
  ) ?? false;
