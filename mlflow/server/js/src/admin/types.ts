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
}

export interface ListUserRolesResponse {
  roles: Role[];
}

export interface UpdatePasswordRequest {
  username: string;
  password: string;
  // Required when a user changes their own password; omitted when an admin
  // changes someone else's password. Backend enforces.
  current_password?: string;
}

/** A workspace-admin role has a (workspace, *, MANAGE) permission row. */
export const isWorkspaceAdminRole = (role: Role): boolean =>
  role.permissions?.some(
    (p) => p.resource_type === 'workspace' && p.resource_pattern === '*' && p.permission === 'MANAGE',
  ) ?? false;
