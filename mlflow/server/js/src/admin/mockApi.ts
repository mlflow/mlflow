/**
 * In-memory mock API for the RBAC admin UI.
 * Toggle on/off via USE_MOCK_API in hooks.ts.
 * Supports full CRUD so you can exercise the complete UI flow.
 *
 * Permission enforcement:
 *   - Admin-only endpoints (user CRUD, role CRUD, permission mgmt) throw 403
 *     when the current cookie user is not is_admin.
 *   - listUserRoles and updatePassword are always allowed (self-service).
 *
 * Dev/test API:
 *   window.__mockAdmin.switchUser('alice')   — sets cookie + returns new user info
 *   window.__mockAdmin.getUsers()            — returns all mock users
 *   window.__mockAdmin.getCurrentUser()      — returns current user from cookie
 */
import type {
  ListRolesResponse,
  RoleResponse,
  ListUsersResponse,
  UserResponse,
  ListAssignmentsResponse,
  AssignmentResponse,
  ListUserRolesResponse,
  RolePermissionResponse,
  CreateRoleRequest,
  UpdateRoleRequest,
  AddPermissionRequest,
  UpdatePermissionRequest,
  CreateUserRequest,
  UpdatePasswordRequest,
  UpdateAdminRequest,
  Role,
  User,
  RolePermission,
  UserRoleAssignment,
} from './types';

let nextId = 100;
const id = () => nextId++;

// ── Seed data ──────────────────────────────────────────────────────────────────

const users: User[] = [
  { id: 1, username: 'admin', is_admin: true },
  { id: 2, username: 'alice', is_admin: false },
  { id: 3, username: 'bob', is_admin: false },
  { id: 4, username: 'charlie', is_admin: false },
];

const roles: Role[] = [
  {
    id: 10,
    name: 'Data Scientists',
    workspace: 'default',
    description: 'Read/write access to experiments and models',
    permissions: [
      { id: 50, role_id: 10, resource_type: 'experiment', resource_pattern: '*', permission: 'EDIT' },
      { id: 51, role_id: 10, resource_type: 'registered_model', resource_pattern: '*', permission: 'EDIT' },
    ],
  },
  {
    id: 11,
    name: 'ML Engineers',
    workspace: 'default',
    description: 'Full access to models and gateway',
    permissions: [
      { id: 52, role_id: 11, resource_type: 'registered_model', resource_pattern: '*', permission: 'MANAGE' },
      { id: 53, role_id: 11, resource_type: 'gateway_endpoint', resource_pattern: '*', permission: 'MANAGE' },
      { id: 54, role_id: 11, resource_type: 'experiment', resource_pattern: '*', permission: 'EDIT' },
    ],
  },
  {
    id: 12,
    name: 'Workspace Admins',
    workspace: 'default',
    description: 'Full workspace administration',
    permissions: [
      { id: 55, role_id: 12, resource_type: 'workspace', resource_pattern: '*', permission: 'MANAGE' },
    ],
  },
  {
    id: 13,
    name: 'Viewers',
    workspace: 'default',
    description: 'Read-only access to experiments',
    permissions: [
      { id: 56, role_id: 13, resource_type: 'experiment', resource_pattern: '*', permission: 'READ' },
    ],
  },
];

const assignments: UserRoleAssignment[] = [
  { id: 60, user_id: 1, role_id: 12 },
  { id: 61, user_id: 2, role_id: 10 },
  { id: 62, user_id: 3, role_id: 11 },
  { id: 63, user_id: 3, role_id: 10 },
  { id: 64, user_id: 4, role_id: 13 },
];

// ── Helpers ────────────────────────────────────────────────────────────────────

const delay = (ms = 200) => new Promise((r) => setTimeout(r, ms));

const findRole = (roleId: number) => {
  const role = roles.find((r) => r.id === roleId);
  if (!role) throw new Error(`Role ${roleId} not found`);
  return role;
};

const getCurrentUsername = (): string =>
  document.cookie
    .split('; ')
    .find((row) => row.startsWith('mlflow_user='))
    ?.substring('mlflow_user='.length) ?? '';

const getCurrentUser = (): User | undefined => {
  const username = getCurrentUsername();
  return users.find((u) => u.username === username);
};

const requireAdmin = () => {
  const user = getCurrentUser();
  if (!user || !user.is_admin) {
    const err = new Error('Permission denied: admin access required');
    (err as any).status = 403;
    throw err;
  }
};

// ── Public state accessors (for dev toolbar & tests) ───────────────────────────

export const mockState = {
  getUsers: () => [...users],
  getRoles: () => [...roles],
  getAssignments: () => [...assignments],
  getCurrentUser,
  switchUser: (username: string) => {
    const user = users.find((u) => u.username === username);
    if (!user) throw new Error(`Mock user '${username}' not found`);
    document.cookie = `mlflow_user=${username}; path=/`;
    return user;
  },
};

// Expose on window for integration tests and browser console
declare global {
  interface Window {
    __mockAdmin: typeof mockState;
  }
}
window.__mockAdmin = mockState;

// ── Mock API ───────────────────────────────────────────────────────────────────

export const MockAdminApi = {
  // Roles — admin-only for mutations, readable by all
  listRoles: async (_workspace?: string): Promise<ListRolesResponse> => {
    await delay();
    requireAdmin();
    return { roles: [...roles] };
  },

  getRole: async (roleId: number): Promise<RoleResponse> => {
    await delay();
    requireAdmin();
    return { role: findRole(roleId) };
  },

  createRole: async (request: CreateRoleRequest): Promise<RoleResponse> => {
    await delay();
    requireAdmin();
    const role: Role = {
      id: id(),
      name: request.name,
      workspace: request.workspace,
      description: request.description ?? null,
      permissions: [],
    };
    roles.push(role);
    return { role };
  },

  updateRole: async (request: UpdateRoleRequest): Promise<RoleResponse> => {
    await delay();
    requireAdmin();
    const role = findRole(request.role_id);
    if (request.name) role.name = request.name;
    if (request.description !== undefined) role.description = request.description || null;
    return { role };
  },

  deleteRole: async (roleId: number): Promise<void> => {
    await delay();
    requireAdmin();
    const idx = roles.findIndex((r) => r.id === roleId);
    if (idx === -1) throw new Error('Role not found');
    roles.splice(idx, 1);
    for (let i = assignments.length - 1; i >= 0; i--) {
      if (assignments[i].role_id === roleId) assignments.splice(i, 1);
    }
  },

  // Permissions — admin-only
  addPermission: async (request: AddPermissionRequest): Promise<RolePermissionResponse> => {
    await delay();
    requireAdmin();
    const role = findRole(request.role_id);
    const perm: RolePermission = {
      id: id(),
      role_id: request.role_id,
      resource_type: request.resource_type,
      resource_pattern: request.resource_pattern,
      permission: request.permission,
    };
    role.permissions.push(perm);
    return { role_permission: perm };
  },

  removePermission: async (rolePermissionId: number): Promise<void> => {
    await delay();
    requireAdmin();
    for (const role of roles) {
      const idx = role.permissions.findIndex((p) => p.id === rolePermissionId);
      if (idx !== -1) {
        role.permissions.splice(idx, 1);
        return;
      }
    }
    throw new Error('Permission not found');
  },

  updatePermission: async (request: UpdatePermissionRequest): Promise<RolePermissionResponse> => {
    await delay();
    requireAdmin();
    for (const role of roles) {
      const perm = role.permissions.find((p) => p.id === request.role_permission_id);
      if (perm) {
        perm.permission = request.permission;
        return { role_permission: perm };
      }
    }
    throw new Error('Permission not found');
  },

  listPermissions: async (roleId: number) => {
    await delay();
    requireAdmin();
    const role = findRole(roleId);
    return { role_permissions: role.permissions };
  },

  // User-Role Assignments — admin-only
  assignRole: async (username: string, roleId: number): Promise<AssignmentResponse> => {
    await delay();
    requireAdmin();
    const user = users.find((u) => u.username === username);
    if (!user) throw new Error(`User ${username} not found`);
    findRole(roleId);
    const assignment: UserRoleAssignment = { id: id(), user_id: user.id, role_id: roleId };
    assignments.push(assignment);
    return { assignment };
  },

  unassignRole: async (username: string, roleId: number): Promise<void> => {
    await delay();
    requireAdmin();
    const user = users.find((u) => u.username === username);
    if (!user) throw new Error(`User ${username} not found`);
    const idx = assignments.findIndex((a) => a.user_id === user.id && a.role_id === roleId);
    if (idx === -1) throw new Error('Assignment not found');
    assignments.splice(idx, 1);
  },

  listRoleUsers: async (roleId: number): Promise<ListAssignmentsResponse> => {
    await delay();
    requireAdmin();
    return { assignments: assignments.filter((a) => a.role_id === roleId) };
  },

  // Self-service — any authenticated user
  listUserRoles: async (username: string): Promise<ListUserRolesResponse> => {
    await delay();
    const user = users.find((u) => u.username === username);
    if (!user) return { roles: [] };
    const roleIds = assignments.filter((a) => a.user_id === user.id).map((a) => a.role_id);
    return { roles: roles.filter((r) => roleIds.includes(r.id)) };
  },

  listAllRoles: async (): Promise<ListRolesResponse> => {
    await delay();
    requireAdmin();
    return { roles: [...roles] };
  },

  // Users — admin-only for list/create/delete/updateAdmin
  listUsers: async (): Promise<ListUsersResponse> => {
    await delay();
    requireAdmin();
    return { users: [...users] };
  },

  createUser: async (request: CreateUserRequest): Promise<UserResponse> => {
    await delay();
    requireAdmin();
    if (users.some((u) => u.username === request.username)) {
      throw new Error(`User '${request.username}' already exists`);
    }
    const user: User = { id: id(), username: request.username, is_admin: false };
    users.push(user);
    return { user };
  },

  // Self-service
  updatePassword: async (_request: UpdatePasswordRequest): Promise<void> => {
    await delay();
  },

  updateAdmin: async (request: UpdateAdminRequest): Promise<void> => {
    await delay();
    requireAdmin();
    const user = users.find((u) => u.username === request.username);
    if (!user) throw new Error('User not found');
    user.is_admin = request.is_admin;
  },

  deleteUser: async (username: string): Promise<void> => {
    await delay();
    requireAdmin();
    const idx = users.findIndex((u) => u.username === username);
    if (idx === -1) throw new Error('User not found');
    const userId = users[idx].id;
    users.splice(idx, 1);
    for (let i = assignments.length - 1; i >= 0; i--) {
      if (assignments[i].user_id === userId) assignments.splice(i, 1);
    }
  },
};
