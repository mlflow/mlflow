import type { Role, RolePermission, User, UserRolePermissionRow } from '../account/types';

// Re-export identity primitives + helpers from account/ so admin pages can
// keep importing them via ``./types``. The canonical home is account/.
export type { Role, RolePermission, User, UserRolePermissionRow };
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

// ``gateway_model_definition`` is a valid backend resource type but isn't
// surfaced anywhere in the admin UI — left out of every frontend enum,
// label map, and picker query. Re-add when it becomes user-facing.
export const RESOURCE_TYPES = [
  'experiment',
  'registered_model',
  'prompt',
  'scorer',
  'gateway_secret',
  'gateway_endpoint',
  'workspace',
] as const;

/**
 * User-facing labels for resource types. Used by the role and direct
 * permission pickers so admins see "Registered model" / "LLM connection"
 * instead of raw discriminators like ``registered_model`` /
 * ``gateway_secret``. ``gateway_secret`` is surfaced as "LLM connection"
 * to match the product page that exposes those secrets.
 */
export const RESOURCE_TYPE_LABELS = {
  experiment: 'Experiment',
  registered_model: 'Registered model',
  prompt: 'Prompt',
  scorer: 'Scorer',
  gateway_secret: 'LLM connection',
  gateway_endpoint: 'LLM endpoint',
  workspace: 'Workspace',
} satisfies Record<(typeof RESOURCE_TYPES)[number], string>;

export const getResourceTypeLabel = (resourceType: string): string =>
  RESOURCE_TYPE_LABELS[resourceType as keyof typeof RESOURCE_TYPE_LABELS] ?? resourceType;

export const PERMISSIONS = ['READ', 'USE', 'EDIT', 'MANAGE', 'NO_PERMISSIONS'] as const;

/**
 * Permission levels the picker should expose per resource type, mirroring
 * the backend's ``_validate_permission_for_resource_type``:
 * ``workspace`` is ``USE`` / ``MANAGE`` only; gateway types expose ``USE``;
 * other types hide ``USE`` (no-op over ``READ``); ``NO_PERMISSIONS`` is
 * hidden everywhere.
 */
export const PERMISSIONS_FOR_RESOURCE_TYPE = {
  experiment: ['READ', 'EDIT', 'MANAGE'],
  registered_model: ['READ', 'EDIT', 'MANAGE'],
  prompt: ['READ', 'EDIT', 'MANAGE'],
  scorer: ['READ', 'EDIT', 'MANAGE'],
  gateway_secret: ['READ', 'USE', 'EDIT', 'MANAGE'],
  gateway_endpoint: ['READ', 'USE', 'EDIT', 'MANAGE'],
  workspace: ['USE', 'MANAGE'],
} satisfies Record<string, readonly string[]>;

export const getGrantablePermissions = (resourceType: string): readonly string[] =>
  PERMISSIONS_FOR_RESOURCE_TYPE[resourceType as keyof typeof PERMISSIONS_FOR_RESOURCE_TYPE] ?? [
    'READ',
    'EDIT',
    'MANAGE',
  ];
