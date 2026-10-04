import type { Role, RolePermission, User, UserRolePermissionRow } from '../account/types';

// Re-export identity primitives + helpers from account/ so admin pages can
// keep importing them via ``./types``. The canonical home is account/.
export type { Role, RolePermission, User, UserRolePermissionRow };
export {
  ALL_RESOURCE_PATTERN,
  ALL_RESOURCE_PATTERN_LABEL,
  DEFAULT_WORKSPACE_NAME,
  formatResourcePattern,
  isSyntheticUserRole,
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
//
// Ordered parent-then-sub-resource so the dropdown reads as a hierarchy:
// an experiment's runs/traces/assessments follow it, each registry entity
// is followed by its versions.
export const RESOURCE_TYPES = [
  'experiment',
  'run',
  'trace',
  'assessment',
  'logged_model',
  'review_queue',
  'registered_model',
  'registered_model_version',
  'prompt',
  'prompt_version',
  'scorer',
  'scorer_version',
  'gateway_secret',
  'gateway_endpoint',
  'mcp_server',
  'mcp_server_version',
  'workspace',
] as const;

/**
 * Resource types whose grants are only ever wildcard (``*``) — they mirror
 * the backend's ``TYPE`` grain map, where these carry ``PatternKind.WILDCARD``
 * alone while the rest also accept ``PatternKind.ID``. A sub-resource tier is
 * granted across a whole workspace rather than per row, so the pickers must not
 * offer a specific-resource scope for one.
 */
export const WILDCARD_ONLY_RESOURCE_TYPES = [
  'run',
  'trace',
  'assessment',
  'logged_model',
  'review_queue',
  'registered_model_version',
  'prompt_version',
  'scorer_version',
  'mcp_server_version',
  'workspace',
] as const satisfies readonly (typeof RESOURCE_TYPES)[number][];

export const isWildcardOnlyResourceType = (resourceType: string): boolean =>
  (WILDCARD_ONLY_RESOURCE_TYPES as readonly string[]).includes(resourceType);

/**
 * User-facing labels for resource types. Used by the role and direct
 * permission pickers so admins see "Registered model" / "LLM connection"
 * instead of raw discriminators like ``registered_model`` /
 * ``gateway_secret``. ``gateway_secret`` is surfaced as "LLM connection"
 * to match the product page that exposes those secrets.
 */
export const RESOURCE_TYPE_LABELS = {
  experiment: 'Experiment',
  run: 'Run',
  trace: 'Trace',
  assessment: 'Assessment',
  logged_model: 'Logged model',
  review_queue: 'Review queue',
  registered_model: 'Registered model',
  registered_model_version: 'Model version',
  prompt: 'Prompt',
  prompt_version: 'Prompt version',
  scorer: 'Scorer',
  scorer_version: 'Scorer version',
  gateway_secret: 'LLM connection',
  gateway_endpoint: 'LLM endpoint',
  mcp_server: 'MCP server',
  mcp_server_version: 'MCP server version',
  workspace: 'Workspace',
} satisfies Record<(typeof RESOURCE_TYPES)[number], string>;

export const getResourceTypeLabel = (resourceType: string): string =>
  RESOURCE_TYPE_LABELS[resourceType as keyof typeof RESOURCE_TYPE_LABELS] ?? resourceType;

// ``NO_PERMISSIONS`` is deliberately absent: the backend's
// ``_validate_permission_for_resource_type`` rejects it for every resource type,
// because an absent grant combined with ``default_permission`` already says
// "no access". ``DENY`` is the grantable veto instead.
export const PERMISSIONS = ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY'] as const;

/**
 * Permission levels the picker should expose per resource type, mirroring
 * the backend's ``_validate_permission_for_resource_type``:
 * ``workspace`` is ``USE`` / ``MANAGE`` only; gateway types expose ``USE``;
 * other types hide ``USE`` (no-op over ``READ``).
 *
 * ``DENY`` comes last on every non-workspace type. It is not a weaker level than
 * the ones above it — it is a veto that beats any positive grant on the same key
 * and is exempt from the ``default_permission`` floor.
 * ``WORKSPACE_GRANTABLE_PERMISSIONS`` excludes it, so the workspace slot does not
 * offer it.
 *
 * ``satisfies Record<(typeof RESOURCE_TYPES)[number], ...>`` keeps this
 * exhaustive: a resource type added above without an entry here fails to
 * compile rather than silently falling back to the default list.
 */
export const PERMISSIONS_FOR_RESOURCE_TYPE = {
  experiment: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  run: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  trace: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  assessment: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  logged_model: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  review_queue: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  registered_model: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  registered_model_version: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  prompt: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  prompt_version: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  scorer: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  scorer_version: ['READ', 'EDIT', 'MANAGE', 'DENY'],
  gateway_secret: ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY'],
  gateway_endpoint: ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY'],
  mcp_server: ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY'],
  mcp_server_version: ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY'],
  workspace: ['USE', 'MANAGE'],
} satisfies Record<(typeof RESOURCE_TYPES)[number], readonly string[]>;

export const getGrantablePermissions = (resourceType: string): readonly string[] =>
  PERMISSIONS_FOR_RESOURCE_TYPE[resourceType as keyof typeof PERMISSIONS_FOR_RESOURCE_TYPE] ?? [
    'READ',
    'EDIT',
    'MANAGE',
    'DENY',
  ];

// ---------------------------------------------------------------------------
// Mutation conditions (condition-based access control)
//
// A condition narrows what a role's grant admits; it never widens it. Two
// independent filters, either of which may be absent:
//
//   valueCondition  -- constrains the values a request may SET (tag_key,
//                      tag_value, alias). Applies on create.
//   targetCondition -- constrains which existing resources may be MUTATED
//                      (tags.<key>, aliases.<name>). Vacuous on create,
//                      because no target state exists yet.
//
// Two properties the UI has to communicate, because neither is what an
// administrator expects from a list of rules:
//
//  1. Objects are CONJUNCTIVE. A role may hold up to 100 per resource type and
//     every applicable one must pass. A list reads like alternatives and most
//     permission systems treat it that way; here "env = dev" plus "owner = me"
//     admits only resources that are both.
//  2. On the TARGET side an absent tag FAILS every comparator. So
//     `tags.lifecycle != 'prod'` denies an UNTAGGED resource. Surprising, and
//     deliberate -- it is the fail-closed reading.

export interface MutationCondition {
  id: number;
  role_id: number;
  resource_type: string;
  /** 1-100. Server-allocated; the client never chooses it. */
  condition_slot: number;
  /** Both parent fields are set together, or both absent for workspace/type scope. */
  parent_resource_type: string | null;
  parent_resource_id: string | null;
  value_condition: string | null;
  target_condition: string | null;
}

export interface MutationConditionResponse {
  mutation_conditions: MutationCondition;
}

export interface ListMutationConditionsResponse {
  mutation_conditions: MutationCondition[];
}

export interface AddMutationConditionRequest {
  role_id: number;
  resource_type: string;
  parent_resource_type?: string | null;
  parent_resource_id?: string | null;
  value_condition?: string | null;
  target_condition?: string | null;
}

/**
 * Add a condition to a user's direct grants. The counterpart of the per-user grant
 * endpoint: the server resolves -- and creates, if needed -- the synthetic role that
 * backs those grants, so no ``role_id`` is supplied and no direct grant has to exist
 * first.
 */
export interface AddUserMutationConditionRequest {
  username: string;
  resource_type: string;
  parent_resource_type?: string | null;
  parent_resource_id?: string | null;
  value_condition?: string | null;
  target_condition?: string | null;
}

export interface UpdateMutationConditionRequest {
  condition_id: number;
  value_condition?: string | null;
  target_condition?: string | null;
  parent_resource_type?: string | null;
  parent_resource_id?: string | null;
  /**
   * The parent pair is replaced as a unit, so a partial update has to say
   * explicitly whether it means to touch the scope -- otherwise omitting the
   * fields is indistinguishable from clearing them.
   */
  update_parent_scope?: boolean;
}

/** Child resource type -> its only supported direct parent for scoped conditions. */
export const CONDITION_PARENT_RESOURCE_TYPES = {
  run: 'experiment',
  trace: 'experiment',
  logged_model: 'experiment',
  registered_model_version: 'registered_model',
  prompt_version: 'prompt',
  mcp_server_version: 'mcp_server',
} satisfies Record<string, string>;

/** Types that have no parent, so a parent scope is rejected for them. */
export const CONDITION_PARENTLESS_RESOURCE_TYPES = ['experiment', 'registered_model', 'prompt', 'mcp_server'];

export const MAX_CONDITIONS_PER_ROLE_TYPE = 100;

/**
 * The resource types a mutation condition may govern. A strict subset of
 * ``RESOURCE_TYPES``: a condition filters on tags or aliases, so a type that carries
 * neither cannot be conditioned. Mirrors ``conditions.SUPPORTED_RESOURCE_TYPES``.
 *
 * ``assessment`` is deliberately absent even though it is a grantable type -- an
 * assessment mutation is gated on the tags of the *trace* it belongs to, not on its own.
 */
export const CONDITION_RESOURCE_TYPES = [
  'experiment',
  'run',
  'trace',
  'logged_model',
  'registered_model',
  'registered_model_version',
  'prompt',
  'prompt_version',
  'mcp_server',
  'mcp_server_version',
] as const;

/**
 * Types that own aliases, and so accept an ``aliases.<name>`` target clause or an
 * ``alias`` request clause. A version's alias list names aliases stored on its parent,
 * so a version is not alias-owning (D18).
 */
export const CONDITION_ALIAS_OWNING_RESOURCE_TYPES = ['registered_model', 'prompt', 'mcp_server'];

export const isConditionAliasOwningType = (resourceType: string): boolean =>
  CONDITION_ALIAS_OWNING_RESOURCE_TYPES.includes(resourceType);

/** The parent type a condition on ``resourceType`` may be scoped to, if any. */
export const getConditionParentType = (resourceType: string): string | undefined =>
  CONDITION_PARENT_RESOURCE_TYPES[resourceType as keyof typeof CONDITION_PARENT_RESOURCE_TYPES];

/**
 * Human label for a condition's scope: either the parent it is confined to, or every
 * resource of its type in the workspace.
 */
export const formatConditionScope = (condition: MutationCondition): string => {
  if (condition.parent_resource_type && condition.parent_resource_id) {
    return `${getResourceTypeLabel(condition.parent_resource_type)} ${condition.parent_resource_id}`;
  }
  return 'All in workspace';
};

/** The request-value identifiers a condition on this type may filter on. */
export const getConditionRequestIdentifiers = (resourceType: string): string[] =>
  isConditionAliasOwningType(resourceType) ? ['tag_key', 'tag_value', 'alias'] : ['tag_key', 'tag_value'];

/**
 * True when the two filters together express nothing. The server refuses such an
 * object -- a condition with neither filter would restrict nothing, so it is not a
 * state worth storing -- and clearing both on an existing one deletes it.
 */
export const isConditionEmpty = (valueCondition?: string | null, targetCondition?: string | null): boolean =>
  !valueCondition?.trim() && !targetCondition?.trim();

/**
 * Parent types the resource picker can enumerate. Mirrors the cases handled by
 * ``useResourceOptionsQuery`` -- there is no MCP-server lister, so a condition scoped to
 * one takes a typed id instead of a picked one.
 *
 * Kept as an explicit list rather than inferred from an empty option set: a workspace
 * that genuinely holds no experiments would otherwise flip the picker to free text and
 * invite a typo'd id, which persists as a scope that silently never matches.
 */
export const CONDITION_PARENT_TYPES_WITH_PICKER = ['experiment', 'registered_model', 'prompt'];

export const conditionParentHasPicker = (parentType: string | undefined): boolean =>
  parentType !== undefined && CONDITION_PARENT_TYPES_WITH_PICKER.includes(parentType);
