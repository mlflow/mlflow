import { describe, it, expect } from '@jest/globals';
import {
  PERMISSIONS,
  PERMISSIONS_FOR_RESOURCE_TYPE,
  RESOURCE_TYPES,
  RESOURCE_TYPE_LABELS,
  WILDCARD_ONLY_RESOURCE_TYPES,
  getGrantablePermissions,
  getResourceTypeLabel,
  isWildcardOnlyResourceType,
} from './types';

describe('getGrantablePermissions', () => {
  // Mirrors the backend's `_validate_permission_for_resource_type` in
  // `mlflow/server/auth/permissions.py`. Keep this table in sync.
  //
  // `RESOURCE_GRANTABLE_PERMISSIONS` there is (DENY, EDIT, MANAGE, READ, USE);
  // `WORKSPACE_GRANTABLE_PERMISSIONS` is (MANAGE, USE). `USE` is hidden on types
  // where it is a no-op over `READ`.
  it.each([
    ['experiment', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['run', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['trace', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['assessment', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['logged_model', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['review_queue', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['registered_model', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['registered_model_version', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['prompt', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['prompt_version', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['scorer', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['scorer_version', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['gateway_secret', ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY']],
    ['gateway_endpoint', ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY']],
    ['mcp_server', ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY']],
    ['mcp_server_version', ['READ', 'USE', 'EDIT', 'MANAGE', 'DENY']],
    ['skill', ['READ', 'EDIT', 'MANAGE', 'DENY']],
    ['workspace', ['USE', 'MANAGE']],
  ])('returns the backend-allowed set for %s', (resourceType, expected) => {
    expect(getGrantablePermissions(resourceType)).toEqual(expected);
  });

  it('offers DENY last on every non-workspace type', () => {
    // DENY is a veto, not the weakest rung — it beats any positive grant on the
    // same key. It sits at the end of the list and is styled apart so it cannot
    // be misread as "less than READ".
    for (const resourceType of RESOURCE_TYPES) {
      if (resourceType === 'workspace') continue;
      const levels = getGrantablePermissions(resourceType);
      expect(levels[levels.length - 1]).toBe('DENY');
    }
  });

  it('never offers DENY at workspace scope', () => {
    // The backend's WORKSPACE_GRANTABLE_PERMISSIONS is (MANAGE, USE) only.
    expect(getGrantablePermissions('workspace')).not.toContain('DENY');
  });

  it('never includes NO_PERMISSIONS for any known type', () => {
    // `_validate_permission_for_resource_type` rejects it for every type: an
    // absent grant plus `default_permission` already expresses "no access".
    for (const resourceType of RESOURCE_TYPES) {
      expect(getGrantablePermissions(resourceType)).not.toContain('NO_PERMISSIONS');
    }
    expect(PERMISSIONS).not.toContain('NO_PERMISSIONS');
  });

  it('falls back to the resource-level set for unknown types', () => {
    // Keeps a future backend resource type safe by default (no USE) while still
    // exposing the veto.
    expect(getGrantablePermissions('something_new')).toEqual(['READ', 'EDIT', 'MANAGE', 'DENY']);
  });
});

describe('resource type tables', () => {
  it('gives every surfaced type a label and a permission list', () => {
    // `satisfies` enforces both at compile time; this pins it at runtime too, so
    // a type added to RESOURCE_TYPES cannot silently fall back to the defaults.
    for (const resourceType of RESOURCE_TYPES) {
      expect(RESOURCE_TYPE_LABELS[resourceType]).toBeTruthy();
      expect(getResourceTypeLabel(resourceType)).not.toBe(resourceType);
      expect(PERMISSIONS_FOR_RESOURCE_TYPE[resourceType]).toBeTruthy();
    }
  });

  it('omits gateway_model_definition until it is user-facing', () => {
    expect(RESOURCE_TYPES).not.toContain('gateway_model_definition');
  });
});

describe('isWildcardOnlyResourceType', () => {
  it('matches the backend TYPE grain map', () => {
    // Wildcard-only in `permissions.py`'s TYPE: these carry PatternKind.WILDCARD
    // alone, so a grant can never name a single row.
    expect([...WILDCARD_ONLY_RESOURCE_TYPES].sort()).toEqual(
      [
        'assessment',
        'logged_model',
        'mcp_server_version',
        'prompt_version',
        'registered_model_version',
        'review_queue',
        'run',
        'scorer_version',
        'trace',
        'workspace',
      ].sort(),
    );
  });

  it('treats the id-capable types as targetable', () => {
    for (const resourceType of [
      'experiment',
      'registered_model',
      'prompt',
      'scorer',
      'gateway_secret',
      'gateway_endpoint',
      'mcp_server',
      'skill',
    ]) {
      expect(isWildcardOnlyResourceType(resourceType)).toBe(false);
    }
  });

  it('reports every wildcard-only type as such', () => {
    for (const resourceType of WILDCARD_ONLY_RESOURCE_TYPES) {
      expect(isWildcardOnlyResourceType(resourceType)).toBe(true);
    }
  });

  it('does not claim an unknown type is wildcard-only', () => {
    expect(isWildcardOnlyResourceType('something_new')).toBe(false);
  });
});
