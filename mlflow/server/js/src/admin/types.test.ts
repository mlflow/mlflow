import { describe, it, expect } from '@jest/globals';
import { getGrantablePermissions } from './types';

describe('getGrantablePermissions', () => {
  // Mirrors the backend's `_validate_permission_for_resource_type` in
  // `mlflow/server/auth/permissions.py`. Keep this table in sync.
  it.each([
    ['experiment', ['READ', 'EDIT', 'MANAGE']],
    ['registered_model', ['READ', 'EDIT', 'MANAGE']],
    ['prompt', ['READ', 'EDIT', 'MANAGE']],
    ['scorer', ['READ', 'EDIT', 'MANAGE']],
    ['gateway_secret', ['READ', 'USE', 'EDIT', 'MANAGE']],
    ['gateway_endpoint', ['READ', 'USE', 'EDIT', 'MANAGE']],
    ['workspace', ['USE', 'MANAGE']],
  ])('returns the backend-allowed set for %s', (resourceType, expected) => {
    expect(getGrantablePermissions(resourceType)).toEqual(expected);
  });

  it('never includes NO_PERMISSIONS for any known type', () => {
    const types = [
      'experiment',
      'registered_model',
      'prompt',
      'scorer',
      'gateway_secret',
      'gateway_endpoint',
      'workspace',
    ];
    for (const t of types) {
      expect(getGrantablePermissions(t)).not.toContain('NO_PERMISSIONS');
    }
  });

  it('falls back to the resource-level set for unknown types', () => {
    // Keeps future backend resource types safe by default (no USE, no
    // NO_PERMISSIONS) until they're added explicitly.
    expect(getGrantablePermissions('something_new')).toEqual(['READ', 'EDIT', 'MANAGE']);
  });
});
