import { describe, it, expect } from '@jest/globals';

import { computeRoleDiff } from './EditRoleModal';
import type { StagedRolePermission } from './RolePermissionsSection';

const perm = (overrides: Partial<StagedRolePermission> = {}): StagedRolePermission => ({
  resourceType: 'experiment',
  resourcePattern: '42',
  permission: 'EDIT',
  ...overrides,
});

describe('computeRoleDiff', () => {
  it('returns an empty diff when current and desired are identical', () => {
    const snap = {
      name: 'editor',
      description: 'edits stuff',
      permissions: [perm({ id: 1 }), perm({ id: 2, resourcePattern: '7' })],
      usernames: ['alice', 'bob'],
    };
    const diff = computeRoleDiff(snap, snap);
    expect(diff).toEqual({
      nameChange: null,
      descriptionChange: null,
      permissionsToAdd: [],
      permissionIdsToRemove: [],
      usersToAssign: [],
      usersToUnassign: [],
    });
  });

  describe('name change', () => {
    it('reports the trimmed new name when it differs from the current', () => {
      const diff = computeRoleDiff(
        { name: 'old', description: '', permissions: [], usernames: [] },
        { name: '  new  ', description: '', permissions: [], usernames: [] },
      );
      expect(diff.nameChange).toBe('new');
    });

    it('reports null when the trimmed name matches (whitespace-only edit is a no-op)', () => {
      const diff = computeRoleDiff(
        { name: 'editor', description: '', permissions: [], usernames: [] },
        { name: '  editor  ', description: '', permissions: [], usernames: [] },
      );
      expect(diff.nameChange).toBeNull();
    });

    it('reports null when the desired name is empty (the modal blocks empty-name submit)', () => {
      const diff = computeRoleDiff(
        { name: 'editor', description: '', permissions: [], usernames: [] },
        { name: '   ', description: '', permissions: [], usernames: [] },
      );
      expect(diff.nameChange).toBeNull();
    });
  });

  describe('description change', () => {
    it('reports the new description when it differs', () => {
      const diff = computeRoleDiff(
        { name: 'r', description: 'old desc', permissions: [], usernames: [] },
        { name: 'r', description: 'new desc', permissions: [], usernames: [] },
      );
      expect(diff.descriptionChange).toBe('new desc');
    });

    it('reports the empty string when the user clears a previously-set description', () => {
      // Empty string is meaningful ("clear the description"); only null means
      // "no change". This distinction matters at the apply step.
      const diff = computeRoleDiff(
        { name: 'r', description: 'old desc', permissions: [], usernames: [] },
        { name: 'r', description: '', permissions: [], usernames: [] },
      );
      expect(diff.descriptionChange).toBe('');
    });

    it('reports null when the description is unchanged', () => {
      const diff = computeRoleDiff(
        { name: 'r', description: 'same', permissions: [], usernames: [] },
        { name: 'r', description: 'same', permissions: [], usernames: [] },
      );
      expect(diff.descriptionChange).toBeNull();
    });
  });

  describe('permissions', () => {
    it('treats staged rows without an id as additions', () => {
      const newRow = perm({ id: undefined, resourcePattern: '99' });
      const diff = computeRoleDiff(
        { name: 'r', description: '', permissions: [perm({ id: 1 })], usernames: [] },
        { name: 'r', description: '', permissions: [perm({ id: 1 }), newRow], usernames: [] },
      );
      expect(diff.permissionsToAdd).toEqual([newRow]);
      expect(diff.permissionIdsToRemove).toEqual([]);
    });

    it('treats current rows whose triple is no longer staged as removals', () => {
      const diff = computeRoleDiff(
        {
          name: 'r',
          description: '',
          permissions: [perm({ id: 1 }), perm({ id: 2, resourcePattern: '7' })],
          usernames: [],
        },
        { name: 'r', description: '', permissions: [perm({ id: 1 })], usernames: [] },
      );
      expect(diff.permissionsToAdd).toEqual([]);
      expect(diff.permissionIdsToRemove).toEqual([2]);
    });

    it('treats a permission-level change as a removal of the old id plus an add for the new triple', () => {
      // Same (resourceType, resourcePattern), different permission level: the
      // dedup key includes ``permission``, so the old persisted row is
      // removed and the new staged row (id=undefined) is added.
      const oldRow = perm({ id: 1, permission: 'READ' });
      const newRow = perm({ id: undefined, permission: 'EDIT' });
      const diff = computeRoleDiff(
        { name: 'r', description: '', permissions: [oldRow], usernames: [] },
        { name: 'r', description: '', permissions: [newRow], usernames: [] },
      );
      expect(diff.permissionsToAdd).toEqual([newRow]);
      expect(diff.permissionIdsToRemove).toEqual([1]);
    });
  });

  describe('users', () => {
    it('detects users to assign and to unassign in the same diff', () => {
      const diff = computeRoleDiff(
        { name: 'r', description: '', permissions: [], usernames: ['alice', 'bob'] },
        { name: 'r', description: '', permissions: [], usernames: ['bob', 'carol'] },
      );
      expect(diff.usersToAssign).toEqual(['carol']);
      expect(diff.usersToUnassign).toEqual(['alice']);
    });

    it('returns empty user changes when the assignment set is unchanged', () => {
      const diff = computeRoleDiff(
        { name: 'r', description: '', permissions: [], usernames: ['alice', 'bob'] },
        { name: 'r', description: '', permissions: [], usernames: ['alice', 'bob'] },
      );
      expect(diff.usersToAssign).toEqual([]);
      expect(diff.usersToUnassign).toEqual([]);
    });
  });

  it('combines name, description, permissions, and user changes into one diff', () => {
    const diff = computeRoleDiff(
      {
        name: 'old-name',
        description: 'old desc',
        permissions: [perm({ id: 1, resourcePattern: '42' })],
        usernames: ['alice'],
      },
      {
        name: 'new-name',
        description: '',
        permissions: [perm({ id: undefined, resourcePattern: '99' })],
        usernames: ['bob'],
      },
    );
    expect(diff).toEqual({
      nameChange: 'new-name',
      descriptionChange: '',
      permissionsToAdd: [perm({ id: undefined, resourcePattern: '99' })],
      permissionIdsToRemove: [1],
      usersToAssign: ['bob'],
      usersToUnassign: ['alice'],
    });
  });
});
