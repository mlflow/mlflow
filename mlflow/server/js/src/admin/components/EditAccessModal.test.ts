import { describe, it, expect } from '@jest/globals';

import { computeAccessDiff } from './EditAccessModal';
import type { StagedDirectPermission } from './DirectPermissionsSection';

const direct = (overrides: Partial<StagedDirectPermission> = {}): StagedDirectPermission => ({
  resourceType: 'experiment',
  resourceId: '42',
  permission: 'EDIT',
  ...overrides,
});

describe('computeAccessDiff', () => {
  it('returns an empty diff when current and desired are identical', () => {
    const snap = {
      roleIds: [1, 2, 3],
      directPermissions: [direct({ resourceId: '42' }), direct({ resourceId: '7' })],
      isAdmin: false,
    };
    const diff = computeAccessDiff(snap, snap, true);
    expect(diff).toEqual({
      rolesToAssign: [],
      rolesToUnassign: [],
      directToGrant: [],
      directToRevoke: [],
      adminChange: false,
    });
  });

  it('detects roles to assign without disturbing existing roles', () => {
    const diff = computeAccessDiff(
      { roleIds: [1, 2], directPermissions: [], isAdmin: false },
      { roleIds: [1, 2, 3, 4], directPermissions: [], isAdmin: false },
      false,
    );
    expect(diff.rolesToAssign).toEqual([3, 4]);
    expect(diff.rolesToUnassign).toEqual([]);
  });

  it('detects roles to unassign without disturbing remaining roles', () => {
    const diff = computeAccessDiff(
      { roleIds: [1, 2, 3], directPermissions: [], isAdmin: false },
      { roleIds: [1], directPermissions: [], isAdmin: false },
      false,
    );
    expect(diff.rolesToAssign).toEqual([]);
    expect(diff.rolesToUnassign).toEqual([2, 3]);
  });

  it('handles a swap of role assignments (assign and unassign in the same diff)', () => {
    const diff = computeAccessDiff(
      { roleIds: [1, 2], directPermissions: [], isAdmin: false },
      { roleIds: [2, 3], directPermissions: [], isAdmin: false },
      false,
    );
    expect(diff.rolesToAssign).toEqual([3]);
    expect(diff.rolesToUnassign).toEqual([1]);
  });

  it('detects direct permissions to grant and to revoke', () => {
    const a = direct({ resourceId: '42', permission: 'READ' });
    const b = direct({ resourceId: '99', permission: 'EDIT' });
    const c = direct({ resourceId: '7', permission: 'READ' });
    const diff = computeAccessDiff(
      { roleIds: [], directPermissions: [a, b], isAdmin: false },
      { roleIds: [], directPermissions: [b, c], isAdmin: false },
      false,
    );
    expect(diff.directToGrant).toEqual([c]);
    expect(diff.directToRevoke).toEqual([a]);
  });

  it('treats permission-level changes on the same resource as revoke + grant', () => {
    // Same (resource_type, resource_id), different permission level — the key
    // includes the permission, so the old grant must be revoked and the new
    // one granted. Operators see both rows in the Review step.
    const oldGrant = direct({ resourceId: '42', permission: 'READ' });
    const newGrant = direct({ resourceId: '42', permission: 'EDIT' });
    const diff = computeAccessDiff(
      { roleIds: [], directPermissions: [oldGrant], isAdmin: false },
      { roleIds: [], directPermissions: [newGrant], isAdmin: false },
      false,
    );
    expect(diff.directToGrant).toEqual([newGrant]);
    expect(diff.directToRevoke).toEqual([oldGrant]);
  });

  it('reports adminChange=true when the platform admin promotes a non-admin', () => {
    const diff = computeAccessDiff(
      { roleIds: [], directPermissions: [], isAdmin: false },
      { roleIds: [], directPermissions: [], isAdmin: true },
      true,
    );
    expect(diff.adminChange).toBe(true);
  });

  it('reports adminChange=true when the platform admin demotes an admin', () => {
    const diff = computeAccessDiff(
      { roleIds: [], directPermissions: [], isAdmin: true },
      { roleIds: [], directPermissions: [], isAdmin: false },
      true,
    );
    expect(diff.adminChange).toBe(true);
  });

  it('reports adminChange=false when the desired admin status matches the current', () => {
    const diff = computeAccessDiff(
      { roleIds: [], directPermissions: [], isAdmin: true },
      { roleIds: [], directPermissions: [], isAdmin: true },
      true,
    );
    expect(diff.adminChange).toBe(false);
  });

  it('suppresses adminChange for non-platform-admin viewers regardless of desired flag', () => {
    // A workspace manager opens the modal — the admin-status section is not
    // rendered for them, but defensively the diff still reports no change so
    // the apply step never tries to call ``updateAdmin``.
    const diff = computeAccessDiff(
      { roleIds: [], directPermissions: [], isAdmin: false },
      { roleIds: [], directPermissions: [], isAdmin: true },
      false,
    );
    expect(diff.adminChange).toBe(false);
  });

  it('combines role, direct-permission, and admin-status changes into one diff', () => {
    const oldGrant = direct({ resourceId: '42', permission: 'READ' });
    const newGrant = direct({ resourceId: '99', permission: 'EDIT' });
    const diff = computeAccessDiff(
      { roleIds: [1, 2], directPermissions: [oldGrant], isAdmin: false },
      { roleIds: [2, 3], directPermissions: [newGrant], isAdmin: true },
      true,
    );
    expect(diff).toEqual({
      rolesToAssign: [3],
      rolesToUnassign: [1],
      directToGrant: [newGrant],
      directToRevoke: [oldGrant],
      adminChange: true,
    });
  });
});
