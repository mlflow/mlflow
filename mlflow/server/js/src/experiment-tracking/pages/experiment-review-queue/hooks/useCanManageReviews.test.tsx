import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import type { ListMyPermissionsResponse, UserRolePermissionRow } from '../../../../account/types';
import { useCanEditReviews, useCanManageReviews } from './useCanManageReviews';

let mockAuthAvailable = true;
let mockPermissions: ListMyPermissionsResponse | undefined = { is_admin: false, permissions: [] };
let mockActiveWorkspace: string | null = 'ws-a';
let mockWorkspacesEnabled = true;
let mockWorkspacesLoading = false;

jest.mock('../../../../account/hooks', () => ({
  useIsAuthAvailable: () => mockAuthAvailable,
  useMyPermissionsQuery: () => ({ data: mockPermissions }),
}));
jest.mock('../../../../workspaces/utils/WorkspaceUtils', () => ({
  useActiveWorkspace: () => mockActiveWorkspace,
}));
jest.mock('../../../hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => ({ workspacesEnabled: mockWorkspacesEnabled, loading: mockWorkspacesLoading }),
}));

const perm = (over: Partial<UserRolePermissionRow> = {}): UserRolePermissionRow => ({
  role_id: 1,
  role_name: '__user_1__',
  workspace: 'ws-a',
  resource_type: 'experiment',
  resource_pattern: '*',
  permission: 'MANAGE',
  ...over,
});

const canManage = (experimentId = 'exp-1') => renderHook(() => useCanManageReviews(experimentId)).result.current;
const canEdit = (experimentId = 'exp-1') => renderHook(() => useCanEditReviews(experimentId)).result.current;

describe('useCanManageReviews / useCanEditReviews', () => {
  beforeEach(() => {
    mockAuthAvailable = true;
    mockActiveWorkspace = 'ws-a';
    mockWorkspacesEnabled = true;
    mockWorkspacesLoading = false;
    mockPermissions = { is_admin: false, permissions: [] };
  });

  it('grants on a no-auth server regardless of permissions/workspace', () => {
    mockAuthAvailable = false;
    mockPermissions = undefined;
    expect(canManage()).toBe(true);
  });

  it('denies while permissions are still loading', () => {
    mockPermissions = undefined;
    expect(canManage()).toBe(false);
  });

  it('grants everything to an admin', () => {
    mockPermissions = { is_admin: true, permissions: [] };
    expect(canManage()).toBe(true);
  });

  it('grants when a matching wildcard grant is in the active workspace', () => {
    mockPermissions = { is_admin: false, permissions: [perm({ workspace: 'ws-a' })] };
    expect(canManage()).toBe(true);
  });

  it('denies when the only matching wildcard grant is in a different workspace', () => {
    // The bug: a `*` grant from another workspace must not unlock controls here.
    mockPermissions = { is_admin: false, permissions: [perm({ workspace: 'ws-other' })] };
    expect(canManage()).toBe(false);
  });

  it('matches an exact experiment grant in the active workspace', () => {
    mockPermissions = { is_admin: false, permissions: [perm({ resource_pattern: 'exp-1', workspace: 'ws-a' })] };
    expect(canManage()).toBe(true);
  });

  it('denies an exact experiment grant carried by a different workspace', () => {
    mockPermissions = { is_admin: false, permissions: [perm({ resource_pattern: 'exp-1', workspace: 'ws-other' })] };
    expect(canManage()).toBe(false);
  });

  it('ignores workspace when the feature is definitively disabled', () => {
    mockWorkspacesEnabled = false;
    mockWorkspacesLoading = false;
    mockActiveWorkspace = null;
    mockPermissions = { is_admin: false, permissions: [perm({ workspace: 'default' })] };
    expect(canManage()).toBe(true);
  });

  it('denies during the load window rather than over-revealing (enabled, workspace unresolved)', () => {
    // workspaces enabled but the active workspace hasn't resolved yet: don't skip
    // the filter on the strength of a null workspace -- deny until it's known.
    mockWorkspacesEnabled = true;
    mockActiveWorkspace = null;
    mockPermissions = { is_admin: false, permissions: [perm({ workspace: 'ws-a' })] };
    expect(canManage()).toBe(false);
  });

  it('denies while the workspaces-enabled flag is still loading', () => {
    mockWorkspacesEnabled = false;
    mockWorkspacesLoading = true;
    mockActiveWorkspace = null;
    mockPermissions = { is_admin: false, permissions: [perm({ workspace: 'ws-a' })] };
    expect(canManage()).toBe(false);
  });

  it('requires MANAGE for manage, while EDIT alone satisfies edit', () => {
    mockPermissions = { is_admin: false, permissions: [perm({ permission: 'EDIT', workspace: 'ws-a' })] };
    expect(canManage()).toBe(false);
    expect(canEdit()).toBe(true);
  });
});
