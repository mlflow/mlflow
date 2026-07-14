import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { fireEvent, renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { EditAccessModal } from './EditAccessModal';

// Per-test override for ``useUserRolesQuery`` so each case can pick its own
// success / error shape. ``mockReset`` in ``beforeEach`` keeps cross-test
// state from leaking.
const mockUseUserRolesQuery = jest.fn();
// Typed as ``(...args: any[]) => any`` so ``mockResolvedValue`` accepts the
// realistic response shapes the component awaits.
const mockGrantPermissionMutateAsync = jest.fn<(...args: any[]) => any>();
const mockRevokePermissionMutateAsync = jest.fn<(...args: any[]) => any>();
const mockUseWorkspacesEnabled = jest.fn<() => { workspacesEnabled: boolean }>();
const mockUseActiveWorkspace = jest.fn<() => string | null>();

jest.mock('../hooks', () => ({
  AdminQueryKeys: {
    users: ['admin_users'],
    roles: ['admin_roles'],
    roleUsers: (roleId: number) => ['admin_role_users', roleId],
    resourceOptions: (resourceType: string) => ['admin_resource_options', resourceType],
  },
  useCurrentUserIsAdmin: () => false,
  useGrantUserPermission: () => ({ mutateAsync: mockGrantPermissionMutateAsync }),
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
  useRevokeUserPermission: () => ({ mutateAsync: mockRevokePermissionMutateAsync }),
  useRolesQuery: () => ({ data: { roles: [] }, isLoading: false, error: null }),
  useUserRolesQuery: (username: string) => mockUseUserRolesQuery(username),
  useUsersQuery: () => ({
    data: { users: [{ id: 1, username: 'alice', is_admin: false, roles: [] }] },
    isLoading: false,
    error: null,
  }),
  useWorkspaceOptions: () => [],
}));

jest.mock('../../workspaces/utils/WorkspaceUtils', () => ({
  useActiveWorkspace: () => mockUseActiveWorkspace(),
}));

jest.mock('../../workspaces/hooks/useWorkspaces', () => ({
  useWorkspaces: () => ({ workspaces: [], isLoading: false }),
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => mockUseWorkspacesEnabled(),
}));

jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));

beforeEach(() => {
  mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: false });
  // ``null`` is what ``useActiveWorkspace`` actually returns on a
  // single-tenant server.
  mockUseActiveWorkspace.mockReturnValue(null);
});

// Direct grants surface through the synthetic ``__user_<id>__`` role
// (``SYNTHETIC_USER_ROLE_NAME_RE``); the modal flattens its ``permissions``
// into the editable pre-filled list. ``EDIT`` (not ``READ``) so a staged
// grant from the form's defaults can't collide with this row's diff key.
const syntheticUserRole = (workspace: string) => ({
  id: 99,
  name: '__user_1__',
  workspace,
  permissions: [{ resource_type: 'experiment', resource_pattern: '*', permission: 'EDIT' }],
});

describe('EditAccessModal — rolesError handling', () => {
  beforeEach(() => {
    mockUseUserRolesQuery.mockReset();
  });

  it('renders an error Alert when useUserRolesQuery fails', () => {
    mockUseUserRolesQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('boom'),
    });
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);
    expect(screen.getByText('Failed to load access state')).toBeInTheDocument();
    expect(screen.getByText('boom')).toBeInTheDocument();
    // Form sections must be suppressed so the admin can't edit against a
    // phantom empty state — see the ``rolesError`` branch in the modal.
    expect(screen.queryByText('Role assignments')).not.toBeInTheDocument();
    expect(screen.queryByText('Direct permissions')).not.toBeInTheDocument();
  });

  it('disables the Review changes button when useUserRolesQuery fails', () => {
    mockUseUserRolesQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('boom'),
    });
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);
    expect(screen.getByRole('button', { name: /Review changes/ })).toBeDisabled();
  });
});

describe('EditAccessModal — workspace targeting on direct grants and revokes', () => {
  beforeEach(() => {
    mockUseUserRolesQuery.mockReset();
    mockGrantPermissionMutateAsync.mockReset();
    mockRevokePermissionMutateAsync.mockReset();
  });

  it('omits the workspace on grant when workspaces are disabled', async () => {
    // Regression: single-tenant servers reject ANY ``X-MLFLOW-WORKSPACE``
    // header (even ``default``) with FEATURE_DISABLED, so the modal must not
    // coerce "no active workspace" into ``default`` on the grant request.
    mockUseUserRolesQuery.mockReturnValue({ data: { roles: [] }, isLoading: false, error: null });
    mockGrantPermissionMutateAsync.mockResolvedValue({});
    const onClose = jest.fn();
    renderWithDesignSystem(<EditAccessModal open onClose={onClose} username="alice" />);

    // ``fireEvent`` instead of ``userEvent`` throughout this describe: the
    // multi-step flows exceeded jest's default 5s timeout on loaded CI
    // runners with the pointer pipeline, and ``jest.setTimeout`` is banned.
    fireEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    // ``onClose`` is the async submit chain's last step — waiting on it
    // settles the whole chain.
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(mockGrantPermissionMutateAsync).toHaveBeenCalledTimes(1);
    expect(mockGrantPermissionMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        resource_type: 'experiment',
        resource_id: '*',
        username: 'alice',
        permission: 'READ',
      }),
    );
    expect(mockGrantPermissionMutateAsync.mock.calls[0][0].workspace).toBeUndefined();
  });

  it('omits the workspace on revoke when workspaces are disabled', async () => {
    // The revoke call got the same single-tenant fix as grant: removing a
    // pre-filled row must not send an ``X-MLFLOW-WORKSPACE`` header.
    mockUseUserRolesQuery.mockReturnValue({
      data: { roles: [syntheticUserRole('default')] },
      isLoading: false,
      error: null,
    });
    mockRevokePermissionMutateAsync.mockResolvedValue({});
    const onClose = jest.fn();
    renderWithDesignSystem(<EditAccessModal open onClose={onClose} username="alice" />);

    // The pre-fill effect seeds the row asynchronously — removing it stages
    // the revoke.
    fireEvent.click(await screen.findByRole('button', { name: 'Remove experiment *' }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(mockRevokePermissionMutateAsync).toHaveBeenCalledTimes(1);
    expect(mockRevokePermissionMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({ resource_type: 'experiment', resource_id: '*', username: 'alice' }),
    );
    expect(mockRevokePermissionMutateAsync.mock.calls[0][0].workspace).toBeUndefined();
    expect(mockGrantPermissionMutateAsync).not.toHaveBeenCalled();
  });

  it('targets the active workspace on grant and revoke when workspaces are enabled', async () => {
    mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: true });
    // A non-default active workspace: the pre-fix coercion also produced
    // ``'default'``, so asserting ``'default'`` would pass on the broken
    // code too.
    mockUseActiveWorkspace.mockReturnValue('team-a');
    mockUseUserRolesQuery.mockReturnValue({
      data: { roles: [syntheticUserRole('team-a')] },
      isLoading: false,
      error: null,
    });
    mockGrantPermissionMutateAsync.mockResolvedValue({});
    mockRevokePermissionMutateAsync.mockResolvedValue({});
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);

    // Stage a revoke (remove the pre-filled ``EDIT`` row) first, then a
    // grant (the form's default ``READ`` — a different diff key, so the
    // pair can't cancel out) — one submit applies both.
    fireEvent.click(await screen.findByRole('button', { name: 'Remove experiment *' }));
    fireEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    // Revokes run after grants in the submit chain — waiting on the revoke
    // settles both.
    await waitFor(() => expect(mockRevokePermissionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockGrantPermissionMutateAsync).toHaveBeenCalledTimes(1);
    expect(mockGrantPermissionMutateAsync.mock.calls[0][0].workspace).toBe('team-a');
    expect(mockRevokePermissionMutateAsync.mock.calls[0][0].workspace).toBe('team-a');
  });
});
