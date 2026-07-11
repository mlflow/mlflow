import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { EditAccessModal } from './EditAccessModal';

// Increase timeout: the grant-flow tests drive multi-step userEvent
// interactions through the full modal, which exceeds the default 5s
// on slower CI runners.
jest.setTimeout(30000);

// Per-test override for ``useUserRolesQuery`` so each case can pick its own
// success / error shape. ``mockReset`` in ``beforeEach`` keeps cross-test
// state from leaking.
const mockUseUserRolesQuery = jest.fn();
// Typed as ``(...args: any[]) => any`` so ``mockResolvedValue`` accepts the
// realistic response shapes the component awaits.
const mockGrantPermissionMutateAsync = jest.fn<(...args: any[]) => any>();

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
  useRevokeUserPermission: () => ({ mutateAsync: jest.fn() }),
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
  useActiveWorkspace: () => null,
}));

jest.mock('../../workspaces/hooks/useWorkspaces', () => ({
  useWorkspaces: () => ({ workspaces: [], isLoading: false }),
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => ({ workspacesEnabled: false }),
}));

jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));

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

describe('EditAccessModal — workspace targeting on direct grants', () => {
  beforeEach(() => {
    mockUseUserRolesQuery.mockReset();
    mockGrantPermissionMutateAsync.mockReset();
  });

  it('omits the workspace on grant when workspaces are disabled', async () => {
    // Regression: single-tenant servers reject ANY ``X-MLFLOW-WORKSPACE``
    // header (even ``default``) with FEATURE_DISABLED, so the modal must not
    // coerce "no active workspace" into ``default`` on the grant request.
    const userEvent = userEventGlobal.setup();
    mockUseUserRolesQuery.mockReturnValue({ data: { roles: [] }, isLoading: false, error: null });
    mockGrantPermissionMutateAsync.mockResolvedValue({});
    const onClose = jest.fn();
    renderWithDesignSystem(<EditAccessModal open onClose={onClose} username="alice" />);

    await userEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    await userEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    await userEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    await userEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

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
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
