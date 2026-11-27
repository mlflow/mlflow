import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { fireEvent, renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { CreateRoleModal } from './CreateRoleModal';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const mockCreateRoleMutateAsync = jest.fn<(...args: any[]) => any>();
// Capturing mock: the workspace argument is what turns into the
// ``X-MLFLOW-WORKSPACE`` header on the resource-picker list requests.
const mockUseResourceOptionsQuery = jest.fn<(...args: any[]) => any>();
const mockUseWorkspacesEnabled = jest.fn<() => { workspacesEnabled: boolean }>();

jest.mock('../hooks', () => ({
  useCreateRole: () => ({ mutateAsync: mockCreateRoleMutateAsync }),
  useResourceOptionsQuery: (resourceType: string, workspace?: string) =>
    mockUseResourceOptionsQuery(resourceType, workspace),
  useUsersQuery: () => ({ data: { users: [] }, isLoading: false, error: null }),
}));

jest.mock('../../workspaces/hooks/useWorkspaces', () => ({
  useWorkspaces: () => ({ workspaces: [{ name: 'default' }, { name: 'team-a' }], isLoading: false }),
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => mockUseWorkspacesEnabled(),
}));

beforeEach(() => {
  mockCreateRoleMutateAsync.mockReset();
  mockUseResourceOptionsQuery.mockReset();
  mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
  mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: false });
});

describe('CreateRoleModal — workspace targeting on the resource picker', () => {
  it('omits the workspace when workspaces are disabled', () => {
    // Regression: single-tenant servers reject ANY ``X-MLFLOW-WORKSPACE``
    // header (even ``default``) with FEATURE_DISABLED, so the picker query
    // must not receive the modal's ``default``-initialized workspace state.
    renderWithDesignSystem(<CreateRoleModal open onClose={jest.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: /Permissions/ }));

    expect(mockUseResourceOptionsQuery).toHaveBeenCalledWith('experiment', undefined);
  });

  it('targets the selected workspace when workspaces are enabled', async () => {
    mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: true });
    renderWithDesignSystem(<CreateRoleModal open onClose={jest.fn()} />);

    // A non-default workspace: the state initializes to ``'default'``, so
    // asserting ``'default'`` here would pass on the broken code too.
    await userEvent.click(document.getElementById('admin-create-role-modal-workspace')!);
    await userEvent.click(await screen.findByRole('option', { name: 'team-a' }));
    fireEvent.click(screen.getByRole('button', { name: /Permissions/ }));

    expect(mockUseResourceOptionsQuery).toHaveBeenLastCalledWith('experiment', 'team-a');
  });
});
