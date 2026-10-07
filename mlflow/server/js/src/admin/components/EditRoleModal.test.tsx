import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { EditRoleModal } from './EditRoleModal';

// Capturing mock: the workspace argument is what turns into the
// ``X-MLFLOW-WORKSPACE`` header on the resource-picker list requests.
const mockUseResourceOptionsQuery = jest.fn<(...args: any[]) => any>();
const mockUseWorkspacesEnabled = jest.fn<() => { workspacesEnabled: boolean }>();

jest.mock('../hooks', () => ({
  useUpdateRole: () => ({ mutateAsync: jest.fn() }),
  useAddPermission: () => ({ mutateAsync: jest.fn() }),
  useRemovePermission: () => ({ mutateAsync: jest.fn() }),
  useAssignRole: () => ({ mutateAsync: jest.fn() }),
  useUnassignRole: () => ({ mutateAsync: jest.fn() }),
  useRoleDetailQuery: () => ({
    data: { role: { id: 1, name: 'team-role', description: '', workspace: 'team-a', permissions: [] } },
    isLoading: false,
  }),
  useRoleUsersQuery: () => ({ data: { assignments: [] }, isLoading: false }),
  useUsersQuery: () => ({ data: { users: [] }, isLoading: false, error: null }),
  useResourceOptionsQuery: (resourceType: string, workspace?: string) =>
    mockUseResourceOptionsQuery(resourceType, workspace),
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => mockUseWorkspacesEnabled(),
}));

beforeEach(() => {
  mockUseResourceOptionsQuery.mockReset();
  mockUseResourceOptionsQuery.mockReturnValue({ options: [], isLoading: false, error: null });
  mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: false });
});

describe('EditRoleModal — workspace targeting on the resource picker', () => {
  it('omits the workspace when workspaces are disabled', async () => {
    // Regression: single-tenant servers reject ANY ``X-MLFLOW-WORKSPACE``
    // header with FEATURE_DISABLED, so the picker query must not receive the
    // role's stored workspace when the workspace feature is off.
    renderWithDesignSystem(<EditRoleModal open onClose={jest.fn()} roleId={1} />);

    expect(await screen.findByText('Add a permission')).toBeInTheDocument();
    expect(mockUseResourceOptionsQuery).toHaveBeenCalledWith('experiment', undefined);
  });

  it("targets the role's workspace when workspaces are enabled", async () => {
    mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: true });
    renderWithDesignSystem(<EditRoleModal open onClose={jest.fn()} roleId={1} />);

    expect(await screen.findByText('Add a permission')).toBeInTheDocument();
    expect(mockUseResourceOptionsQuery).toHaveBeenCalledWith('experiment', 'team-a');
  });
});
