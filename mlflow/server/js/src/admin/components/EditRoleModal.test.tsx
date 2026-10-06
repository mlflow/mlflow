import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { fireEvent, renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { EditRoleModal } from './EditRoleModal';

// Capturing mock: the workspace argument is what turns into the
// ``X-MLFLOW-WORKSPACE`` header on the resource-picker list requests.
const mockUseResourceOptionsQuery = jest.fn<(...args: any[]) => any>();
const mockUseWorkspacesEnabled = jest.fn<() => { workspacesEnabled: boolean }>();
const mockAddConditionMutateAsync = jest.fn<(...args: any[]) => any>();

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
  // Conditions now share these modals; stub them so the cases below keep testing
  // what they were written for.
  useRoleMutationConditionsQuery: () => ({ data: { mutation_conditions: [] }, isLoading: false, error: null }),
  useAddMutationCondition: () => ({ mutateAsync: mockAddConditionMutateAsync, isLoading: false }),
  useRemoveMutationCondition: () => ({ mutateAsync: jest.fn(), isLoading: false }),
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
  mockAddConditionMutateAsync.mockReset();
  mockAddConditionMutateAsync.mockResolvedValue({});
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

describe('EditRoleModal — condition scope on the wire', () => {
  it('sends the staged scope rather than letting the server default it', async () => {
    // Same omission as the per-user modal: `resource_pattern` never left the client, and
    // the server normalises an absent scope to the wildcard -- so a condition scoped to
    // one resource was persisted covering the whole workspace. `objectContaining` fails
    // on an ABSENT key, which is what makes this catch it even at the default scope.
    renderWithDesignSystem(<EditRoleModal open onClose={jest.fn()} roleId={1} />);

    expect(await screen.findByText('Add a mutation condition')).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), {
      target: { value: "tags.env = 'dev'" },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    await waitFor(() => expect(mockAddConditionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockAddConditionMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        role_id: 1,
        resource_type: 'experiment',
        target_condition: "tags.env = 'dev'",
        resource_pattern: '*',
        container_resource_type: 'workspace',
        container_resource_pattern: '*',
      }),
    );
  });
});
