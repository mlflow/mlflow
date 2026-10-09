import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { fireEvent, renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { CreateRoleModal } from './CreateRoleModal';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const mockCreateRoleMutateAsync = jest.fn<(...args: any[]) => any>();
// Capturing mock: the workspace argument is what turns into the
// ``X-MLFLOW-WORKSPACE`` header on the resource-picker list requests.
const mockUseResourceOptionsQuery = jest.fn<(...args: any[]) => any>();
const mockUseWorkspacesEnabled = jest.fn<() => { workspacesEnabled: boolean }>();
const mockAddMutationCondition = jest.fn<(...args: any[]) => any>();
const mockAddPermission = jest.fn<(...args: any[]) => any>();
const mockAssignRole = jest.fn<(...args: any[]) => any>();

jest.mock('../hooks', () => ({
  useCreateRole: () => ({ mutateAsync: mockCreateRoleMutateAsync }),
  useResourceOptionsQuery: (resourceType: string, workspace?: string) =>
    mockUseResourceOptionsQuery(resourceType, workspace),
  useUsersQuery: () => ({ data: { users: [] }, isLoading: false, error: null }),
}));

jest.mock('../api', () => ({
  AdminApi: {
    addMutationCondition: (...args: any[]) => mockAddMutationCondition(...args),
    addPermission: (...args: any[]) => mockAddPermission(...args),
    assignRole: (...args: any[]) => mockAssignRole(...args),
  },
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
  mockCreateRoleMutateAsync.mockResolvedValue({ role: { id: 7 } });
  mockAddMutationCondition.mockReset();
  mockAddMutationCondition.mockResolvedValue({ mutation_conditions: { id: 11 } });
  mockAddPermission.mockReset();
  mockAddPermission.mockResolvedValue({});
  mockAssignRole.mockReset();
  mockAssignRole.mockResolvedValue({});
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

describe('CreateRoleModal — restrictions land before capability', () => {
  // The create flow had no conditions step at all, so an admin building a deliberately
  // narrowed role had to create it WITH its permissions and WITH its users -- the role is
  // live and unrestricted the moment that submit returns -- and then reopen Edit role to
  // add the restrictions. Every assigned user held the unnarrowed grant for that whole
  // window. A new role grants nothing until a permission is attached, so the conditions
  // can and must be applied first.

  const fillAndSubmit = async () => {
    fireEvent.change(screen.getByPlaceholderText('Enter role name'), { target: { value: 'writers' } });
    fireEvent.click(screen.getByRole('button', { name: /Mutation conditions/ }));
    fireEvent.change(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), {
      target: { value: "tags.env = 'dev'" },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));
    fireEvent.click(screen.getByRole('button', { name: /Permissions/ }));
    fireEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Create role$/ }));
  };

  it('creates the condition before the permission it narrows', async () => {
    const order: string[] = [];
    mockAddMutationCondition.mockImplementation(async () => {
      order.push('condition');
      return { mutation_conditions: { id: 11 } };
    });
    mockAddPermission.mockImplementation(async () => {
      order.push('permission');
      return {};
    });
    const onClose = jest.fn();
    renderWithDesignSystem(<CreateRoleModal open onClose={onClose} />);

    await fillAndSubmit();

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(order).toEqual(['condition', 'permission']);
  });

  it('sends the staged scope rather than letting the server default it', async () => {
    renderWithDesignSystem(<CreateRoleModal open onClose={jest.fn()} />);

    await fillAndSubmit();

    // `objectContaining` fails on an ABSENT key, which is what makes this catch the
    // omission even at the default scope -- the server normalises an absent
    // `resource_pattern` to the wildcard.
    await waitFor(() =>
      expect(mockAddMutationCondition).toHaveBeenCalledWith(
        expect.objectContaining({
          role_id: 7,
          resource_type: 'experiment',
          resource_pattern: '*',
          container_resource_type: 'workspace',
          container_resource_pattern: '*',
          target_condition: "tags.env = 'dev'",
        }),
      ),
    );
  });

  it('adds no permission and assigns no user when the condition failed', async () => {
    // The whole point. Granting here would hand assigned users exactly the unrestricted
    // access the condition was meant to take away.
    mockAddMutationCondition.mockRejectedValue(new Error('slot limit reached'));
    const onClose = jest.fn();
    renderWithDesignSystem(<CreateRoleModal open onClose={onClose} />);

    await fillAndSubmit();

    await waitFor(() => expect(screen.getByText(/slot limit reached/)).toBeInTheDocument());
    expect(mockAddPermission).not.toHaveBeenCalled();
    expect(mockAssignRole).not.toHaveBeenCalled();
    // The role itself exists, so the modal must stay open for the retry rather than
    // reporting success.
    expect(onClose).not.toHaveBeenCalled();
  });

  it('does not re-create a condition that already landed when a retry happens', async () => {
    // Each add allocates a fresh slot rather than deduplicating, so a replay leaves a
    // duplicate restriction behind and spends the per-type limit.
    mockAddPermission.mockRejectedValue(new Error('permission refused'));
    renderWithDesignSystem(<CreateRoleModal open onClose={jest.fn()} />);

    await fillAndSubmit();

    await waitFor(() => expect(screen.getByText(/permission refused/)).toBeInTheDocument());
    expect(mockAddMutationCondition).toHaveBeenCalledTimes(1);

    // The button relabels itself once the role exists.
    fireEvent.click(screen.getByRole('button', { name: /^Retry failed grants$/ }));
    await waitFor(() => expect(mockAddPermission).toHaveBeenCalledTimes(2));
    // The role is not re-created either, and the condition is not replayed.
    expect(mockCreateRoleMutateAsync).toHaveBeenCalledTimes(1);
    expect(mockAddMutationCondition).toHaveBeenCalledTimes(1);
  });
});
