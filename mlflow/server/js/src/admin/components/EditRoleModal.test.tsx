import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { fireEvent, renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { EditRoleModal } from './EditRoleModal';

// Capturing mock: the workspace argument is what turns into the
// ``X-MLFLOW-WORKSPACE`` header on the resource-picker list requests.
const mockUseResourceOptionsQuery = jest.fn<(...args: any[]) => any>();
const mockUseWorkspacesEnabled = jest.fn<() => { workspacesEnabled: boolean }>();
const mockAddConditionMutateAsync = jest.fn<(...args: any[]) => any>();
const mockUseRoleMutationConditionsQuery = jest.fn<() => any>();
const mockAddPermissionMutateAsync = jest.fn<(...args: any[]) => any>();

jest.mock('../hooks', () => ({
  useUpdateRole: () => ({ mutateAsync: jest.fn() }),
  useAddPermission: () => ({ mutateAsync: mockAddPermissionMutateAsync }),
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
  useRoleMutationConditionsQuery: () => mockUseRoleMutationConditionsQuery(),
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
  mockUseRoleMutationConditionsQuery.mockReset();
  mockUseRoleMutationConditionsQuery.mockReturnValue({
    data: { mutation_conditions: [] },
    isLoading: false,
    error: null,
  });
  mockAddPermissionMutateAsync.mockReset();
  mockAddPermissionMutateAsync.mockResolvedValue({});
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

describe('EditRoleModal — restrictions land before capability', () => {
  // A role's conditions apply to everyone holding it, and adding a permission widens that
  // role for all of them. The submit chain used to add permissions and assign users first
  // and conditions last, so a condition that failed left the widened role in force.

  const stageConditionAndPermission = () => {
    fireEvent.change(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), {
      target: { value: "tags.env = 'dev'" },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));
    fireEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));
  };

  it('adds the condition before the permission it narrows', async () => {
    const order: string[] = [];
    mockAddConditionMutateAsync.mockImplementation(async () => {
      order.push('condition');
      return {};
    });
    mockAddPermissionMutateAsync.mockImplementation(async () => {
      order.push('permission');
      return {};
    });
    const onClose = jest.fn();
    renderWithDesignSystem(<EditRoleModal open onClose={onClose} roleId={1} />);

    expect(await screen.findByText('Add a mutation condition')).toBeInTheDocument();
    stageConditionAndPermission();

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(order).toEqual(['condition', 'permission']);
  });

  it('confirms before discarding a half-filled condition draft', async () => {
    // F-0036. `hasUnsavedConditionDraft` was reported by MutationConditionsSection and
    // then never read -- only `hasUnsavedDraft` gated the transition -- so a condition
    // typed but not Added was dropped silently and the submit went on to grant the
    // permission it was meant to narrow. The dialog's own copy already names both drafts.
    renderWithDesignSystem(<EditRoleModal open onClose={jest.fn()} roleId={1} />);
    expect(await screen.findByText('Add a mutation condition')).toBeInTheDocument();

    // Type a condition but deliberately do NOT click "Add mutation condition".
    fireEvent.change(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), {
      target: { value: "tags.env = 'dev'" },
    });
    // Stage a permission, so Review is enabled by a real change rather than the draft.
    fireEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));

    // The discard dialog must intervene instead of the review step appearing.
    expect(await screen.findByText('Discard unsaved entry?')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Apply changes$/ })).not.toBeInTheDocument();
  });

  it('does not add the permission when the condition could not be created', async () => {
    mockAddConditionMutateAsync.mockRejectedValue(new Error('slot exhausted'));
    renderWithDesignSystem(<EditRoleModal open onClose={jest.fn()} roleId={1} />);

    expect(await screen.findByText('Add a mutation condition')).toBeInTheDocument();
    stageConditionAndPermission();

    await waitFor(() => expect(mockAddConditionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockAddPermissionMutateAsync).not.toHaveBeenCalled();
  });

  it('does not re-add a condition that already landed when a later step is retried', async () => {
    // The staged row used to keep no id, so the add list -- which selects on a missing
    // id -- still contained a condition the server had already created. A later step
    // failing leaves the modal open and re-submittable, and the retry replayed the add.
    // Every add allocates a fresh slot rather than deduplicating, so the replay left a
    // duplicate restriction behind and spent the per-type limit; enough retries and
    // later grants are refused for want of a slot.
    mockAddConditionMutateAsync.mockResolvedValue({ mutation_conditions: { id: 7 } });
    mockAddPermissionMutateAsync.mockRejectedValue(new Error('transient'));
    renderWithDesignSystem(<EditRoleModal open onClose={jest.fn()} roleId={1} />);

    expect(await screen.findByText('Add a mutation condition')).toBeInTheDocument();
    stageConditionAndPermission();

    await waitFor(() => expect(mockAddPermissionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockAddConditionMutateAsync).toHaveBeenCalledTimes(1);

    // Retry. A failed submit returns to the edit step, so the retry goes back through
    // review. The permission is attempted again -- it never landed -- but the condition
    // must not be, because it did.
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(await screen.findByRole('button', { name: /^Apply changes$/ }));
    await waitFor(() => expect(mockAddPermissionMutateAsync).toHaveBeenCalledTimes(2));
    expect(mockAddConditionMutateAsync).toHaveBeenCalledTimes(1);
  });
});

describe('EditRoleModal — an unknown condition policy blocks the form', () => {
  it('blocks editing and submitting when the conditions fetch failed', async () => {
    // An errored fetch is an UNKNOWN list, not an empty one. Pre-filling `[]` shows a role
    // whose restrictions failed to load as a role carrying none, and the admin then reviews
    // and applies name, permission and assignment changes against that false picture.
    mockUseRoleMutationConditionsQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('conditions endpoint unavailable'),
    });
    renderWithDesignSystem(<EditRoleModal open onClose={jest.fn()} roleId={1} />);

    expect(await screen.findByText('Failed to load mutation conditions')).toBeInTheDocument();
    expect(screen.getByText('conditions endpoint unavailable')).toBeInTheDocument();
    // The form itself is gone, so there is nothing to edit against the wrong picture.
    expect(screen.queryByText('Add a mutation condition')).not.toBeInTheDocument();
    expect(screen.queryByText('Add a permission')).not.toBeInTheDocument();
  });

  it('renders the form normally when the fetch succeeded', async () => {
    renderWithDesignSystem(<EditRoleModal open onClose={jest.fn()} roleId={1} />);

    expect(await screen.findByText('Add a mutation condition')).toBeInTheDocument();
    expect(screen.queryByText('Failed to load mutation conditions')).not.toBeInTheDocument();
  });
});
