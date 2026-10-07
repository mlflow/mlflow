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
const mockAddConditionMutateAsync = jest.fn<(...args: any[]) => any>();
const mockRemoveConditionMutateAsync = jest.fn<(...args: any[]) => any>();
const mockUseRoleMutationConditionsQuery = jest.fn<() => any>();

jest.mock('../hooks', () => ({
  // Conditions now share these modals; stub them so the cases below keep testing
  // what they were written for.
  useRoleMutationConditionsQuery: () => mockUseRoleMutationConditionsQuery(),
  useAddUserMutationCondition: () => ({ mutateAsync: mockAddConditionMutateAsync, isLoading: false }),
  useRemoveMutationCondition: () => ({ mutateAsync: mockRemoveConditionMutateAsync, isLoading: false }),
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
  mockUseRoleMutationConditionsQuery.mockReturnValue({
    data: { mutation_conditions: [] },
    isLoading: false,
    error: null,
  });
  mockAddConditionMutateAsync.mockReset();
  mockAddConditionMutateAsync.mockResolvedValue({});
  mockRemoveConditionMutateAsync.mockReset();
  mockRemoveConditionMutateAsync.mockResolvedValue({});
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

describe('EditAccessModal — condition scope on the wire', () => {
  beforeEach(() => {
    mockUseUserRolesQuery.mockReset();
    mockUseUserRolesQuery.mockReturnValue({ data: { roles: [] }, isLoading: false, error: null });
  });

  // An existing condition, as the server returns it: narrowed to ONE experiment.
  const scopedCondition = {
    id: 11,
    role_id: 99,
    condition_slot: 1,
    resource_type: 'experiment',
    resource_pattern: '7',
    container_resource_type: 'workspace',
    container_resource_pattern: '*',
    value_condition: null,
    target_condition: "tags.lifecycle != 'prod'",
  };
  // Same type, same clauses, different scope: the whole workspace. These two differ ONLY
  // in `resource_pattern`, which is exactly what the modal's own key used to drop.
  const workspaceCondition = { ...scopedCondition, id: 12, resource_pattern: '*' };

  it('sends the staged scope rather than letting the server default it', async () => {
    // `resource_pattern` was omitted entirely, and the server normalises an absent scope
    // to the wildcard -- so a condition the admin scoped to one experiment was persisted
    // as one covering every experiment in the workspace. `objectContaining` fails on an
    // ABSENT key, which is what makes this catch the omission even at the default scope.
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);

    fireEvent.change(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), {
      target: { value: "tags.env = 'dev'" },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    await waitFor(() => expect(mockAddConditionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockAddConditionMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        request: expect.objectContaining({
          username: 'alice',
          resource_type: 'experiment',
          target_condition: "tags.env = 'dev'",
          resource_pattern: '*',
          container_resource_type: 'workspace',
          container_resource_pattern: '*',
        }),
      }),
    );
  });

  it('removes the one condition the admin removed, not whichever shared its other fields', async () => {
    // Two conditions differing only in scope. The modal's key omitted the scope, so both
    // produced the same key: removing the scoped one left its key in the desired set (the
    // workspace-wide one still carried it), the removal was dropped from the diff, and the
    // restriction the admin had just lifted stayed in force.
    mockUseRoleMutationConditionsQuery.mockReturnValue({
      data: { mutation_conditions: [scopedCondition, workspaceCondition] },
      isLoading: false,
      error: null,
    });
    const onClose = jest.fn();
    renderWithDesignSystem(<EditAccessModal open onClose={onClose} username="alice" />);

    // Both rows are listed; remove the FIRST (the one scoped to experiment 7). Both are
    // `experiment` conditions, so they share one aria-label and are told apart by order.
    const removeButtons = await waitFor(() => {
      const found = screen.getAllByRole('button', { name: /Remove Experiment mutation condition/ });
      expect(found).toHaveLength(2);
      return found;
    });
    fireEvent.click(removeButtons[0]);
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(mockRemoveConditionMutateAsync).toHaveBeenCalledTimes(1);
    expect(mockRemoveConditionMutateAsync).toHaveBeenCalledWith(scopedCondition.id);
  });
});

describe('EditAccessModal — prefill waits for the conditions query', () => {
  // The conditions query is keyed on the synthetic role id, which is derived from the
  // ROLES response -- so it cannot even start until the roles query has resolved, which
  // is exactly when `stateLoaded` flips true. The prefill then ran against an empty
  // condition list and latched `filledForWorkspaceRef`, so the real conditions arriving a
  // moment later never reached the editable state. The diff compared a populated
  // `currentConditions` against an empty staged list and concluded the admin had removed
  // all of them: changing anything unrelated silently deleted every restriction on the
  // user. Fail-OPEN, and deterministic rather than a rare interleaving.

  const existingCondition = {
    id: 21,
    role_id: 99,
    condition_slot: 1,
    resource_type: 'experiment',
    resource_pattern: '*',
    container_resource_type: 'workspace',
    container_resource_pattern: '*',
    value_condition: null,
    target_condition: "tags.lifecycle != 'prod'",
  };

  beforeEach(() => {
    mockUseUserRolesQuery.mockReset();
    mockUseUserRolesQuery.mockReturnValue({
      data: { roles: [syntheticUserRole('default')] },
      isLoading: false,
      error: null,
    });
    mockGrantPermissionMutateAsync.mockReset();
    mockGrantPermissionMutateAsync.mockResolvedValue({});
    mockRevokePermissionMutateAsync.mockReset();
    mockRevokePermissionMutateAsync.mockResolvedValue({});
  });

  it('does not stage a removal for conditions that arrive after the roles query', async () => {
    // Roles have resolved but the conditions request is still in flight: the state the
    // modal is always in on first open.
    mockUseRoleMutationConditionsQuery.mockReturnValue({ data: undefined, isLoading: true, error: null });
    const onClose = jest.fn();
    const { rerender } = renderWithDesignSystem(<EditAccessModal open onClose={onClose} username="alice" />);

    // The conditions land, and the settling query re-renders the modal.
    mockUseRoleMutationConditionsQuery.mockReturnValue({
      data: { mutation_conditions: [existingCondition] },
      isLoading: false,
      error: null,
    });
    rerender(<EditAccessModal open onClose={onClose} username="alice" />);

    // An unrelated edit: granting a direct permission.
    fireEvent.click(await screen.findByRole('radio', { name: /^All experiments$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Add$/ }));

    // The prefilled condition is now visible, which is the positive half of the fix.
    expect(await screen.findByRole('button', { name: /Remove Experiment mutation condition/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    // The admin touched a permission, not a condition. Nothing may be removed.
    expect(mockRemoveConditionMutateAsync).not.toHaveBeenCalled();
    expect(mockGrantPermissionMutateAsync).toHaveBeenCalledTimes(1);
  });

  it('blocks the form when the conditions query fails rather than pre-filling an empty list', async () => {
    // An errored query is an UNKNOWN condition list, not an empty one. Treating it as
    // empty would stage the same mass removal, so the form is blocked exactly as it is
    // for a failed roles fetch.
    mockUseRoleMutationConditionsQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('conditions boom'),
    });
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);

    expect(await screen.findByText('Failed to load access state')).toBeInTheDocument();
    expect(screen.getByText('conditions boom')).toBeInTheDocument();
    expect(screen.queryByText('Role assignments')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Review changes/ })).toBeDisabled();
  });

  it('still pre-fills when the user has no synthetic role, where the query never runs', async () => {
    // With no direct grants there is no synthetic role, so the query is disabled -- and a
    // disabled react-query v4 query reports `isLoading: true` forever. Gating on that flag
    // alone would hang the modal on a permanent spinner.
    mockUseUserRolesQuery.mockReturnValue({ data: { roles: [] }, isLoading: false, error: null });
    mockUseRoleMutationConditionsQuery.mockReturnValue({ data: undefined, isLoading: true, error: null });
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);

    // The form is usable, not stuck behind a spinner.
    expect(await screen.findByText('Role assignments')).toBeInTheDocument();
  });
});

describe('EditAccessModal — restrictions land before capability', () => {
  // Grants add access and conditions subtract it, so capability must never exist without
  // the restriction the admin paired with it. The submit chain used to apply admin status,
  // roles and grants first and conditions last, so a condition that failed left the grant
  // it was meant to narrow live and unrestricted -- and the modal reported only the
  // condition failure, which reads as "the restriction didn't apply", not as "the user now
  // has unrestricted access".

  const existingCondition = {
    id: 31,
    role_id: 99,
    condition_slot: 1,
    resource_type: 'experiment',
    resource_pattern: '*',
    container_resource_type: 'workspace',
    container_resource_pattern: '*',
    value_condition: null,
    target_condition: "tags.lifecycle != 'prod'",
  };

  beforeEach(() => {
    mockUseUserRolesQuery.mockReset();
    mockGrantPermissionMutateAsync.mockReset();
    mockGrantPermissionMutateAsync.mockResolvedValue({});
    mockRevokePermissionMutateAsync.mockReset();
    mockRevokePermissionMutateAsync.mockResolvedValue({});
  });

  const stageConditionAndGrant = () => {
    fireEvent.change(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), {
      target: { value: "tags.env = 'dev'" },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));
    fireEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));
  };

  it('adds the condition before granting the permission it narrows', async () => {
    mockUseUserRolesQuery.mockReturnValue({ data: { roles: [] }, isLoading: false, error: null });
    const order: string[] = [];
    mockAddConditionMutateAsync.mockImplementation(async () => {
      order.push('condition');
      return {};
    });
    mockGrantPermissionMutateAsync.mockImplementation(async () => {
      order.push('grant');
      return {};
    });
    const onClose = jest.fn();
    renderWithDesignSystem(<EditAccessModal open onClose={onClose} username="alice" />);

    stageConditionAndGrant();

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(order).toEqual(['condition', 'grant']);
  });

  it('does not grant when the condition could not be created', async () => {
    // The whole point: a restriction that failed must not be followed by the capability it
    // was supposed to narrow.
    mockUseUserRolesQuery.mockReturnValue({ data: { roles: [] }, isLoading: false, error: null });
    mockAddConditionMutateAsync.mockRejectedValue(new Error('slot exhausted'));
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);

    stageConditionAndGrant();

    await waitFor(() => expect(mockAddConditionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockGrantPermissionMutateAsync).not.toHaveBeenCalled();
  });

  it('keeps the condition when the revoke it was covering fails', async () => {
    // The mirror rule. The admin revokes a grant and drops the now-pointless condition in
    // one save; if the revoke fails and the condition removal still runs, the grant is
    // left live and unrestricted.
    mockUseUserRolesQuery.mockReturnValue({
      data: { roles: [syntheticUserRole('default')] },
      isLoading: false,
      error: null,
    });
    mockUseRoleMutationConditionsQuery.mockReturnValue({
      data: { mutation_conditions: [existingCondition] },
      isLoading: false,
      error: null,
    });
    mockRevokePermissionMutateAsync.mockRejectedValue(new Error('revoke boom'));
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);

    // Remove the prefilled grant and the prefilled condition together.
    fireEvent.click(await screen.findByRole('button', { name: 'Remove experiment *' }));
    fireEvent.click(await screen.findByRole('button', { name: /Remove Experiment mutation condition/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    await waitFor(() => expect(mockRevokePermissionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockRemoveConditionMutateAsync).not.toHaveBeenCalled();
  });

  it('keeps the old condition when its replacement could not be created', async () => {
    // F-0039. Editing a condition is an ADD plus a REMOVE. The removal gate only tested
    // whether a CAPABILITY removal had failed, so a failed add still dropped the old
    // restriction -- leaving the grant less restricted than before the admin touched it.
    mockUseUserRolesQuery.mockReturnValue({
      data: { roles: [syntheticUserRole('default')] },
      isLoading: false,
      error: null,
    });
    mockUseRoleMutationConditionsQuery.mockReturnValue({
      data: { mutation_conditions: [existingCondition] },
      isLoading: false,
      error: null,
    });
    mockAddConditionMutateAsync.mockRejectedValue(new Error('slot exhausted'));
    renderWithDesignSystem(<EditAccessModal open onClose={jest.fn()} username="alice" />);

    // Drop the prefilled condition and stage a new one in its place -- a replacement.
    fireEvent.click(await screen.findByRole('button', { name: /Remove Experiment mutation condition/ }));
    stageConditionAndGrant();

    await waitFor(() => expect(mockAddConditionMutateAsync).toHaveBeenCalledTimes(1));
    // The replacement failed, so the original must survive.
    expect(mockRemoveConditionMutateAsync).not.toHaveBeenCalled();
  });

  it('removes the condition once the revoke actually succeeds', async () => {
    mockUseUserRolesQuery.mockReturnValue({
      data: { roles: [syntheticUserRole('default')] },
      isLoading: false,
      error: null,
    });
    mockUseRoleMutationConditionsQuery.mockReturnValue({
      data: { mutation_conditions: [existingCondition] },
      isLoading: false,
      error: null,
    });
    const onClose = jest.fn();
    renderWithDesignSystem(<EditAccessModal open onClose={onClose} username="alice" />);

    fireEvent.click(await screen.findByRole('button', { name: 'Remove experiment *' }));
    fireEvent.click(await screen.findByRole('button', { name: /Remove Experiment mutation condition/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Review changes$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Apply changes$/ }));

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(mockRemoveConditionMutateAsync).toHaveBeenCalledWith(existingCondition.id);
  });
});
