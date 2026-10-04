import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { fireEvent, renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { CreateUserModal } from './CreateUserModal';

let userEvent: ReturnType<typeof userEventGlobal.setup>;
beforeEach(() => {
  userEvent = userEventGlobal.setup();
  mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: false });
  // ``null`` is what ``useActiveWorkspace`` actually returns on a
  // single-tenant server — the fix must not coerce it back to a workspace.
  mockUseActiveWorkspace.mockReturnValue(null);
});

// ``fireEvent.change``-based credential fill: per-keystroke ``userEvent.type``
// pushed the grant-flow tests past jest's default 5s per-test timeout on
// loaded CI runners, and ``jest.setTimeout`` is banned by lint.
const fillCredentials = () => {
  fireEvent.change(screen.getByPlaceholderText('Enter username'), { target: { value: 'newbie' } });
  fireEvent.change(screen.getByPlaceholderText('Enter password'), { target: { value: 'hunter2' } });
};

// Typed as ``(...args: any[]) => any`` so ``mockResolvedValue`` accepts the
// realistic response shapes the component awaits. ``jest.fn()``'s default
// signature is ``() => never``, which would reject the payload.
const mockCreateUserMutateAsync = jest.fn<(...args: any[]) => any>();
const mockGrantPermissionMutateAsync = jest.fn<(...args: any[]) => any>();
const mockAddConditionMutateAsync = jest.fn<(...args: any[]) => any>();
const mockUseWorkspacesEnabled = jest.fn<() => { workspacesEnabled: boolean }>();
const mockUseActiveWorkspace = jest.fn<() => string | null>();

jest.mock('../hooks', () => ({
  AdminQueryKeys: {
    users: ['admin_users'],
    roles: ['admin_roles'],
    roleUsers: (roleId: number) => ['admin_role_users', roleId],
    resourceOptions: (resourceType: string) => ['admin_resource_options', resourceType],
  },
  useCreateUser: () => ({ mutateAsync: mockCreateUserMutateAsync }),
  useCurrentUserIsAdmin: () => true,
  useGrantUserPermission: () => ({ mutateAsync: mockGrantPermissionMutateAsync }),
  useAddUserMutationCondition: () => ({ mutateAsync: mockAddConditionMutateAsync }),
  // ``useResourceOptionsQuery`` is reached by ``DirectPermissionForm`` even
  // when its parent section is collapsed (``hidden`` keeps it mounted), so
  // the stub has to exist; only the shape matters.
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
  useRolesQuery: () => ({ data: { roles: [] }, isLoading: false, error: null }),
  useWorkspaceOptions: () => ['default', 'team-a'],
}));

jest.mock('../../workspaces/utils/WorkspaceUtils', () => ({
  useActiveWorkspace: () => mockUseActiveWorkspace(),
}));

jest.mock('../../workspaces/hooks/useWorkspaces', () => ({
  useWorkspaces: () => ({ workspaces: [{ name: 'default' }, { name: 'team-a' }], isLoading: false }),
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => mockUseWorkspacesEnabled(),
}));

jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));

// The submit path also reaches AdminApi directly; stub the methods we touch
// so a click submit doesn't end up making real network calls.
jest.mock('../api', () => ({
  AdminApi: {
    assignRole: jest.fn(),
    updateAdmin: jest.fn(),
  },
}));

describe('CreateUserModal — collapsible optional sections', () => {
  beforeEach(() => {
    mockCreateUserMutateAsync.mockReset();
    mockGrantPermissionMutateAsync.mockReset();
  });

  it('renders Role assignment + Direct permissions collapsed and Admin status expanded', () => {
    // Locks the rationale documented next to ``Admin status`` in the modal:
    // multi-field optional sections collapse for density; a single-Switch
    // section stays open because hiding a one-liner behind a click is just
    // extra friction.
    renderWithDesignSystem(<CreateUserModal open onClose={jest.fn()} />);
    expect(screen.getByRole('button', { name: /Role assignment/ })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByRole('button', { name: /Direct permissions/ })).toHaveAttribute('aria-expanded', 'false');
    // ``Admin status`` is rendered (the admin mock returns true) but is not
    // collapsible, so it must not surface a toggle button.
    expect(screen.getByText('Admin status')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Admin status/ })).not.toBeInTheDocument();
  });

  it('submits with the optional sections still collapsed and closes the modal on success', async () => {
    // The whole point of collapsing the optional sections is that the admin
    // shouldn't have to interact with them to create a user. Pin that the
    // submit path works against the default-collapsed state and that the
    // modal closes itself on success (the only signal that the happy path
    // ran end-to-end, since there's no toast / redirect to observe).
    mockCreateUserMutateAsync.mockResolvedValue({ user: { username: 'newbie' } });
    const onClose = jest.fn();
    renderWithDesignSystem(<CreateUserModal open onClose={onClose} />);

    fillCredentials();
    // Sanity-check we're still in the default-collapsed state before submit
    // — otherwise the test isn't really exercising what we claim.
    expect(screen.getByRole('button', { name: /Role assignment/ })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByRole('button', { name: /Direct permissions/ })).toHaveAttribute('aria-expanded', 'false');

    await userEvent.click(screen.getByRole('button', { name: /^Create user$/ }));

    expect(mockCreateUserMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({ username: 'newbie', password: 'hunter2' }),
    );
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe('CreateUserModal — discard-confirm gate on unsaved direct-grant draft', () => {
  beforeEach(() => {
    mockCreateUserMutateAsync.mockReset();
    mockGrantPermissionMutateAsync.mockReset();
  });

  it('intercepts submit with a discard-confirm dialog when there is an unsaved draft, and proceeds on confirm', async () => {
    // Admin expands Direct permissions, flips scope to "All experiments",
    // forgets to click ``Add``, fills creds, clicks Create user. The modal
    // does NOT silently create the user — it pops a confirm dialog so the
    // admin can either go back and click ``Add``, or knowingly discard.
    mockCreateUserMutateAsync.mockResolvedValue({ user: { username: 'newbie' } });
    renderWithDesignSystem(<CreateUserModal open onClose={jest.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: /Direct permissions/ }));
    await userEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));

    fillCredentials();

    // Submit button stays enabled — the gate is the dialog, not a lock.
    const submit = screen.getByRole('button', { name: /^Create user and grant access$|^Create user$/ });
    expect(submit).not.toBeDisabled();
    await userEvent.click(submit);

    // Confirm dialog appears; ``createUser`` hasn't been called yet.
    expect(await screen.findByText('Discard unsaved entry?')).toBeInTheDocument();
    expect(mockCreateUserMutateAsync).not.toHaveBeenCalled();

    // Confirm "Continue" → submit proceeds.
    await userEvent.click(screen.getByRole('button', { name: /^Continue$/ }));
    expect(mockCreateUserMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({ username: 'newbie', password: 'hunter2' }),
    );
    // Draft was discarded — never staged, never sent.
    expect(mockGrantPermissionMutateAsync).not.toHaveBeenCalled();
  });

  it('cancelling the discard-confirm keeps the modal on the edit step without creating the user', async () => {
    // Same setup — but the admin clicks ``Back`` on the dialog because
    // they meant to ``Add`` the permission. Dialog closes, no API calls
    // fire, the draft is preserved so the admin can click ``Add`` now.
    renderWithDesignSystem(<CreateUserModal open onClose={jest.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: /Direct permissions/ }));
    await userEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));

    fillCredentials();

    await userEvent.click(screen.getByRole('button', { name: /^Create user and grant access$|^Create user$/ }));
    expect(await screen.findByText('Discard unsaved entry?')).toBeInTheDocument();

    // ``Back`` is unique to the discard-confirm dialog — the outer modal's
    // secondary button is still labelled ``Cancel``, so the role+name query
    // resolves unambiguously.
    await userEvent.click(screen.getByRole('button', { name: /^Back$/ }));
    expect(mockCreateUserMutateAsync).not.toHaveBeenCalled();
    expect(mockGrantPermissionMutateAsync).not.toHaveBeenCalled();
  });
});

describe('CreateUserModal — workspace targeting on direct grants', () => {
  beforeEach(() => {
    mockCreateUserMutateAsync.mockReset();
    mockGrantPermissionMutateAsync.mockReset();
  });

  // ``fireEvent`` instead of ``userEvent``: this is the heaviest flow in the
  // file (mounts the collapsed direct-permissions form), and the pointer
  // pipeline made it exceed the default 5s timeout on loaded CI runners.
  const stageGrantAndSubmit = () => {
    fireEvent.click(screen.getByRole('button', { name: /Direct permissions/ }));
    fireEvent.click(screen.getByRole('radio', { name: /^All experiments$/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Add$/ }));
    fillCredentials();
    fireEvent.click(screen.getByRole('button', { name: /^Create user and grant access$/ }));
  };

  it('omits the workspace when workspaces are disabled', async () => {
    // Regression: single-tenant servers reject ANY ``X-MLFLOW-WORKSPACE``
    // header (even ``default``) with FEATURE_DISABLED, so the modal must not
    // coerce "no active workspace" into ``default`` on the grant request.
    mockCreateUserMutateAsync.mockResolvedValue({ user: { username: 'newbie' } });
    mockGrantPermissionMutateAsync.mockResolvedValue({});
    const onClose = jest.fn();
    renderWithDesignSystem(<CreateUserModal open onClose={onClose} />);

    stageGrantAndSubmit();

    // The submit handler is async (create → grant → close); ``onClose`` is
    // its last step, so waiting on it settles the whole chain.
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(mockGrantPermissionMutateAsync).toHaveBeenCalledTimes(1);
    expect(mockGrantPermissionMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        resource_type: 'experiment',
        resource_id: '*',
        username: 'newbie',
        permission: 'READ',
      }),
    );
    expect(mockGrantPermissionMutateAsync.mock.calls[0][0].workspace).toBeUndefined();
  });

  it('targets the selected workspace when workspaces are enabled', async () => {
    mockUseWorkspacesEnabled.mockReturnValue({ workspacesEnabled: true });
    // A non-default active workspace: the pre-fix coercion also produced
    // ``'default'``, so asserting ``'default'`` here would pass on the
    // broken code too.
    mockUseActiveWorkspace.mockReturnValue('team-a');
    mockCreateUserMutateAsync.mockResolvedValue({ user: { username: 'newbie' } });
    mockGrantPermissionMutateAsync.mockResolvedValue({});
    renderWithDesignSystem(<CreateUserModal open onClose={jest.fn()} />);

    stageGrantAndSubmit();

    await waitFor(() => expect(mockGrantPermissionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockGrantPermissionMutateAsync.mock.calls[0][0].workspace).toBe('team-a');
  });
});

describe('CreateUserModal — direct mutation conditions', () => {
  beforeEach(() => {
    mockCreateUserMutateAsync.mockReset();
    mockGrantPermissionMutateAsync.mockReset();
    mockAddConditionMutateAsync.mockReset();
    mockAddConditionMutateAsync.mockResolvedValue({});
  });

  it('applies a staged condition after the user is created, addressed by username', async () => {
    // The user does not exist when the condition is staged, so there is no role to
    // address. The user-addressed add resolves and creates the per-user role
    // server-side, which is the whole reason a condition can be staged here at all.
    mockCreateUserMutateAsync.mockResolvedValue({ user: { username: 'newbie' } });
    renderWithDesignSystem(<CreateUserModal open onClose={jest.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: /Direct mutation conditions/ }));
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.env = 'dev'");
    await userEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));

    fillCredentials();
    await userEvent.click(screen.getByRole('button', { name: /^Create user and grant access$/ }));

    await waitFor(() => expect(mockAddConditionMutateAsync).toHaveBeenCalledTimes(1));
    expect(mockAddConditionMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        request: expect.objectContaining({
          username: 'newbie',
          resource_type: 'experiment',
          target_condition: "tags.env = 'dev'",
          // Defaults travel explicitly rather than being omitted: the wire shape has no
          // nullable scope, so "everything" is the wildcard, not an absent field.
          resource_pattern: '*',
          container_resource_type: 'workspace',
          container_resource_pattern: '*',
        }),
      }),
    );
  });

  it('creates the user before applying conditions, not after', async () => {
    // A condition only subtracts from granted access, so it cannot be stored against a
    // user who does not exist yet. Ordering is the guarantee, so it is asserted.
    const order: string[] = [];
    mockCreateUserMutateAsync.mockImplementation(async () => {
      order.push('createUser');
      return { user: { username: 'newbie' } };
    });
    mockAddConditionMutateAsync.mockImplementation(async () => {
      order.push('addCondition');
      return {};
    });
    renderWithDesignSystem(<CreateUserModal open onClose={jest.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: /Direct mutation conditions/ }));
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.env = 'dev'");
    await userEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));

    fillCredentials();
    await userEvent.click(screen.getByRole('button', { name: /^Create user and grant access$/ }));

    await waitFor(() => expect(order).toEqual(['createUser', 'addCondition']));
  });

  it('reports a failed condition without rolling back the user', async () => {
    // Same best-effort contract the grants follow: the user survives and the admin is
    // told what to retry, because a created user is not something to silently undo.
    mockCreateUserMutateAsync.mockResolvedValue({ user: { username: 'newbie' } });
    mockAddConditionMutateAsync.mockRejectedValue(new Error('filter syntax error'));
    const onClose = jest.fn();
    renderWithDesignSystem(<CreateUserModal open onClose={onClose} />);

    await userEvent.click(screen.getByRole('button', { name: /Direct mutation conditions/ }));
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.env = 'dev'");
    await userEvent.click(screen.getByRole('button', { name: 'Add mutation condition' }));

    fillCredentials();
    await userEvent.click(screen.getByRole('button', { name: /^Create user and grant access$/ }));

    expect(await screen.findByText(/Mutation condition on experiment failed/)).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('gates submit behind the discard confirm for an unsaved condition draft', async () => {
    // The gate previously watched only the permissions draft, so a touched-but-unadded
    // condition would have been dropped silently.
    mockCreateUserMutateAsync.mockResolvedValue({ user: { username: 'newbie' } });
    renderWithDesignSystem(<CreateUserModal open onClose={jest.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: /Direct mutation conditions/ }));
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.env = 'dev'");

    fillCredentials();
    await userEvent.click(screen.getByRole('button', { name: /^Create user$|^Create user and grant access$/ }));

    expect(await screen.findByText('Discard unsaved entry?')).toBeInTheDocument();
    expect(mockCreateUserMutateAsync).not.toHaveBeenCalled();
  });
});
