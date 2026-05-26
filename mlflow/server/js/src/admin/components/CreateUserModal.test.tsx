import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { CreateUserModal } from './CreateUserModal';

let userEvent: ReturnType<typeof userEventGlobal.setup>;
beforeEach(() => {
  userEvent = userEventGlobal.setup();
});

// Typed as ``(...args: any[]) => any`` so ``mockResolvedValue`` accepts the
// realistic response shapes the component awaits. ``jest.fn()``'s default
// signature is ``() => never``, which would reject the payload.
const mockCreateUserMutateAsync = jest.fn<(...args: any[]) => any>();

jest.mock('../hooks', () => ({
  AdminQueryKeys: {
    users: ['admin_users'],
    roles: ['admin_roles'],
    roleUsers: (roleId: number) => ['admin_role_users', roleId],
    resourceOptions: (resourceType: string) => ['admin_resource_options', resourceType],
  },
  useCreateUser: () => ({ mutateAsync: mockCreateUserMutateAsync }),
  useCurrentUserIsAdmin: () => true,
  // ``useGrantUserPermission`` is invoked by the modal even though our tests
  // never stage a direct permission (and so the ``wantsDirect`` branch in
  // ``handleSubmit`` is skipped). Inline the stub so we don't carry an
  // unused captured spy.
  useGrantUserPermission: () => ({ mutateAsync: jest.fn() }),
  // ``useResourceOptionsQuery`` is reached by ``DirectPermissionForm`` even
  // when its parent section is collapsed (``hidden`` keeps it mounted), so
  // the stub has to exist; only the shape matters.
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
  useRolesQuery: () => ({ data: { roles: [] }, isLoading: false, error: null }),
  useWorkspaceOptions: () => ['default'],
}));

jest.mock('../../workspaces/utils/WorkspaceUtils', () => ({
  useActiveWorkspace: () => 'default',
}));

jest.mock('../../workspaces/hooks/useWorkspaces', () => ({
  useWorkspaces: () => ({ workspaces: [{ name: 'default' }], isLoading: false }),
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => ({ workspacesEnabled: false }),
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

    await userEvent.type(screen.getByPlaceholderText('Enter username'), 'newbie');
    await userEvent.type(screen.getByPlaceholderText('Enter password'), 'hunter2');
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
