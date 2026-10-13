import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import UserDetailPage from './UserDetailPage';

// Per-test override for ``useUserRolesQuery`` and the active tab, so each case
// can pick its own role payload and land on the Roles or Permissions tab.
const mockUseUserRolesQuery = jest.fn();
let mockSearchParams = new URLSearchParams();

jest.mock('../hooks', () => ({
  useCurrentUserIsAdmin: () => true,
  useUserRolesQuery: (username: string) => mockUseUserRolesQuery(username),
  useUsersQuery: () => ({
    data: { users: [{ id: 1, username: 'alice', is_admin: false, roles: [] }] },
    isLoading: false,
    error: null,
  }),
  useWithSettingsReturnTo: () => (route: string) => route,
}));

jest.mock('../../workspaces/utils/WorkspaceUtils', () => ({
  useActiveWorkspace: () => null,
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => ({ workspacesEnabled: false }),
}));

// The modal drags in its own query stack; it's irrelevant to the roles/
// permissions split under test, so stub it out.
jest.mock('../components/EditAccessModal', () => ({
  EditAccessModal: () => null,
}));

jest.mock('../../common/utils/RoutingUtils', () => ({
  // Keep ``createMLflowRoutePath`` et al. real so ``admin/routes`` still
  // builds; only stub the router hooks and Link (no Router provider here).
  ...jest.requireActual<typeof import('../../common/utils/RoutingUtils')>('../../common/utils/RoutingUtils'),
  useParams: () => ({ username: 'alice' }),
  useSearchParams: () => [mockSearchParams, jest.fn()],
  Link: ({ children }: { children: React.ReactNode }) => <a>{children}</a>,
}));

const syntheticOnlyRoles = {
  roles: [
    {
      id: 19,
      name: '__user_1__',
      workspace: 'default',
      description: null,
      permissions: [{ id: 1, role_id: 19, resource_type: 'experiment', resource_pattern: '19', permission: 'MANAGE' }],
    },
  ],
};

describe('UserDetailPage — synthetic role wiring (#24170)', () => {
  beforeEach(() => {
    mockUseUserRolesQuery.mockReset();
    mockSearchParams = new URLSearchParams();
  });

  it("surfaces a synthetic-only user's direct grant on the Permissions tab", () => {
    // Regression guard for #24170: the page must hand the *unfiltered* role
    // list (including the synthetic ``__user_<id>__`` role that backs direct
    // grants) to PermissionsSection, not the synthetic-filtered ``displayRoles``.
    mockSearchParams = new URLSearchParams('tab=permissions');
    mockUseUserRolesQuery.mockReturnValue({ data: syntheticOnlyRoles, isLoading: false, error: null });

    renderWithDesignSystem(<UserDetailPage />);

    expect(screen.getByText(/experiment:19/)).toBeInTheDocument();
    expect(screen.getByText('MANAGE')).toBeInTheDocument();
    // Rendered under the localized "Direct" source, never the raw role name.
    expect(screen.getByText('Direct')).toBeInTheDocument();
    expect(screen.queryByText('__user_1__')).not.toBeInTheDocument();
    expect(screen.queryByText('No resource permissions to show.')).not.toBeInTheDocument();
  });

  it('hides the synthetic role from the human-facing Roles tab', () => {
    // The flip side of the split: the synthetic role must stay out of the
    // Roles table, so a synthetic-only user reads as having no named roles.
    mockUseUserRolesQuery.mockReturnValue({ data: syntheticOnlyRoles, isLoading: false, error: null });

    renderWithDesignSystem(<UserDetailPage />);

    expect(screen.queryByText('__user_1__')).not.toBeInTheDocument();
    expect(screen.getByText('This user has not been assigned to any roles.')).toBeInTheDocument();
  });
});
