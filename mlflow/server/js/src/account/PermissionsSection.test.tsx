import { describe, it, expect } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { PermissionsSection } from './PermissionsSection';
import type { Role } from './types';

const role = (overrides: Partial<Role>): Role => ({
  id: 1,
  name: 'reader',
  workspace: 'default',
  description: null,
  permissions: [],
  ...overrides,
});

// Convenience factory for a synthetic per-user role (renders as "Direct"
// in the Source column).
const syntheticRole = (overrides: Partial<Role>): Role =>
  role({
    id: 99,
    name: '__user_1__',
    permissions: [{ id: 1, role_id: 99, resource_type: 'experiment', resource_pattern: 'exp-1', permission: 'READ' }],
    ...overrides,
  });

describe('PermissionsSection', () => {
  it('renders an empty state when no roles are present', () => {
    renderWithDesignSystem(<PermissionsSection roles={[]} componentId="test" workspacesEnabled={false} />);
    expect(screen.getByText('No permissions')).toBeInTheDocument();
    expect(screen.getByText('No resource permissions to show.')).toBeInTheDocument();
  });

  it('renders a row for a role-derived permission with the role name as Source', () => {
    const roles = [
      role({
        id: 1,
        name: 'team-readers',
        permissions: [{ id: 1, role_id: 1, resource_type: 'experiment', resource_pattern: '42', permission: 'READ' }],
      }),
    ];
    renderWithDesignSystem(<PermissionsSection roles={roles} componentId="test" workspacesEnabled={false} />);
    expect(screen.getByText(/experiment:42/)).toBeInTheDocument();
    expect(screen.getByText('READ')).toBeInTheDocument();
    expect(screen.getByText('team-readers')).toBeInTheDocument();
  });

  it('renders synthetic __user_N__ rows with a localized "Direct" Source', () => {
    renderWithDesignSystem(
      <PermissionsSection
        roles={[
          syntheticRole({
            permissions: [
              { id: 1, role_id: 99, resource_type: 'experiment', resource_pattern: 'exp-1', permission: 'EDIT' },
            ],
          }),
        ]}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText('Direct')).toBeInTheDocument();
    expect(screen.getByText('EDIT')).toBeInTheDocument();
    expect(screen.queryByText('__user_1__')).not.toBeInTheDocument();
  });

  it('dedupes a triple granted via two roles into one row, listing both sources', () => {
    // Same (workspace, type, pattern, permission) emitted by two roles -
    // collapses to a single row, sources joined.
    const roles = [
      role({
        id: 1,
        name: 'role-a',
        permissions: [{ id: 1, role_id: 1, resource_type: 'experiment', resource_pattern: '42', permission: 'READ' }],
      }),
      role({
        id: 2,
        name: 'role-b',
        permissions: [{ id: 2, role_id: 2, resource_type: 'experiment', resource_pattern: '42', permission: 'READ' }],
      }),
    ];
    renderWithDesignSystem(<PermissionsSection roles={roles} componentId="test" workspacesEnabled={false} />);
    // Sources are sorted alphabetically and joined with ', '.
    expect(screen.getByText('role-a, role-b')).toBeInTheDocument();
    // Only one resource cell - no duplicate row.
    expect(screen.getAllByText(/experiment:42/)).toHaveLength(1);
  });

  it('dedupes a role-derived and direct grant on the same triple, joining both labels', () => {
    // Synthetic and regular role granting the same (ws, type, pattern, perm)
    // collapse into one row whose Source lists both the role name and "Direct".
    const roles = [
      role({
        id: 1,
        name: 'team-readers',
        permissions: [{ id: 1, role_id: 1, resource_type: 'experiment', resource_pattern: '42', permission: 'READ' }],
      }),
      syntheticRole({
        permissions: [{ id: 2, role_id: 99, resource_type: 'experiment', resource_pattern: '42', permission: 'READ' }],
      }),
    ];
    renderWithDesignSystem(<PermissionsSection roles={roles} componentId="test" workspacesEnabled={false} />);
    expect(screen.getByText('Direct, team-readers')).toBeInTheDocument();
    expect(screen.getAllByText(/experiment:42/)).toHaveLength(1);
  });

  it('keeps the same triple granted in two workspaces as separate rows', () => {
    // Workspace is part of the dedup key - a multi-tenant deployment
    // should NOT collapse cross-workspace grants.
    const roles = [
      role({
        id: 1,
        name: 'role-a',
        workspace: 'ws-a',
        permissions: [{ id: 1, role_id: 1, resource_type: 'experiment', resource_pattern: '42', permission: 'READ' }],
      }),
      role({
        id: 2,
        name: 'role-b',
        workspace: 'ws-b',
        permissions: [{ id: 2, role_id: 2, resource_type: 'experiment', resource_pattern: '42', permission: 'READ' }],
      }),
    ];
    renderWithDesignSystem(<PermissionsSection roles={roles} componentId="test" workspacesEnabled />);
    expect(screen.getAllByText(/experiment:42/)).toHaveLength(2);
    expect(screen.getByText('ws-a')).toBeInTheDocument();
    expect(screen.getByText('ws-b')).toBeInTheDocument();
  });

  it('hides the Workspace column when workspacesEnabled is false', () => {
    renderWithDesignSystem(
      <PermissionsSection roles={[syntheticRole({})]} componentId="test" workspacesEnabled={false} />,
    );
    expect(screen.queryByText('Workspace')).not.toBeInTheDocument();
  });

  it('shows the Workspace column when workspacesEnabled is true', () => {
    renderWithDesignSystem(<PermissionsSection roles={[syntheticRole({})]} componentId="test" workspacesEnabled />);
    expect(screen.getByText('Workspace')).toBeInTheDocument();
  });

  it('surfaces rolesError as a warning Alert above the table', () => {
    renderWithDesignSystem(
      <PermissionsSection
        roles={[syntheticRole({})]}
        rolesError={new Error('roles backend exploded')}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText('Failed to load permissions')).toBeInTheDocument();
    expect(screen.getByText('roles backend exploded')).toBeInTheDocument();
    // Any rows we managed to receive still render below the alert.
    expect(screen.getByText('Direct')).toBeInTheDocument();
  });

  it('shows a Spinner instead of the table when isLoading', () => {
    renderWithDesignSystem(<PermissionsSection roles={[]} isLoading componentId="test" workspacesEnabled={false} />);
    // No table rendered (no header text); the spinner takes the place.
    expect(screen.queryByText('Resource')).not.toBeInTheDocument();
    expect(screen.queryByText('No permissions')).not.toBeInTheDocument();
  });

  it('sorts rows by workspace, then resource_type, pattern, permission', () => {
    const roles = [
      role({
        id: 1,
        name: 'r1',
        workspace: 'ws-b',
        permissions: [{ id: 1, role_id: 1, resource_type: 'experiment', resource_pattern: '5', permission: 'READ' }],
      }),
      role({
        id: 2,
        name: 'r2',
        workspace: 'ws-a',
        permissions: [
          { id: 2, role_id: 2, resource_type: 'registered_model', resource_pattern: 'm', permission: 'EDIT' },
        ],
      }),
      role({
        id: 3,
        name: 'r3',
        workspace: 'ws-a',
        permissions: [{ id: 3, role_id: 3, resource_type: 'experiment', resource_pattern: '7', permission: 'READ' }],
      }),
    ];
    const { container } = renderWithDesignSystem(
      <PermissionsSection roles={roles} componentId="test" workspacesEnabled />,
    );

    // Find the resource cells in document order; first-row resource should be
    // experiment:7 (ws-a + experiment first), then registered_model:m (ws-a),
    // then experiment:5 (ws-b).
    const codes = within(container).getAllByText(/^(experiment|registered_model):/, { selector: 'code' });
    expect(codes.map((el) => el.textContent)).toEqual(['experiment:7', 'registered_model:m', 'experiment:5']);
  });
});
