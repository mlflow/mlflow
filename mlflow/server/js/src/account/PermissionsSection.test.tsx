import { describe, it, expect } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { PermissionsSection } from './PermissionsSection';
import type { DirectPermission, Role } from './types';

const role = (overrides: Partial<Role>): Role => ({
  id: 1,
  name: 'reader',
  workspace: 'default',
  description: null,
  permissions: [],
  ...overrides,
});

const direct = (overrides: Partial<DirectPermission>): DirectPermission => ({
  resource_type: 'experiment',
  resource_pattern: 'exp-1',
  permission: 'READ',
  workspace: 'default',
  ...overrides,
});

describe('PermissionsSection', () => {
  it('renders an empty state when neither roles nor direct grants are present', () => {
    renderWithDesignSystem(
      <PermissionsSection roles={[]} directPermissions={[]} componentId="test" workspacesEnabled={false} />,
    );
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
    renderWithDesignSystem(
      <PermissionsSection roles={roles} directPermissions={[]} componentId="test" workspacesEnabled={false} />,
    );
    expect(screen.getByText(/experiment:42/)).toBeInTheDocument();
    expect(screen.getByText('READ')).toBeInTheDocument();
    expect(screen.getByText('team-readers')).toBeInTheDocument();
  });

  it('renders direct grants with a localized "Direct" Source', () => {
    renderWithDesignSystem(
      <PermissionsSection
        roles={[]}
        directPermissions={[direct({ permission: 'EDIT' })]}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText('Direct')).toBeInTheDocument();
    expect(screen.getByText('EDIT')).toBeInTheDocument();
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
    renderWithDesignSystem(
      <PermissionsSection roles={roles} directPermissions={[]} componentId="test" workspacesEnabled={false} />,
    );
    // Sources are sorted alphabetically and joined with ', '.
    expect(screen.getByText('role-a, role-b')).toBeInTheDocument();
    // Only one resource cell - no duplicate row.
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
    renderWithDesignSystem(
      <PermissionsSection roles={roles} directPermissions={[]} componentId="test" workspacesEnabled />,
    );
    expect(screen.getAllByText(/experiment:42/)).toHaveLength(2);
    expect(screen.getByText('ws-a')).toBeInTheDocument();
    expect(screen.getByText('ws-b')).toBeInTheDocument();
  });

  it('hides the Workspace column when workspacesEnabled is false', () => {
    renderWithDesignSystem(
      <PermissionsSection roles={[]} directPermissions={[direct({})]} componentId="test" workspacesEnabled={false} />,
    );
    expect(screen.queryByText('Workspace')).not.toBeInTheDocument();
  });

  it('shows the Workspace column when workspacesEnabled is true', () => {
    renderWithDesignSystem(
      <PermissionsSection roles={[]} directPermissions={[direct({})]} componentId="test" workspacesEnabled />,
    );
    expect(screen.getByText('Workspace')).toBeInTheDocument();
  });

  it('surfaces rolesError as a warning Alert above the table', () => {
    renderWithDesignSystem(
      <PermissionsSection
        roles={[]}
        directPermissions={[direct({})]}
        rolesError={new Error('roles backend exploded')}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText('Failed to load role-derived permissions')).toBeInTheDocument();
    expect(screen.getByText('roles backend exploded')).toBeInTheDocument();
    // Direct grants still render below the alert.
    expect(screen.getByText('Direct')).toBeInTheDocument();
  });

  it('surfaces directPermsError as a warning Alert above the table', () => {
    const roles = [
      role({
        id: 1,
        name: 'team-readers',
        permissions: [{ id: 1, role_id: 1, resource_type: 'experiment', resource_pattern: '42', permission: 'READ' }],
      }),
    ];
    renderWithDesignSystem(
      <PermissionsSection
        roles={roles}
        directPermissions={[]}
        directPermsError={new Error('direct backend exploded')}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText('Failed to load direct permissions')).toBeInTheDocument();
    expect(screen.getByText('direct backend exploded')).toBeInTheDocument();
    // Role-derived rows still render.
    expect(screen.getByText('team-readers')).toBeInTheDocument();
  });

  it('shows a Spinner instead of the table when isLoading', () => {
    renderWithDesignSystem(
      <PermissionsSection roles={[]} directPermissions={[]} isLoading componentId="test" workspacesEnabled={false} />,
    );
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
      <PermissionsSection roles={roles} directPermissions={[]} componentId="test" workspacesEnabled />,
    );

    // Find the resource cells in document order; first-row resource should be
    // experiment:7 (ws-a + experiment first), then registered_model:m (ws-a),
    // then experiment:5 (ws-b).
    const codes = within(container).getAllByText(/^(experiment|registered_model):/, { selector: 'code' });
    expect(codes.map((el) => el.textContent)).toEqual(['experiment:7', 'registered_model:m', 'experiment:5']);
  });
});
