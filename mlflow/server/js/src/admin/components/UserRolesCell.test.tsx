import { describe, it, expect } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { UserRolesCell } from './UserRolesCell';
import type { Role } from '../types';

const role = (overrides: Partial<Role>): Role => ({
  id: 1,
  name: 'reader',
  workspace: 'default',
  description: null,
  permissions: [],
  ...overrides,
});

describe('UserRolesCell', () => {
  it('renders an em-dash placeholder when the user has no roles', () => {
    renderWithDesignSystem(<UserRolesCell roles={[]} scopeWorkspace={null} />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('renders one line per role formatted as <workspace> → <name>', () => {
    renderWithDesignSystem(
      <UserRolesCell
        roles={[role({ id: 1, name: 'editor', workspace: 'foo' }), role({ id: 2, name: 'viewer', workspace: 'bar' })]}
        scopeWorkspace={null}
      />,
    );
    expect(screen.getByText('foo')).toBeInTheDocument();
    expect(screen.getByText(/editor/)).toBeInTheDocument();
    expect(screen.getByText('bar')).toBeInTheDocument();
    expect(screen.getByText(/viewer/)).toBeInTheDocument();
  });

  it('filters to roles in scopeWorkspace when set, hiding roles in other workspaces', () => {
    // Per-workspace admin pages pass the active workspace; the cell must
    // omit roles from any other workspace even if the backend returned them
    // (e.g. when the requester is a self-viewer who got their global roles).
    renderWithDesignSystem(
      <UserRolesCell
        roles={[role({ id: 1, name: 'editor', workspace: 'foo' }), role({ id: 2, name: 'viewer', workspace: 'bar' })]}
        scopeWorkspace="foo"
      />,
    );
    expect(screen.getByText('foo')).toBeInTheDocument();
    expect(screen.getByText(/editor/)).toBeInTheDocument();
    expect(screen.queryByText('bar')).not.toBeInTheDocument();
    expect(screen.queryByText(/viewer/)).not.toBeInTheDocument();
  });

  it('falls back to em-dash when scopeWorkspace excludes every role', () => {
    renderWithDesignSystem(<UserRolesCell roles={[role({ workspace: 'foo' })]} scopeWorkspace="bar" />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('shows every role when scopeWorkspace is null (platform-admin path)', () => {
    // Platform admins see roles unscoped — both ``foo`` and ``bar`` rows must
    // render side by side rather than getting filtered.
    renderWithDesignSystem(
      <UserRolesCell
        roles={[role({ id: 1, name: 'editor', workspace: 'foo' }), role({ id: 2, name: 'viewer', workspace: 'bar' })]}
        scopeWorkspace={null}
      />,
    );
    expect(screen.getByText('foo')).toBeInTheDocument();
    expect(screen.getByText('bar')).toBeInTheDocument();
  });
});
