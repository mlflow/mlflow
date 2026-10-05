import { describe, it, expect } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { MutationConditionsSection } from './MutationConditionsSection';
import type { UserRoleConditionRow } from './types';

const condition = (overrides: Partial<UserRoleConditionRow> = {}): UserRoleConditionRow => ({
  role_id: 1,
  role_name: 'team-writers',
  workspace: 'default',
  resource_type: 'run',
  resource_pattern: '*',
  container_resource_type: 'workspace',
  container_resource_pattern: '*',
  value_condition: "tag_key != 'bob'",
  target_condition: null,
  condition_slot: 0,
  ...overrides,
});

describe('MutationConditionsSection', () => {
  it('tells an unrestricted user their permissions apply in full', () => {
    // Rendered rather than hidden: the point of this view is discoverability right
    // after a refused write, and "nothing restricts you" is itself the answer.
    renderWithDesignSystem(<MutationConditionsSection conditions={[]} componentId="test" workspacesEnabled={false} />);
    expect(screen.getByText('No conditions')).toBeInTheDocument();
    expect(screen.getByText(/Your permissions apply in full/)).toBeInTheDocument();
  });

  it('shows the condition text and the role it came from', () => {
    renderWithDesignSystem(
      <MutationConditionsSection conditions={[condition()]} componentId="test" workspacesEnabled={false} />,
    );
    expect(screen.getByText("tag_key != 'bob'")).toBeInTheDocument();
    expect(screen.getByText('team-writers')).toBeInTheDocument();
    expect(screen.getByText('run')).toBeInTheDocument();
  });

  it('attributes a directly-attached condition to "Direct" rather than a synthetic role name', () => {
    // A per-user condition lives on the synthetic ``__user_<id>__`` role (D10). That is
    // an internal name and must never be shown to the user.
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[condition({ role_id: 99, role_name: '__user_1__' })]}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText('Direct')).toBeInTheDocument();
    expect(screen.queryByText('__user_1__')).not.toBeInTheDocument();
  });

  it('marks the half of a condition that is absent as unrestricted', () => {
    // Either filter may be null, meaning unconstrained in that direction. Showing a
    // blank cell would read as "nothing may be set" -- the opposite of the truth.
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[condition({ value_condition: null, target_condition: "tags.lifecycle = 'dev'" })]}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText("tags.lifecycle = 'dev'")).toBeInTheDocument();
    expect(screen.getByText('Unrestricted')).toBeInTheDocument();
  });

  it('says "All" when a condition is not narrowed to a resource or container', () => {
    renderWithDesignSystem(
      <MutationConditionsSection conditions={[condition()]} componentId="test" workspacesEnabled={false} />,
    );
    expect(screen.getByText('All')).toBeInTheDocument();
  });

  it('describes a container-scoped condition by its container', () => {
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[condition({ container_resource_type: 'experiment', container_resource_pattern: '42' })]}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText(/in experiment:42/)).toBeInTheDocument();
  });

  it('keeps two conditions from different roles as two rows', () => {
    // They are independent restrictions that must BOTH pass, so deduping them the way
    // the permissions table dedupes grants would misstate the semantics.
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[condition({ role_id: 1, role_name: 'role-a' }), condition({ role_id: 2, role_name: 'role-b' })]}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText('role-a')).toBeInTheDocument();
    expect(screen.getByText('role-b')).toBeInTheDocument();
  });

  it('surfaces a load failure without hiding the rows it has', () => {
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[condition()]}
        error={new Error('boom')}
        componentId="test"
        workspacesEnabled={false}
      />,
    );
    expect(screen.getByText('Failed to load conditions')).toBeInTheDocument();
    expect(screen.getByText("tag_key != 'bob'")).toBeInTheDocument();
  });

  it('hides the workspace column when workspaces are disabled', () => {
    renderWithDesignSystem(
      <MutationConditionsSection conditions={[condition()]} componentId="test" workspacesEnabled={false} />,
    );
    expect(screen.queryByText('Workspace')).not.toBeInTheDocument();
  });
});
