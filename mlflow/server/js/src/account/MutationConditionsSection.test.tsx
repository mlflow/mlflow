import { describe, it, expect } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { MutationConditionsSection } from './MutationConditionsSection';
import type { UserRoleConditionRow } from './types';

const condition = (overrides: Partial<UserRoleConditionRow> = {}): UserRoleConditionRow => ({
  id: 1,
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
  it('tells an unrestricted user that nothing restricts them', () => {
    // Rendered rather than hidden: the point of this view is discoverability right
    // after a refused write, and "nothing restricts you" is itself the answer.
    renderWithDesignSystem(<MutationConditionsSection conditions={[]} />);
    expect(screen.getByText('None of your roles carry a mutation condition.')).toBeInTheDocument();
  });

  it('shows the condition and the role it came from', () => {
    renderWithDesignSystem(<MutationConditionsSection conditions={[condition()]} />);
    expect(screen.getByText("tag_key != 'bob'")).toBeInTheDocument();
    expect(screen.getByText('team-writers')).toBeInTheDocument();
  });

  it('renders the same columns the admin per-user view renders', () => {
    // The whole reason this delegates to the shared ``ConditionsTable``: a second table
    // would be free to drift, and both views describe the same policy.
    renderWithDesignSystem(<MutationConditionsSection conditions={[condition()]} />);
    expect(screen.getByText('From Role')).toBeInTheDocument();
  });

  it('attributes a directly-attached condition to "Direct grants", not a synthetic role name', () => {
    // A per-user condition sits on the synthetic ``__user_<id>__`` role (D10). That is an
    // internal name and must never reach the user.
    renderWithDesignSystem(
      <MutationConditionsSection conditions={[condition({ role_id: 99, role_name: '__user_1__' })]} />,
    );
    expect(screen.getByText('Direct grants')).toBeInTheDocument();
    expect(screen.queryByText('__user_1__')).not.toBeInTheDocument();
  });

  it('keeps two conditions from different roles as two rows', () => {
    // They are independent restrictions that must BOTH pass, so collapsing them would
    // misstate the semantics.
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[
          condition({ id: 1, role_id: 1, role_name: 'role-a' }),
          condition({ id: 2, role_id: 2, role_name: 'role-b' }),
        ]}
      />,
    );
    expect(screen.getByText('role-a')).toBeInTheDocument();
    expect(screen.getByText('role-b')).toBeInTheDocument();
  });

  it('shows a target condition as well as a value condition', () => {
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[condition({ value_condition: null, target_condition: "tags.lifecycle = 'dev'" })]}
      />,
    );
    expect(screen.getByText("tags.lifecycle = 'dev'")).toBeInTheDocument();
  });

  it('distinguishes a failed load from an empty list', () => {
    // They look identical and mean opposite things: "nothing restricts you" versus
    // "we do not know".
    renderWithDesignSystem(<MutationConditionsSection conditions={[]} error={new Error('boom')} />);
    expect(screen.queryByText('None of your roles carry a mutation condition.')).not.toBeInTheDocument();
  });
});

describe('MutationConditionsSection — naming the workspace a restriction comes from', () => {
  it('appends the workspace to the source label when workspaces are enabled', () => {
    // The self endpoint deliberately returns conditions from every workspace, and role
    // names repeat across them, so the role name alone does not identify the restriction.
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[
          condition({ id: 1, role_id: 1, role_name: 'writers', workspace: 'team-a' }),
          condition({ id: 2, role_id: 2, role_name: 'writers', workspace: 'team-b' }),
        ]}
        workspacesEnabled
      />,
    );
    expect(screen.getByText('writers (team-a)')).toBeInTheDocument();
    expect(screen.getByText('writers (team-b)')).toBeInTheDocument();
  });

  it('leaves the label bare when workspaces are disabled', () => {
    // A single-workspace deployment has nothing to disambiguate, and the suffix would be
    // noise on every row.
    renderWithDesignSystem(
      <MutationConditionsSection conditions={[condition({ role_name: 'writers', workspace: 'default' })]} />,
    );
    expect(screen.getByText('writers')).toBeInTheDocument();
    expect(screen.queryByText('writers (default)')).not.toBeInTheDocument();
  });

  it('names a direct condition by what it is, with its workspace', () => {
    renderWithDesignSystem(
      <MutationConditionsSection
        conditions={[condition({ role_name: '__user_1__', workspace: 'team-a' })]}
        workspacesEnabled
      />,
    );
    expect(screen.getByText('Direct grants (team-a)')).toBeInTheDocument();
  });
});
