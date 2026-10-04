import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { ConditionsTable } from './ConditionsTable';

const condition = (overrides: Record<string, unknown> = {}) => ({
  id: 1,
  role_id: 7,
  resource_type: 'run',
  condition_slot: 1,
  parent_resource_type: null,
  parent_resource_id: null,
  value_condition: null,
  target_condition: "tags.lifecycle != 'prod'",
  ...overrides,
});

describe('ConditionsTable', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows both filters and the scope', () => {
    renderWithDesignSystem(
      <ConditionsTable
        conditions={[
          condition() as any,
          condition({
            id: 2,
            resource_type: 'trace',
            parent_resource_type: 'experiment',
            parent_resource_id: '42',
            value_condition: "tag_value != 'prod'",
            target_condition: null,
          }) as any,
        ]}
      />,
    );
    expect(screen.getByText("tags.lifecycle != 'prod'")).toBeInTheDocument();
    expect(screen.getByText("tag_value != 'prod'")).toBeInTheDocument();
    expect(screen.getByText('All in workspace')).toBeInTheDocument();
    expect(screen.getByText('Experiment 42')).toBeInTheDocument();
  });

  it('reports a fetch failure instead of rendering an empty list', () => {
    // An empty table and a failed fetch look identical and mean opposite things:
    // "nothing restricts this" versus "we do not know".
    renderWithDesignSystem(<ConditionsTable conditions={[]} error={new Error('boom')} />);
    expect(screen.getByText('Failed to load mutation conditions')).toBeInTheDocument();
    expect(screen.queryByText('No mutation conditions')).not.toBeInTheDocument();
  });

  it('renders the supplied empty description', () => {
    renderWithDesignSystem(
      <ConditionsTable conditions={[]} emptyDescription="Use Edit role to add mutation conditions." />,
    );
    expect(screen.getByText('No mutation conditions')).toBeInTheDocument();
    expect(screen.getByText('Use Edit role to add mutation conditions.')).toBeInTheDocument();
  });

  it('shows the carrying role only when a suffix header is given', () => {
    // Asserting the column COUNT, not just the absent label: rendering the column
    // unconditionally with an undefined header produces a blank fifth column, which a
    // label-only assertion cannot see.
    const rows = [condition({ role_id: 19 }) as any];
    const { unmount } = renderWithDesignSystem(<ConditionsTable conditions={rows} />);
    expect(screen.getAllByRole('columnheader')).toHaveLength(4);
    expect(screen.queryByText('From Role')).not.toBeInTheDocument();
    unmount();

    renderWithDesignSystem(
      <ConditionsTable conditions={rows} suffixHeader="From Role" rowSuffix={() => 'Direct grants'} />,
    );
    expect(screen.getAllByRole('columnheader')).toHaveLength(5);
    expect(screen.getByText('From Role')).toBeInTheDocument();
    expect(screen.getByText('Direct grants')).toBeInTheDocument();
  });

  it('keys rows by role and condition id together', () => {
    // Condition slots are allocated per (role, resource type), so two roles can both
    // carry id 1. A condition-id-only key duplicates, which React reports rather than
    // renders wrongly -- so the warning is the only observable symptom.
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    try {
      renderWithDesignSystem(
        <ConditionsTable
          conditions={[
            condition({ id: 1, role_id: 7, target_condition: "tags.a = '1'" }) as any,
            condition({ id: 1, role_id: 8, target_condition: "tags.b = '2'" }) as any,
          ]}
        />,
      );
      expect(screen.getByText("tags.a = '1'")).toBeInTheDocument();
      expect(screen.getByText("tags.b = '2'")).toBeInTheDocument();
      const duplicateKeyWarning = errorSpy.mock.calls.some((args) =>
        args.some((a) => typeof a === 'string' && a.includes('same key')),
      );
      expect(duplicateKeyWarning).toBe(false);
    } finally {
      errorSpy.mockRestore();
    }
  });
});
