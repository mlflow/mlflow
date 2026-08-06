import { describe, it, expect, jest } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { RoleUsersSection } from './RoleUsersSection';

// DialogCombobox masks pointer-event hit detection on its overlay; disable
// the check so userEvent.click can reach the trigger.
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

// ``jest.mock`` is hoisted above imports by the Jest transformer, so this
// mock applies before ``./RoleUsersSection`` reaches in to ``../hooks``.
jest.mock('../hooks', () => ({
  useUsersQuery: () => ({
    data: {
      users: [
        { id: 1, username: 'alice', is_admin: false, roles: [] },
        { id: 2, username: 'bob', is_admin: false, roles: [] },
        { id: 3, username: 'carol', is_admin: false, roles: [] },
      ],
    },
    isLoading: false,
    error: null,
  }),
}));

describe('RoleUsersSection — multi-select trigger', () => {
  // Regression for #23399: see ``RoleAssignmentForm.test.tsx`` for the bug
  // description. Both pickers share the same `triggerValue` collapsing
  // pattern; this exercises the user-multi-select half.

  it('renders "N users selected" exactly once when multiple users are picked', () => {
    renderWithDesignSystem(<RoleUsersSection value={['alice', 'bob', 'carol']} onChange={() => {}} />);
    const summary = screen.getAllByText('3 users selected');
    expect(summary).toHaveLength(1);
  });

  it('renders the single username (not "1 users selected") when only one is picked', () => {
    renderWithDesignSystem(<RoleUsersSection value={['bob']} onChange={() => {}} />);
    expect(screen.getByText('bob')).toBeInTheDocument();
    expect(screen.queryByText(/users selected/)).not.toBeInTheDocument();
  });

  it('falls through to the placeholder when nothing is picked', () => {
    renderWithDesignSystem(<RoleUsersSection value={[]} onChange={() => {}} />);
    expect(screen.getByText('Select one or more users')).toBeInTheDocument();
    expect(screen.queryByText(/users selected/)).not.toBeInTheDocument();
  });

  it('renders pre-selected users as aria-selected when the dropdown is opened', async () => {
    // See ``RoleAssignmentForm.test.tsx`` — same value-vs-item decoupling.
    renderWithDesignSystem(<RoleUsersSection value={['alice', 'carol']} onChange={() => {}} />);
    await userEvent.click(screen.getByRole('combobox'));
    const aliceOption = await screen.findByRole('option', { name: /alice/ });
    const bobOption = await screen.findByRole('option', { name: /bob/ });
    const carolOption = await screen.findByRole('option', { name: /carol/ });
    expect(aliceOption).toHaveAttribute('aria-selected', 'true');
    expect(bobOption).toHaveAttribute('aria-selected', 'false');
    expect(carolOption).toHaveAttribute('aria-selected', 'true');
  });
});
