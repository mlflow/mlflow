import { describe, it, expect, jest } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { RoleAssignmentForm } from './RoleAssignmentForm';

// DialogCombobox masks pointer-event hit detection on its overlay; disable
// the check so userEvent.click can reach the trigger.
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

// ``jest.mock`` is hoisted above imports by the Jest transformer, so these
// mocks apply before ``./RoleAssignmentForm`` reaches in to ``../hooks``.
jest.mock('../hooks', () => ({
  useCurrentUserIsAdmin: () => true,
  useRolesQuery: () => ({
    data: {
      roles: [
        { id: 1, name: 'reader', workspace: 'default', description: null, permissions: [] },
        { id: 2, name: 'writer', workspace: 'default', description: null, permissions: [] },
        { id: 3, name: 'admin', workspace: 'default', description: null, permissions: [] },
        // Synthetic per-user role — must be filtered out of the dropdown.
        { id: 99, name: '__user_99__', workspace: 'default', description: null, permissions: [] },
      ],
    },
    isLoading: false,
    error: null,
  }),
}));

jest.mock('../../workspaces/utils/WorkspaceUtils', () => ({
  useActiveWorkspace: () => null,
}));

describe('RoleAssignmentForm — multi-select trigger', () => {
  // Regression for #23399: the trigger used to call ``renderDisplayedValue``
  // once per entry in the ``value`` array, producing "N roles selected, N
  // roles selected, …" repeated N times. The fix collapses ``value`` to a
  // single-element array carrying the summarised label so the count text
  // renders exactly once.

  it('renders "N roles selected" exactly once when multiple roles are picked', () => {
    renderWithDesignSystem(<RoleAssignmentForm value={{ roleIds: [1, 2, 3] }} onChange={() => {}} />);
    const summary = screen.getAllByText('3 roles selected');
    expect(summary).toHaveLength(1);
  });

  it('renders the single role label (not "1 roles selected") when only one is picked', () => {
    renderWithDesignSystem(<RoleAssignmentForm value={{ roleIds: [2] }} onChange={() => {}} />);
    expect(screen.getByText('default/writer')).toBeInTheDocument();
    expect(screen.queryByText(/roles selected/)).not.toBeInTheDocument();
  });

  it('falls through to the placeholder when nothing is picked', () => {
    renderWithDesignSystem(<RoleAssignmentForm value={{ roleIds: [] }} onChange={() => {}} />);
    expect(screen.getByText('Select one or more roles')).toBeInTheDocument();
    expect(screen.queryByText(/roles selected/)).not.toBeInTheDocument();
  });

  it('renders pre-selected items as aria-selected when the dropdown is opened', async () => {
    // Pins that each item's explicit ``checked`` prop — not the parent's
    // ``value`` array — drives ``aria-selected`` after the trigger fix.
    renderWithDesignSystem(<RoleAssignmentForm value={{ roleIds: [1, 3] }} onChange={() => {}} />);
    await userEvent.click(screen.getByRole('combobox'));
    const readerOption = await screen.findByRole('option', { name: /default\/reader/ });
    const writerOption = await screen.findByRole('option', { name: /default\/writer/ });
    const adminOption = await screen.findByRole('option', { name: /default\/admin/ });
    expect(readerOption).toHaveAttribute('aria-selected', 'true');
    expect(writerOption).toHaveAttribute('aria-selected', 'false');
    expect(adminOption).toHaveAttribute('aria-selected', 'true');
  });

  it('hides synthetic __user_N__ roles from the dropdown options', async () => {
    renderWithDesignSystem(<RoleAssignmentForm value={{ roleIds: [] }} onChange={() => {}} />);
    await userEvent.click(screen.getByRole('combobox'));
    await screen.findByRole('option', { name: /default\/reader/ });
    expect(screen.queryByRole('option', { name: /__user_99__/ })).not.toBeInTheDocument();
  });
});
