import { describe, jest, it, expect } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ReviewerChecklistCombobox } from './ReviewerChecklistCombobox';

const renderBox = ({ usernames, checkedUsers }: { usernames: string[]; checkedUsers: Set<string> }) => {
  const onToggle = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ReviewerChecklistCombobox
          componentId="test.reviewers"
          usernames={usernames}
          checkedUsers={checkedUsers}
          onToggle={onToggle}
          triggerValue={[]}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  // Open the dropdown.
  fireEvent.click(screen.getByRole('combobox'));
  return { onToggle };
};

describe('ReviewerChecklistCombobox', () => {
  it('toggles a roster user when its checkbox is clicked', () => {
    const { onToggle } = renderBox({ usernames: ['alice', 'bob'], checkedUsers: new Set() });
    fireEvent.click(screen.getByRole('checkbox', { name: 'alice' }));
    expect(onToggle).toHaveBeenCalledWith('alice');
  });

  it('filters the list by the search query', () => {
    renderBox({ usernames: ['alice', 'bob'], checkedUsers: new Set() });
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'bo' } });
    expect(screen.queryByRole('checkbox', { name: 'alice' })).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'bob' })).toBeInTheDocument();
  });

  it('groups the selection under a "Selected" header when not searching, including off-roster members', () => {
    // bob is in the roster + selected; carol is selected but not in the roster.
    renderBox({ usernames: ['alice', 'bob'], checkedUsers: new Set(['bob', 'carol']) });
    expect(screen.getByText('Selected')).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'bob' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'carol' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'alice' })).not.toBeChecked();
  });

  it('caps the unselected list and hints to search when there are more than three reviewers', () => {
    renderBox({ usernames: ['u1', 'u2', 'u3', 'u4', 'u5'], checkedUsers: new Set() });
    // Only the first three unselected reviewers show before searching.
    expect(screen.getByRole('checkbox', { name: 'u1' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'u3' })).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'u4' })).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'u5' })).not.toBeInTheDocument();
    expect(screen.getByText(/search to find more reviewers/i)).toBeInTheDocument();
    // The capped reviewers are still reachable by search.
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'u5' } });
    expect(screen.getByRole('checkbox', { name: 'u5' })).toBeInTheDocument();
  });

  it('orders the selection newest-first (the latest-added reviewer on top)', () => {
    // The Set's insertion order is the order the modal added picks; the newest
    // (last inserted) should render at the top of the "Selected" group.
    renderBox({ usernames: [], checkedUsers: new Set(['first', 'second', 'third']) });
    const third = screen.getByText('third');
    const second = screen.getByText('second');
    const first = screen.getByText('first');
    expect(third.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(second.compareDocumentPosition(first) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('shows an empty state with no assignable reviewers', () => {
    renderBox({ usernames: [], checkedUsers: new Set() });
    expect(screen.getByText(/no assignable reviewers/i)).toBeInTheDocument();
  });

  it('shows an empty state when the search matches nothing', () => {
    renderBox({ usernames: ['alice', 'bob'], checkedUsers: new Set() });
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'zzz' } });
    expect(screen.getByText(/no matching reviewers/i)).toBeInTheDocument();
  });
});
