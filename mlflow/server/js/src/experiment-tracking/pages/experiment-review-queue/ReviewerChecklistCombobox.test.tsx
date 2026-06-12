import { describe, jest, it, expect } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import React, { useState } from 'react';

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
  fireEvent.click(screen.getByRole('combobox'));
  return { onToggle };
};

// Stateful harness so toggles actually flip checked state and re-render, exercising
// the order-stability and search-prepend behavior.
const StatefulBox = ({ usernames, initial = [] }: { usernames: string[]; initial?: string[] }) => {
  const [checked, setChecked] = useState<Set<string>>(new Set(initial));
  const onToggle = (u: string) =>
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(u)) {
        next.delete(u);
      } else {
        next.add(u);
      }
      return next;
    });
  return (
    <ReviewerChecklistCombobox
      componentId="test.reviewers"
      usernames={usernames}
      checkedUsers={checked}
      onToggle={onToggle}
      triggerValue={[]}
    />
  );
};

const renderStateful = (props: { usernames: string[]; initial?: string[] }) => {
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <StatefulBox {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  fireEvent.click(screen.getByRole('combobox'));
};

// Assert the named rows appear in this top-to-bottom order in the DOM.
const expectRowOrder = (names: string[]) => {
  for (let i = 1; i < names.length; i++) {
    const prev = screen.getByText(names[i - 1]);
    const curr = screen.getByText(names[i]);
    expect(prev.compareDocumentPosition(curr) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  }
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

  it('seeds the selection first and the default reviewers at the end', () => {
    // bob + carol are selected (carol is off-roster); alice is the only unselected default.
    renderBox({ usernames: ['alice', 'bob'], checkedUsers: new Set(['bob', 'carol']) });
    expectRowOrder(['bob', 'carol', 'alice']);
    expect(screen.getByRole('checkbox', { name: 'bob' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'carol' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'alice' })).not.toBeChecked();
  });

  it('caps the default reviewers at three and hints to search for the rest', () => {
    renderBox({ usernames: ['u1', 'u2', 'u3', 'u4', 'u5'], checkedUsers: new Set() });
    expect(screen.getByRole('checkbox', { name: 'u1' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'u3' })).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'u4' })).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'u5' })).not.toBeInTheDocument();
    expect(screen.getByText(/search to find more reviewers/i)).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'u5' } });
    expect(screen.getByRole('checkbox', { name: 'u5' })).toBeInTheDocument();
  });

  it('does not move a row when it is checked or unchecked', () => {
    renderStateful({ usernames: ['alice', 'bob', 'carol'] });
    expectRowOrder(['alice', 'bob', 'carol']);
    // Selecting then deselecting alice leaves it exactly where it was.
    fireEvent.click(screen.getByRole('checkbox', { name: 'alice' }));
    expect(screen.getByRole('checkbox', { name: 'alice' })).toBeChecked();
    expectRowOrder(['alice', 'bob', 'carol']);
    fireEvent.click(screen.getByRole('checkbox', { name: 'alice' }));
    expect(screen.getByRole('checkbox', { name: 'alice' })).not.toBeChecked();
    expectRowOrder(['alice', 'bob', 'carol']);
  });

  it('compacts to the selected reviewers (pick on top) with fresh defaults when one is chosen from search', () => {
    // sel1/sel2 are already selected; a/b/c/d/target are unselected roster users.
    renderStateful({
      usernames: ['sel1', 'sel2', 'a', 'b', 'c', 'd', 'target'],
      initial: ['sel1', 'sel2'],
    });
    // Seed: selection first, then the first three unselected defaults.
    expectRowOrder(['sel1', 'sel2', 'a', 'b', 'c']);
    expect(screen.queryByRole('checkbox', { name: 'target' })).not.toBeInTheDocument();
    // Find target by search and select it.
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'target' } });
    fireEvent.click(screen.getByRole('checkbox', { name: 'target' }));
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: '' } });
    // Compacts to the selected reviewers (target on top), then a fresh set of three
    // unselected defaults at the bottom.
    expectRowOrder(['target', 'sel1', 'sel2', 'a', 'b', 'c']);
    expect(screen.getByRole('checkbox', { name: 'target' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'a' })).not.toBeChecked();
    // Only three defaults — the fourth unselected user stays search-only.
    expect(screen.queryByRole('checkbox', { name: 'd' })).not.toBeInTheDocument();
  });

  // Sets up a lingering unselected row: add `p` via search, then deselect it in
  // place so it stays in the list (unchecked) until the next recompaction.
  const seedLingeringUnselected = () => {
    renderStateful({ usernames: ['a', 'b', 'c', 'd', 'p'] });
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'p' } });
    fireEvent.click(screen.getByRole('checkbox', { name: 'p' }));
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: '' } });
    fireEvent.click(screen.getByRole('checkbox', { name: 'p' }));
    expect(screen.getByRole('checkbox', { name: 'p' })).not.toBeChecked();
  };

  it('drops a deselected reviewer when the dropdown is reopened', () => {
    seedLingeringUnselected();
    const trigger = screen.getByRole('combobox');
    fireEvent.click(trigger); // close
    fireEvent.click(trigger); // reopen → recompact
    expect(screen.queryByRole('checkbox', { name: 'p' })).not.toBeInTheDocument();
    expectRowOrder(['a', 'b', 'c']);
  });

  it('drops a deselected reviewer when the search is cleared', () => {
    seedLingeringUnselected();
    // Searching and clearing (without selecting anyone) recompacts the list.
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'a' } });
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: '' } });
    expect(screen.queryByRole('checkbox', { name: 'p' })).not.toBeInTheDocument();
    expectRowOrder(['a', 'b', 'c']);
  });

  it('disables unchecked rows once the selection cap is reached', () => {
    const onToggle = jest.fn();
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ReviewerChecklistCombobox
            componentId="test.reviewers"
            usernames={['a', 'b', 'c']}
            checkedUsers={new Set(['a', 'b'])}
            onToggle={onToggle}
            triggerValue={[]}
            maxSelected={2}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    fireEvent.click(screen.getByRole('combobox'));
    // Cap of 2 reached: the unchecked row is disabled (its accessible name carries
    // the disabled reason), while the checked rows stay toggleable.
    expect(screen.getByRole('checkbox', { name: /^c/ })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: 'a' })).not.toBeDisabled();
  });

  it('shows a loading state until the roster resolves, then seeds the defaults', () => {
    // While loading, the roster is still empty (mirrors useAssignableUsersQuery).
    const renderWith = (isLoading: boolean) => (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ReviewerChecklistCombobox
            componentId="test.reviewers"
            usernames={isLoading ? [] : ['alice', 'bob']}
            checkedUsers={new Set()}
            onToggle={jest.fn()}
            triggerValue={[]}
            isLoading={isLoading}
          />
        </DesignSystemProvider>
      </IntlProvider>
    );
    const { rerender } = render(renderWith(true));
    fireEvent.click(screen.getByRole('combobox'));
    expect(screen.getByText(/loading reviewers/i)).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'alice' })).not.toBeInTheDocument();
    // Once the roster resolves, the defaults seed in.
    rerender(renderWith(false));
    expect(screen.getByRole('checkbox', { name: 'alice' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'bob' })).toBeInTheDocument();
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
