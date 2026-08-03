import { describe, jest, it, expect } from '@jest/globals';
import { render, screen, fireEvent, within } from '@testing-library/react';
import React, { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { OwnerCombobox } from './OwnerCombobox';

const renderBox = ({
  usernames,
  selectedUser = '',
  error,
}: {
  usernames: string[];
  selectedUser?: string;
  error?: Error | null;
}) => {
  const onSelect = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <OwnerCombobox
          componentId="test.owner"
          usernames={usernames}
          selectedUser={selectedUser}
          onSelect={onSelect}
          error={error}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  fireEvent.click(screen.getByRole('combobox'));
  return { onSelect };
};

// Stateful harness so a selection actually re-leads the list on the next open.
const StatefulBox = ({ usernames, initial = '' }: { usernames: string[]; initial?: string }) => {
  const [selected, setSelected] = useState(initial);
  return (
    <OwnerCombobox componentId="test.owner" usernames={usernames} selectedUser={selected} onSelect={setSelected} />
  );
};

const renderStateful = (props: { usernames: string[]; initial?: string }) => {
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <StatefulBox {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  fireEvent.click(screen.getByRole('combobox'));
};

const expectRowOrder = (names: string[]) => {
  const listbox = screen.getByRole('listbox');
  for (let i = 1; i < names.length; i++) {
    const prev = within(listbox).getByText(names[i - 1]);
    const curr = within(listbox).getByText(names[i]);
    expect(prev.compareDocumentPosition(curr) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  }
};

describe('OwnerCombobox', () => {
  it('selects exactly one roster user when an option is clicked', () => {
    const { onSelect } = renderBox({ usernames: ['alice', 'bob'] });
    fireEvent.click(within(screen.getByRole('listbox')).getByText('alice'));
    // Single-select: one call, with just the clicked name (close-on-choose is
    // covered by the reopen test below).
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith('alice');
  });

  it('filters the list by the search query', () => {
    renderBox({ usernames: ['alice', 'bob'] });
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'bo' } });
    const listbox = screen.getByRole('listbox');
    expect(within(listbox).queryByText('alice')).not.toBeInTheDocument();
    expect(within(listbox).getByText('bob')).toBeInTheDocument();
  });

  it('leads with the current selection, even when it is off-roster', () => {
    // carol is selected but not in the roster; she still leads the list.
    renderBox({ usernames: ['alice', 'bob'], selectedUser: 'carol' });
    expectRowOrder(['carol', 'alice', 'bob']);
  });

  it('caps the default users at five and hints to search for the rest', () => {
    renderBox({ usernames: ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7'] });
    const listbox = screen.getByRole('listbox');
    expect(within(listbox).getByText('u1')).toBeInTheDocument();
    expect(within(listbox).getByText('u5')).toBeInTheDocument();
    expect(within(listbox).queryByText('u6')).not.toBeInTheDocument();
    expect(screen.getByText(/search to find more users/i)).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'u7' } });
    expect(within(screen.getByRole('listbox')).getByText('u7')).toBeInTheDocument();
  });

  it('caps search matches and hints to refine when too many users match', () => {
    const many = Array.from({ length: 25 }, (_, i) => `user${i}`);
    renderBox({ usernames: many });
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'user' } });
    expect(within(screen.getByRole('listbox')).getAllByRole('option')).toHaveLength(20);
    expect(screen.getByText(/showing the first 20 matches/i)).toBeInTheDocument();
  });

  it('re-leads with a newly selected user on reopen', () => {
    renderStateful({ usernames: ['alice', 'bob', 'carol'], initial: 'alice' });
    fireEvent.click(within(screen.getByRole('listbox')).getByText('carol'));
    // Single-select closes on choose; reopen to see the recompacted order.
    fireEvent.click(screen.getByRole('combobox'));
    expectRowOrder(['carol', 'alice', 'bob']);
  });

  it('shows a loading state until the roster resolves', () => {
    const renderWith = (isLoading: boolean) => (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <OwnerCombobox
            componentId="test.owner"
            usernames={isLoading ? [] : ['alice', 'bob']}
            selectedUser=""
            onSelect={jest.fn()}
            isLoading={isLoading}
          />
        </DesignSystemProvider>
      </IntlProvider>
    );
    const { rerender } = render(renderWith(true));
    fireEvent.click(screen.getByRole('combobox'));
    expect(screen.getByText(/loading users/i)).toBeInTheDocument();
    rerender(renderWith(false));
    expect(within(screen.getByRole('listbox')).getByText('alice')).toBeInTheDocument();
  });

  it('shows an empty state with no assignable users', () => {
    renderBox({ usernames: [] });
    expect(screen.getByText(/no assignable users/i)).toBeInTheDocument();
  });

  it('surfaces a load error instead of the empty state', () => {
    renderBox({ usernames: [], error: new Error('boom') });
    expect(screen.getByText(/couldn't load users/i)).toBeInTheDocument();
    expect(screen.queryByText(/no assignable users/i)).not.toBeInTheDocument();
  });

  it('shows an empty state when the search matches nothing', () => {
    renderBox({ usernames: ['alice', 'bob'] });
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'zzz' } });
    expect(screen.getByText(/no matching users/i)).toBeInTheDocument();
  });
});
