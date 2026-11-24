import { describe, jest, it, expect } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ReviewQueueSidebar } from './ReviewQueueSidebar';
import type { ReviewQueue } from './types';

let mockAuthAvailable = true;
jest.mock('../../../account/hooks', () => ({
  useIsAuthAvailable: () => mockAuthAvailable,
}));
// The pending-count fetch is irrelevant to visibility/sorting/greying; return no
// results so every count renders blank.
jest.mock('@databricks/web-shared/query-client', () => ({
  useQueries: () => [],
}));

const queue = (overrides: Partial<ReviewQueue>): ReviewQueue => ({
  queue_id: 'rq',
  experiment_id: 'exp-1',
  name: 'Queue',
  queue_type: 'CUSTOM',
  creation_time_ms: 0,
  last_update_time_ms: 0,
  ...overrides,
});

// alice owns Alpha, is an assigned member of Gamma, and is neither for Beta.
const QUEUES: ReviewQueue[] = [
  queue({ queue_id: 'a', name: 'Alpha', created_by: 'alice' }),
  queue({ queue_id: 'b', name: 'Beta', created_by: 'bob', users: ['carol'] }),
  queue({ queue_id: 'g', name: 'Gamma', created_by: 'bob', users: ['alice'] }),
];

const renderSidebar = (props: Partial<React.ComponentProps<typeof ReviewQueueSidebar>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ReviewQueueSidebar
          queues={QUEUES}
          selectedQueueId={undefined}
          canManage={false}
          canEdit
          canCreateQueue
          reviewer="alice"
          onSelect={jest.fn()}
          onNewQueue={jest.fn()}
          onManageQuestions={jest.fn()}
          {...props}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ReviewQueueSidebar', () => {
  it('greys queues an editor can see but cannot open (non-owner, non-member)', () => {
    renderSidebar();
    // Owner (Alpha) and member (Gamma) queues are openable; Beta is greyed.
    expect(screen.getByText('Alpha').closest('[role="button"]')).not.toBeNull();
    expect(screen.getByText('Gamma').closest('[role="button"]')).not.toBeNull();
    const beta = screen.getByText('Beta').closest('div');
    expect(beta?.getAttribute('aria-disabled')).toBe('true');
    expect(beta?.getAttribute('role')).toBeNull();
  });

  it('shows the owner column and the My-queues filter on an auth server', () => {
    renderSidebar();
    expect(screen.getByText('Owner')).toBeTruthy();
    expect(screen.getByText('My queues')).toBeTruthy();
  });

  it('narrows to owned queues when My queues is selected', () => {
    renderSidebar();
    fireEvent.click(screen.getByText('My queues'));
    expect(screen.getByText('Alpha')).toBeTruthy();
    // Member-only (Gamma) and unrelated (Beta) queues drop out of the owned view.
    expect(screen.queryByText('Gamma')).toBeNull();
    expect(screen.queryByText('Beta')).toBeNull();
  });

  it('sorts by name and toggles direction when the Queue header is clicked', () => {
    renderSidebar();
    const order = () => screen.getAllByText(/^(Alpha|Beta|Gamma)$/).map((el) => el.textContent);
    expect(order()).toEqual(['Alpha', 'Beta', 'Gamma']);
    fireEvent.click(screen.getByText('Queue'));
    expect(order()).toEqual(['Gamma', 'Beta', 'Alpha']);
  });

  it('does not fire onSelect when a non-inspectable (greyed) row is clicked', () => {
    const onSelect = jest.fn();
    renderSidebar({ onSelect });
    fireEvent.click(screen.getByText('Alpha'));
    expect(onSelect).toHaveBeenCalledWith('a');
    onSelect.mockClear();
    fireEvent.click(screen.getByTitle("You don't have access to this queue."));
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('hides the owner column and filter on a no-auth server', () => {
    mockAuthAvailable = false;
    renderSidebar();
    expect(screen.queryByText('Owner')).toBeNull();
    expect(screen.queryByText('My queues')).toBeNull();
    mockAuthAvailable = true;
  });
});
