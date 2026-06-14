import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { QueueSettingsModal } from './QueueSettingsModal';
import type { ReviewQueue, ReviewQueueItem } from './types';

// The questions checklist and rich previews aren't under test here — the focus is
// the schema-freeze save logic — so stub them out.
jest.mock('./QuestionChecklistCombobox', () => ({ QuestionChecklistCombobox: () => null }));
jest.mock('../../components/label-schemas', () => ({
  useListLabelSchemasQuery: () => ({
    labelSchemas: [{ schema_id: 's1', name: 'Q1', type: 'FEEDBACK', input: { text: {} } }],
    isLoading: false,
  }),
  LabelSchemaInputRenderer: () => null,
  LabelSchemaFormModal: () => null,
}));
let mockAuthAvailable = true;
jest.mock('../../../account/hooks', () => ({
  useIsAuthAvailable: () => mockAuthAvailable,
}));
// Stable reference across renders, like the real react-query hook — a fresh
// object each call would churn the members typeahead's derived state.
jest.mock('./hooks/useAssignableUsersQuery', () => {
  const result = { users: [] };
  return { useAssignableUsersQuery: () => result };
});

let mockTraces: ReviewQueueItem[] = [];
jest.mock('./hooks/useListReviewQueueItemsQuery', () => ({
  useListReviewQueueItemsQuery: () => ({ items: mockTraces, isLoading: false }),
}));

const mockUpdate = jest.fn();
jest.mock('./hooks/useUpdateReviewQueueMutation', () => ({
  useUpdateReviewQueueMutation: () => ({
    updateReviewQueueAsync: mockUpdate,
    isUpdatingQueue: false,
    error: null,
  }),
}));

const queue: ReviewQueue = {
  queue_id: 'rq-1',
  experiment_id: 'exp-1',
  name: 'My Queue',
  queue_type: 'CUSTOM',
  creation_time_ms: 0,
  last_update_time_ms: 0,
  users: ['alice'],
  schema_ids: ['s1'],
};

const trace = (itemId: string): ReviewQueueItem => ({
  queue_id: 'rq-1',
  item_type: 'TRACE',
  item_id: itemId,
  status: 'PENDING',
  creation_time_ms: 0,
  last_update_time_ms: 0,
});

const renderModal = ({ canManage = true }: { canManage?: boolean } = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueueSettingsModal queue={queue} canManage={canManage} onClose={jest.fn()} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('QueueSettingsModal save', () => {
  beforeEach(() => {
    mockAuthAvailable = true;
    mockTraces = [];
    mockUpdate.mockReset();
    mockUpdate.mockImplementation(() => Promise.resolve());
  });

  it('sends schema_ids when the queue has no traces (questions editable)', async () => {
    mockTraces = [];
    renderModal();
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate).toHaveBeenCalledWith(expect.objectContaining({ queue_id: 'rq-1', schema_ids: ['s1'] }));
  });

  it('omits schema_ids once the queue has traces (questions frozen)', async () => {
    mockTraces = [trace('tr-1')];
    renderModal();
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const arg = mockUpdate.mock.calls[0][0] as Record<string, unknown>;
    expect('schema_ids' in arg).toBe(false);
    expect(arg).toMatchObject({ queue_id: 'rq-1' });
  });

  it('freezes questions for a non-manager editor (schema_ids omitted on save)', async () => {
    renderModal({ canManage: false });
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const arg = mockUpdate.mock.calls[0][0] as Record<string, unknown>;
    expect('schema_ids' in arg).toBe(false);
    expect(arg).toMatchObject({ queue_id: 'rq-1' });
  });

  it('omits unchanged fields entirely on a no-op save', async () => {
    renderModal();
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const arg = mockUpdate.mock.calls[0][0] as Record<string, unknown>;
    // No member change => no `users` write that could clobber a concurrent edit.
    expect('users' in arg).toBe(false);
    expect('name' in arg).toBe(false);
    expect('new_owner' in arg).toBe(false);
  });

  it('hides the queue owner from the picker but keeps them assigned on save', async () => {
    const ownedQueue: ReviewQueue = { ...queue, created_by: 'owner1', users: ['owner1', 'alice'] };
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueueSettingsModal queue={ownedQueue} canManage onClose={jest.fn()} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    fireEvent.click(screen.getByRole('combobox'));
    // The owner is auto-assigned, so it isn't a toggleable row; the other member is.
    expect(screen.queryByRole('checkbox', { name: 'owner1' })).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'alice' })).toBeInTheDocument();
    // Remove the only visible member, then save — the owner stays assigned.
    fireEvent.click(screen.getByRole('checkbox', { name: 'alice' }));
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate).toHaveBeenCalledWith(expect.objectContaining({ users: ['owner1'] }));
  });
});
