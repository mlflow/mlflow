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

  it('sends the new name when renamed and omits an unchanged owner', async () => {
    renderModal();
    fireEvent.change(screen.getByDisplayValue('My Queue'), { target: { value: 'Renamed' } });
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const arg = mockUpdate.mock.calls[0][0] as Record<string, unknown>;
    expect(arg).toMatchObject({ name: 'Renamed' });
    expect('new_owner' in arg).toBe(false);
  });

  it('sends new_owner when a manager changes the owner', async () => {
    renderModal();
    fireEvent.change(screen.getByPlaceholderText('Owner username or email'), { target: { value: 'bob' } });
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate).toHaveBeenCalledWith(expect.objectContaining({ new_owner: 'bob' }));
  });

  it('hides the owner field and freezes questions for a non-manager editor', async () => {
    renderModal({ canManage: false });
    expect(screen.queryByPlaceholderText('Owner username or email')).toBeNull();
    // A non-manager editor can still rename the queue they own.
    fireEvent.change(screen.getByDisplayValue('My Queue'), { target: { value: 'Renamed' } });
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const arg = mockUpdate.mock.calls[0][0] as Record<string, unknown>;
    // ...but never the questions or owner.
    expect(arg).toMatchObject({ queue_id: 'rq-1', name: 'Renamed' });
    expect('schema_ids' in arg).toBe(false);
    expect('new_owner' in arg).toBe(false);
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
});
