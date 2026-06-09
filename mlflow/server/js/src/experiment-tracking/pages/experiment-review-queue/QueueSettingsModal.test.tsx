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
}));
jest.mock('../../../account/hooks', () => ({
  useCurrentUserIsAdmin: () => false,
  useCurrentUserIsWorkspaceAdmin: () => false,
  useIsAuthAvailable: () => false,
}));
jest.mock('./hooks/useAssignableUsersQuery', () => ({ useAssignableUsersQuery: () => ({ users: [] }) }));

let mockTraces: ReviewQueueItem[] = [];
jest.mock('./hooks/useListReviewQueueTracesQuery', () => ({
  useListReviewQueueTracesQuery: () => ({ items: mockTraces, isLoading: false }),
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

const trace = (targetId: string): ReviewQueueItem => ({
  queue_id: 'rq-1',
  target_type: 'TRACE',
  target_id: targetId,
  status: 'PENDING',
  creation_time_ms: 0,
  last_update_time_ms: 0,
});

const renderModal = () =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueueSettingsModal queue={queue} onClose={jest.fn()} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('QueueSettingsModal save', () => {
  beforeEach(() => {
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
});
