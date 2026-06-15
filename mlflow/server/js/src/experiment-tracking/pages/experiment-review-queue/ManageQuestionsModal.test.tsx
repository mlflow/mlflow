import { describe, beforeEach, jest, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ManageQuestionsModal } from './ManageQuestionsModal';
import type { ReviewQueue } from './types';

const mockDelete = jest.fn();
jest.mock('../../components/label-schemas', () => ({
  useListLabelSchemasQuery: () => ({
    labelSchemas: [{ schema_id: 's1', name: 'Q1', type: 'FEEDBACK', input: { text: {} } }],
    isLoading: false,
  }),
  useDeleteLabelSchemaMutation: () => ({ deleteLabelSchema: mockDelete, isDeleting: false }),
  LabelSchemaFormModal: () => null,
}));

// Typed to the real wire shape (Pick) so the queue_type literal union and field
// names stay in sync with the production filter under test.
let mockQueues: Pick<ReviewQueue, 'queue_id' | 'name' | 'queue_type' | 'schema_ids'>[] = [];
jest.mock('./hooks/useListReviewQueuesQuery', () => ({
  useListReviewQueuesQuery: () => ({ reviewQueues: mockQueues }),
}));

jest.mock('./hooks/useReviewQueueSearchParams', () => ({
  getReviewQueuePageRoute: (experimentId: string, queueId?: string) =>
    `/experiments/${experimentId}/review-queue${queueId ? `?selectedQueueId=${queueId}` : ''}`,
}));

// Stub the router Link to a plain anchor so the test needs no Router context;
// forward `rel` too so the new-tab safety attribute can be asserted.
jest.mock('../../../common/utils/RoutingUtils', () => ({
  Link: ({ to, children, target, rel }: { to: string; children: React.ReactNode; target?: string; rel?: string }) => (
    <a href={to} target={target} rel={rel}>
      {children}
    </a>
  ),
}));

const renderModal = () =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ManageQuestionsModal experimentId="exp-1" onClose={jest.fn()} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ManageQuestionsModal delete confirmation', () => {
  // Reset to a known baseline so a prior test's queues can't bleed into the next.
  beforeEach(() => {
    mockQueues = [];
  });

  it('lists the custom queues that use the question (as deep links) and excludes others', async () => {
    mockQueues = [
      { queue_id: 'rq-a', name: 'Queue A', queue_type: 'CUSTOM', schema_ids: ['s1'] },
      { queue_id: 'rq-b', name: 'Queue B', queue_type: 'CUSTOM', schema_ids: ['s2'] },
      { queue_id: 'rq-u', name: 'alice', queue_type: 'USER', schema_ids: [] },
    ];
    renderModal();
    await userEvent.click(screen.getByRole('button', { name: 'Delete question' }));

    expect(screen.getByText('Delete question?')).toBeInTheDocument();
    // Heading counts both the custom queues using it and the user queues (which
    // inherit every question).
    expect(screen.getByText('Used by 1 custom queue and 1 user queue:')).toBeInTheDocument();
    // The one custom queue that uses the question is a new-tab deep link to it,
    // carrying the rel safety attributes that pair with target="_blank".
    const link = screen.getByRole('link', { name: 'Queue A' });
    expect(link).toHaveAttribute('href', '/experiments/exp-1/review-queue?selectedQueueId=rq-a');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    // A custom queue that doesn't use it, and the inheriting user queue, are not listed.
    expect(screen.queryByRole('link', { name: 'Queue B' })).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'alice' })).not.toBeInTheDocument();
  });

  it('shows a plain user-queue note when no custom queue uses the question', async () => {
    mockQueues = [
      { queue_id: 'rq-b', name: 'Queue B', queue_type: 'CUSTOM', schema_ids: ['s2'] },
      { queue_id: 'rq-u1', name: 'alice', queue_type: 'USER', schema_ids: [] },
      { queue_id: 'rq-u2', name: 'bob', queue_type: 'USER', schema_ids: [] },
    ];
    renderModal();
    await userEvent.click(screen.getByRole('button', { name: 'Delete question' }));

    expect(screen.getByText('Inherited by 2 user queues.')).toBeInTheDocument();
    // No custom queue uses it, so there are no queue links.
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('omits the queue section entirely when no custom or user queue is affected', async () => {
    mockQueues = [{ queue_id: 'rq-b', name: 'Queue B', queue_type: 'CUSTOM', schema_ids: ['s2'] }];
    renderModal();
    await userEvent.click(screen.getByRole('button', { name: 'Delete question' }));

    expect(screen.getByText('Delete question?')).toBeInTheDocument();
    expect(screen.queryByText(/Used by/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Inherited by/)).not.toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });
});
