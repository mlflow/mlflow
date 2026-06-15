import { describe, jest, it, expect } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ManageQuestionsModal } from './ManageQuestionsModal';

const mockDelete = jest.fn();
jest.mock('../../components/label-schemas', () => ({
  useListLabelSchemasQuery: () => ({
    labelSchemas: [{ schema_id: 's1', name: 'Q1', type: 'FEEDBACK', input: { text: {} } }],
    isLoading: false,
  }),
  useDeleteLabelSchemaMutation: () => ({ deleteLabelSchema: mockDelete, isDeleting: false }),
  LabelSchemaFormModal: () => null,
}));

let mockQueues: { queue_id: string; name: string; queue_type: string; schema_ids: string[] }[] = [];
jest.mock('./hooks/useListReviewQueuesQuery', () => ({
  useListReviewQueuesQuery: () => ({ reviewQueues: mockQueues }),
}));

jest.mock('./hooks/useReviewQueueSearchParams', () => ({
  getReviewQueuePageRoute: (experimentId: string, queueId?: string) =>
    `/experiments/${experimentId}/review-queue${queueId ? `?selectedQueueId=${queueId}` : ''}`,
}));

// Stub the router Link to a plain anchor so the test needs no Router context.
jest.mock('../../../common/utils/RoutingUtils', () => ({
  Link: ({ to, children, target }: { to: string; children: React.ReactNode; target?: string }) => (
    <a href={to} target={target}>
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
  it('lists the custom queues that use the question (as deep links) and excludes others', () => {
    mockQueues = [
      { queue_id: 'rq-a', name: 'Queue A', queue_type: 'CUSTOM', schema_ids: ['s1'] },
      { queue_id: 'rq-b', name: 'Queue B', queue_type: 'CUSTOM', schema_ids: ['s2'] },
      { queue_id: 'rq-u', name: 'alice', queue_type: 'USER', schema_ids: [] },
    ];
    renderModal();
    fireEvent.click(screen.getByRole('button', { name: 'Delete question' }));

    expect(screen.getByText('Delete question?')).toBeInTheDocument();
    // The one custom queue that uses the question is a new-tab deep link to it.
    const link = screen.getByRole('link', { name: 'Queue A' });
    expect(link).toHaveAttribute('href', '/experiments/exp-1/review-queue?selectedQueueId=rq-a');
    expect(link).toHaveAttribute('target', '_blank');
    // A custom queue that doesn't use it, and the inheriting user queue, are not listed.
    expect(screen.queryByRole('link', { name: 'Queue B' })).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'alice' })).not.toBeInTheDocument();
  });

  it('omits the queue list when no custom queue uses the question', () => {
    mockQueues = [{ queue_id: 'rq-b', name: 'Queue B', queue_type: 'CUSTOM', schema_ids: ['s2'] }];
    renderModal();
    fireEvent.click(screen.getByRole('button', { name: 'Delete question' }));

    expect(screen.getByText('Delete question?')).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });
});
