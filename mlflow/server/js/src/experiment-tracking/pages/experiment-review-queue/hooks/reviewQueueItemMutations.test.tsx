import { describe, it, expect, jest } from '@jest/globals';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { useAddItemsToReviewQueueMutation } from './useAddItemsToReviewQueueMutation';
import { LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY } from './useListReviewQueueItemsQuery';
import { LIST_REVIEW_QUEUES_QUERY_KEY } from './useListReviewQueuesQuery';
import { useRemoveItemsFromReviewQueueMutation } from './useRemoveItemsFromReviewQueueMutation';

jest.mock('../../../../common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(() => Promise.resolve({})),
  getAjaxUrl: (path: string) => path,
}));

// Renders the given mutation hook and fires it on click, against a real
// QueryClient whose invalidateQueries is spied on.
const renderMutation = (useMutationHook: () => { run: (p: { queue_id: string; item_ids: string[] }) => unknown }) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const spy = jest.spyOn(queryClient, 'invalidateQueries');
  const Probe = () => {
    const { run } = useMutationHook();
    return (
      <button type="button" onClick={() => run({ queue_id: 'rq-1', item_ids: ['tr-1'] })}>
        go
      </button>
    );
  };
  render(
    <QueryClientProvider client={queryClient}>
      <Probe />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByText('go'));
  return spy;
};

describe('review-queue item add/remove mutations', () => {
  it('add invalidates both the queue trace list and the queue list (per-trace membership)', async () => {
    const spy = renderMutation(() => {
      const { addItemsToReviewQueueAsync } = useAddItemsToReviewQueueMutation();
      return { run: addItemsToReviewQueueAsync };
    });
    await waitFor(() => {
      expect(spy).toHaveBeenCalledWith([LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY]);
      expect(spy).toHaveBeenCalledWith([LIST_REVIEW_QUEUES_QUERY_KEY]);
    });
  });

  it('remove invalidates both the queue trace list and the queue list (per-trace membership)', async () => {
    const spy = renderMutation(() => {
      const { removeItemsFromReviewQueueAsync } = useRemoveItemsFromReviewQueueMutation();
      return { run: removeItemsFromReviewQueueAsync };
    });
    await waitFor(() => {
      expect(spy).toHaveBeenCalledWith([LIST_REVIEW_QUEUE_ITEMS_QUERY_KEY]);
      expect(spy).toHaveBeenCalledWith([LIST_REVIEW_QUEUES_QUERY_KEY]);
    });
  });
});
