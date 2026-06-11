import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { AddToReviewQueueModal } from './AddToReviewQueueModal';
import Utils from '../../../common/utils/Utils';
import { generatePath } from '../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../routes';

jest.mock('../../../account/hooks', () => ({
  useIsAuthAvailable: () => false,
  useCurrentUserIsAdmin: () => false,
  useCurrentUserIsWorkspaceAdmin: () => false,
}));
let mockReviewerResolved = true;
jest.mock('./hooks/useReviewer', () => ({
  useReviewer: () => 'default',
  DEFAULT_REVIEWER: 'default',
  useIsReviewerResolved: () => mockReviewerResolved,
}));
// No MANAGE; `useCanEditReviews` defaults permissive (matching the real hook on
// a no-auth server) so flagging stays enabled, but is overridable to model a
// READ-only user who can't flag.
let mockCanEdit = true;
jest.mock('./hooks/useCanManageReviews', () => ({
  useCanManageReviews: () => false,
  useCanEditReviews: () => mockCanEdit,
}));
jest.mock('./CreateReviewQueueModal', () => ({ CreateReviewQueueModal: () => null }));

const mockAddItems = jest.fn<(...args: any[]) => Promise<any>>();
jest.mock('./hooks/useAddItemsToReviewQueueMutation', () => ({
  useAddItemsToReviewQueueMutation: () => ({
    addItemsToReviewQueueAsync: mockAddItems,
    isAddingItems: false,
    error: null,
    reset: jest.fn(),
  }),
}));

const mockGetOrCreateUserQueue = jest.fn<(...args: any[]) => Promise<any>>();
jest.mock('./hooks/useGetOrCreateUserQueueMutation', () => ({
  useGetOrCreateUserQueueMutation: () => ({
    getOrCreateUserQueueAsync: mockGetOrCreateUserQueue,
    isResolvingUserQueue: false,
    error: null,
    reset: jest.fn(),
  }),
}));

jest.mock('./hooks/useAssignableUsersQuery', () => ({
  useAssignableUsersQuery: () => ({ users: [], isLoading: false }),
}));
// One assignable CUSTOM queue (its schema_id resolves against the experiment
// schema below), so tests can route to it alongside the no-auth default queue.
jest.mock('./hooks/useListReviewQueuesQuery', () => ({
  useListReviewQueuesQuery: () => ({
    reviewQueues: [
      { queue_id: 'rq-custom', queue_type: 'CUSTOM', name: 'Relevance', created_by: 'default', schema_ids: ['s1'] },
    ],
    isLoading: false,
    error: null,
  }),
}));
// One experiment question, so the (no-auth) default queue is assignable.
jest.mock('../../components/label-schemas', () => ({
  useListLabelSchemasQuery: () => ({
    labelSchemas: [{ schema_id: 's1', name: 'Q1', type: 'FEEDBACK', input: { text: {} } }],
    isLoading: false,
  }),
  LabelSchemaFormModal: () => null,
}));

const renderModal = () =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AddToReviewQueueModal
          experimentId="exp-1"
          visible
          setVisible={jest.fn()}
          selectedTraceInfos={[{ trace_id: 'tr-1' } as any]}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('AddToReviewQueueModal', () => {
  beforeEach(() => {
    mockReviewerResolved = true;
    mockCanEdit = true;
    mockAddItems.mockReset();
    mockAddItems.mockResolvedValue({});
    mockGetOrCreateUserQueue.mockReset();
    mockGetOrCreateUserQueue.mockResolvedValue({ review_queue: { queue_id: 'rq-default' } });
    jest
      .spyOn(Utils, 'displayGlobalInfoNotification')
      .mockReset()
      .mockImplementation(() => {});
  });

  it('routes the traces and shows a confirmation toast when a destination is selected', async () => {
    renderModal();

    // Open the destination picker and select the (no-auth) default queue.
    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));

    // Confirm.
    fireEvent.click(screen.getByRole('button', { name: 'Add to 1 queue' }));

    await waitFor(() => expect(mockAddItems).toHaveBeenCalledWith({ queue_id: 'rq-default', item_ids: ['tr-1'] }));
    expect(Utils.displayGlobalInfoNotification).toHaveBeenCalledTimes(1);

    const [toastNode, , style] = jest.mocked(Utils.displayGlobalInfoNotification).mock.calls[0];
    // The toast widens past the default fixed width so the link stays on one line.
    expect(style).toEqual({ width: 'auto' });
    // The toast embeds a Link pointing at the experiment's review-queue page.
    const findLinkTarget = (node: any): string | undefined => {
      if (!node || typeof node !== 'object') return undefined;
      if (node.props?.to) return node.props.to;
      for (const child of React.Children.toArray(node.props?.children)) {
        const target = findLinkTarget(child);
        if (target) return target;
      }
      return undefined;
    };
    expect(findLinkTarget(toastNode)).toBe(
      generatePath(RoutePaths.experimentPageTabReviewQueue, { experimentId: 'exp-1' }),
    );
  });

  it('surfaces a failed destination resolution instead of swallowing it', async () => {
    mockGetOrCreateUserQueue.mockRejectedValue(new Error('Queue resolution failed'));
    renderModal();

    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add to 1 queue' }));

    // The batch failure surfaces in the modal; traces are not added and no
    // success toast fires.
    expect(await screen.findByText('Queue resolution failed')).toBeInTheDocument();
    expect(mockAddItems).not.toHaveBeenCalled();
    expect(Utils.displayGlobalInfoNotification).not.toHaveBeenCalled();
  });

  it('surfaces a failed trace attach instead of swallowing it', async () => {
    // Resolution succeeds; the attach step is the one that rejects.
    mockAddItems.mockRejectedValue(new Error('Attach failed'));
    renderModal();

    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add to 1 queue' }));

    expect(await screen.findByText('Attach failed')).toBeInTheDocument();
    expect(Utils.displayGlobalInfoNotification).not.toHaveBeenCalled();
    // The modal stays open and Add is re-enabled (isSubmitting cleared in
    // `finally`), so the reviewer can retry without reopening.
    expect(screen.getByRole('button', { name: 'Add to 1 queue' })).toBeEnabled();
  });

  it('surfaces a partial attach failure while still issuing the successful attach', async () => {
    // Two destinations: the default queue succeeds, the custom queue rejects.
    mockAddItems.mockImplementation((arg: any) =>
      arg.queue_id === 'rq-custom' ? Promise.reject(new Error('Attach failed for rq-custom')) : Promise.resolve({}),
    );
    renderModal();

    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Relevance' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add to 2 queues' }));

    // The failing destination surfaces an error...
    expect(await screen.findByText('Attach failed for rq-custom')).toBeInTheDocument();
    // ...both attaches were still issued in the same batch (allSettled doesn't
    // abort the successful one when a sibling rejects)...
    expect(mockAddItems).toHaveBeenCalledTimes(2);
    expect(mockAddItems).toHaveBeenCalledWith({ queue_id: 'rq-default', item_ids: ['tr-1'] });
    expect(mockAddItems).toHaveBeenCalledWith({ queue_id: 'rq-custom', item_ids: ['tr-1'] });
    // ...and no success toast fires while any destination failed.
    expect(Utils.displayGlobalInfoNotification).not.toHaveBeenCalled();
  });

  it('aborts the whole add (including selected CUSTOM queues) when a destination resolution fails', async () => {
    // A user-queue resolution failure aborts before any attach, so the
    // independently-selected CUSTOM queue is intentionally NOT attached either.
    mockGetOrCreateUserQueue.mockRejectedValue(new Error('Queue resolution failed'));
    renderModal();

    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Relevance' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add to 2 queues' }));

    expect(await screen.findByText('Queue resolution failed')).toBeInTheDocument();
    // No attach is issued at all — not even for the CUSTOM queue that resolved fine.
    expect(mockAddItems).not.toHaveBeenCalled();
    expect(Utils.displayGlobalInfoNotification).not.toHaveBeenCalled();
  });

  it('keeps Add disabled until the reviewer identity is resolved', () => {
    mockReviewerResolved = false;
    renderModal();

    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));

    // A destination is selected, but the reviewer is still loading, so the write
    // (which stamps created_by) must not be allowed yet.
    expect(screen.getByRole('button', { name: 'Add to 1 queue' })).toBeDisabled();
  });

  it('keeps Add disabled and hides the New-queue link for a READ-only user', () => {
    mockCanEdit = false;
    renderModal();

    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));

    // Flagging traces requires EDIT, so a destination can be picked but not committed.
    expect(screen.getByRole('button', { name: 'Add to 1 queue' })).toBeDisabled();
    // ...and the create-a-queue affordance (also EDIT) is gone from the dropdown.
    expect(screen.queryByText('New queue')).toBeNull();
  });
});
