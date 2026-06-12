import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { AddToReviewQueueDropdown } from './AddToReviewQueueDropdown';
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

const mockRemoveItems = jest.fn<(...args: any[]) => Promise<any>>();
jest.mock('./hooks/useRemoveItemsFromReviewQueueMutation', () => ({
  useRemoveItemsFromReviewQueueMutation: () => ({
    removeItemsFromReviewQueueAsync: mockRemoveItems,
    isRemovingItems: false,
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
// Queues the single selected trace is already a member of (the `itemId`-scoped
// call). Default: none. Tests override to exercise the membership pre-check.
let mockMemberQueues: any[] = [];
jest.mock('./hooks/useListReviewQueuesQuery', () => ({
  useListReviewQueuesQuery: ({ itemId }: { itemId?: string }) =>
    itemId
      ? { reviewQueues: mockMemberQueues, isLoading: false, error: null }
      : {
          reviewQueues: [
            {
              queue_id: 'rq-custom',
              queue_type: 'CUSTOM',
              name: 'Relevance',
              created_by: 'default',
              schema_ids: ['s1'],
            },
          ],
          isLoading: false,
          error: null,
        },
}));
// One experiment question, so the (no-auth) default queue is assignable.
jest.mock('../../components/label-schemas', () => ({
  useListLabelSchemasQuery: () => ({
    labelSchemas: [{ schema_id: 's1', name: 'Q1', type: 'FEEDBACK', input: { text: {} } }],
    isLoading: false,
  }),
  LabelSchemaFormModal: () => null,
}));

const renderDropdown = (props?: { open?: boolean; onOpenChange?: (open: boolean) => void }) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AddToReviewQueueDropdown
          experimentId="exp-1"
          selectedTraceInfos={[{ trace_id: 'tr-1' } as any]}
          open={props?.open}
          onOpenChange={props?.onOpenChange}
        >
          <button type="button">Trigger</button>
        </AddToReviewQueueDropdown>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('AddToReviewQueueDropdown', () => {
  beforeEach(() => {
    mockReviewerResolved = true;
    mockCanEdit = true;
    mockMemberQueues = [];
    mockAddItems.mockReset();
    mockAddItems.mockResolvedValue({});
    mockRemoveItems.mockReset();
    mockRemoveItems.mockResolvedValue({});
    mockGetOrCreateUserQueue.mockReset();
    mockGetOrCreateUserQueue.mockResolvedValue({ review_queue: { queue_id: 'rq-default' } });
    jest
      .spyOn(Utils, 'displayGlobalInfoNotification')
      .mockReset()
      .mockImplementation(() => {});
  });

  it('opens the dropdown when controlled with open=true', () => {
    renderDropdown({ open: true });
    expect(screen.getByText('Default queue')).toBeInTheDocument();
  });

  it('opens the dropdown when clicking the trigger (uncontrolled)', () => {
    renderDropdown();
    expect(screen.queryByText('Default queue')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Trigger'));
    expect(screen.getByText('Default queue')).toBeInTheDocument();
  });

  it('immediately adds traces when a queue is clicked and shows a toast', async () => {
    renderDropdown({ open: true });

    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));

    await waitFor(() => expect(mockAddItems).toHaveBeenCalledWith({ queue_id: 'rq-default', item_ids: ['tr-1'] }));
    expect(Utils.displayGlobalInfoNotification).toHaveBeenCalledTimes(1);

    const [toastNode, duration] = jest.mocked(Utils.displayGlobalInfoNotification).mock.calls[0];
    expect(duration).toBe(3);
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
    renderDropdown({ open: true });

    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));

    expect(await screen.findByText('Queue resolution failed')).toBeInTheDocument();
    expect(mockAddItems).not.toHaveBeenCalled();
    expect(Utils.displayGlobalInfoNotification).not.toHaveBeenCalled();
  });

  it('surfaces a failed trace attach instead of swallowing it', async () => {
    mockAddItems.mockRejectedValue(new Error('Attach failed'));
    renderDropdown({ open: true });

    fireEvent.click(screen.getByRole('checkbox', { name: 'Relevance' }));

    expect(await screen.findByText('Attach failed')).toBeInTheDocument();
    expect(Utils.displayGlobalInfoNotification).not.toHaveBeenCalled();
  });

  it('un-checking a queue removes the traces from it', async () => {
    renderDropdown({ open: true });

    // First click adds.
    fireEvent.click(screen.getByRole('checkbox', { name: 'Relevance' }));
    await waitFor(() => expect(mockAddItems).toHaveBeenCalledWith({ queue_id: 'rq-custom', item_ids: ['tr-1'] }));
    // Wait for the checked state to settle before clicking again.
    await waitFor(() => expect(screen.getByRole('checkbox', { name: 'Relevance' })).toBeChecked());

    // Second click removes.
    fireEvent.click(screen.getByRole('checkbox', { name: 'Relevance' }));
    await waitFor(() => expect(mockRemoveItems).toHaveBeenCalledWith({ queue_id: 'rq-custom', item_ids: ['tr-1'] }));
  });

  it('allows adding to multiple queues sequentially', async () => {
    renderDropdown({ open: true });

    // Add to a custom queue.
    fireEvent.click(screen.getByRole('checkbox', { name: 'Relevance' }));
    await waitFor(() => expect(mockAddItems).toHaveBeenCalledWith({ queue_id: 'rq-custom', item_ids: ['tr-1'] }));

    // Then add to the default queue.
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));
    await waitFor(() => expect(mockAddItems).toHaveBeenCalledWith({ queue_id: 'rq-default', item_ids: ['tr-1'] }));

    expect(Utils.displayGlobalInfoNotification).toHaveBeenCalledTimes(2);
  });

  it('hides the New-queue link for a READ-only user', () => {
    mockCanEdit = false;
    renderDropdown({ open: true });

    // The create-a-queue affordance (requires EDIT) is gone from the dropdown.
    expect(screen.queryByText('New queue')).toBeNull();
  });

  it('shows the queues a single trace is already in as checked but locked', async () => {
    // The trace already belongs to the custom queue, so its option is rendered
    // checked and disabled — neither selectable nor unselectable here. Its
    // accessible name carries the disabled reason, hence the regex match.
    mockMemberQueues = [
      { queue_id: 'rq-custom', queue_type: 'CUSTOM', name: 'Relevance', created_by: 'default', schema_ids: ['s1'] },
    ];
    renderDropdown({ open: true });

    const option = await screen.findByRole('checkbox', { name: /Relevance/ });
    expect(option).toBeChecked();
    expect(option).toBeDisabled();
    // It's a read-only indicator, not an add.
    expect(mockAddItems).not.toHaveBeenCalled();
  });
});
