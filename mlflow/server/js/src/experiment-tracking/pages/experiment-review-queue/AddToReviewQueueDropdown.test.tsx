import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { AddToReviewQueueDropdown } from './AddToReviewQueueDropdown';
import Utils from '../../../common/utils/Utils';
import { generatePath } from '../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../routes';

let mockAuthAvailable = false;
jest.mock('../../../account/hooks', () => ({
  useIsAuthAvailable: () => mockAuthAvailable,
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
let mockCanManage = false;
jest.mock('./hooks/useCanManageReviews', () => ({
  useCanManageReviews: () => mockCanManage,
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

let mockUsersError: Error | null = null;
jest.mock('./hooks/useAssignableUsersQuery', () => ({
  useAssignableUsersQuery: () => ({ users: [], isLoading: false, error: mockUsersError }),
}));
// The experiment's queues (the unfiltered call). One assignable CUSTOM queue
// (its schema_id resolves against the experiment schema below), so tests can
// route to it alongside the no-auth default queue.
let mockListQueues: any[] = [
  { queue_id: 'rq-custom', queue_type: 'CUSTOM', name: 'Relevance', created_by: 'default', schema_ids: ['s1'] },
];
// Queues the single selected trace is already a member of (the `itemId`-scoped
// call). Default: none, loaded. Tests override to exercise the membership behavior.
let mockMemberQueues: any[] = [];
let mockMembersLoading = false;
jest.mock('./hooks/useListReviewQueuesQuery', () => ({
  useListReviewQueuesQuery: ({ itemId }: { itemId?: string }) =>
    itemId
      ? { reviewQueues: mockMemberQueues, isLoading: mockMembersLoading, error: null }
      : {
          reviewQueues: mockListQueues,
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
    mockAuthAvailable = false;
    mockCanEdit = true;
    mockCanManage = false;
    mockUsersError = null;
    mockMemberQueues = [];
    mockMembersLoading = false;
    mockListQueues = [
      { queue_id: 'rq-custom', queue_type: 'CUSTOM', name: 'Relevance', created_by: 'default', schema_ids: ['s1'] },
    ];
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
    jest
      .spyOn(Utils, 'displayGlobalErrorNotification')
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

  it('surfaces a failed user-roster load in the Users section instead of an empty state', () => {
    // On an auth server the Users section lists assignable users; a failed load
    // must show the error rather than the "Search by name" prompt (which would
    // otherwise mask the failure until the reviewer types).
    mockAuthAvailable = true;
    mockUsersError = new Error('boom');
    renderDropdown({ open: true });
    expect(screen.getByText(/couldn't load users/i)).toBeInTheDocument();
    expect(screen.queryByText(/search by name to add/i)).not.toBeInTheDocument();
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
    // The toast deep-links to the queue the traces were just added to.
    expect(findLinkTarget(toastNode)).toBe(
      `${generatePath(RoutePaths.experimentPageTabReviewQueue, { experimentId: 'exp-1' })}?selectedQueueId=rq-default`,
    );
  });

  it('surfaces a failed destination resolution instead of swallowing it', async () => {
    mockGetOrCreateUserQueue.mockRejectedValue(new Error('Queue resolution failed'));
    renderDropdown({ open: true });

    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));

    await waitFor(() => expect(Utils.displayGlobalErrorNotification).toHaveBeenCalledTimes(1));
    expect(jest.mocked(Utils.displayGlobalErrorNotification).mock.calls[0][0]).toContain('Queue resolution failed');
    expect(mockAddItems).not.toHaveBeenCalled();
    expect(Utils.displayGlobalInfoNotification).not.toHaveBeenCalled();
  });

  it('surfaces a failed trace attach instead of swallowing it', async () => {
    mockAddItems.mockRejectedValue(new Error('Attach failed'));
    renderDropdown({ open: true });

    fireEvent.click(screen.getByRole('checkbox', { name: 'Relevance' }));

    await waitFor(() => expect(Utils.displayGlobalErrorNotification).toHaveBeenCalledTimes(1));
    expect(jest.mocked(Utils.displayGlobalErrorNotification).mock.calls[0][0]).toContain('Attach failed');
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

  it('lets a permitted reviewer uncheck a queue the trace is in to remove it', async () => {
    // reviewer 'default' owns rq-custom, so they may remove from it: the seeded
    // membership renders checked + enabled, and unchecking removes the trace.
    mockMemberQueues = [
      { queue_id: 'rq-custom', queue_type: 'CUSTOM', name: 'Relevance', created_by: 'default', schema_ids: ['s1'] },
    ];
    renderDropdown({ open: true });

    const option = await screen.findByRole('checkbox', { name: 'Relevance' });
    await waitFor(() => expect(option).toBeChecked());
    expect(option).not.toBeDisabled();

    fireEvent.click(screen.getByRole('checkbox', { name: 'Relevance' }));
    await waitFor(() => expect(mockRemoveItems).toHaveBeenCalledWith({ queue_id: 'rq-custom', item_ids: ['tr-1'] }));
    expect(mockAddItems).not.toHaveBeenCalled();
  });

  it('locks a queue the trace is in when the reviewer cannot remove from it', async () => {
    // The queue is owned by someone else and the reviewer is not a manager, so
    // they can't remove from it: it renders checked but disabled (locked). Its
    // accessible name carries the disabled reason, hence the regex match.
    const queue = {
      queue_id: 'rq-custom',
      queue_type: 'CUSTOM',
      name: 'Relevance',
      created_by: 'someone-else',
      schema_ids: ['s1'],
    };
    mockListQueues = [queue];
    mockMemberQueues = [queue];
    renderDropdown({ open: true });

    const option = await screen.findByRole('checkbox', { name: /Relevance/ });
    expect(option).toBeChecked();
    expect(option).toBeDisabled();
    expect(mockRemoveItems).not.toHaveBeenCalled();
  });

  it('pre-checks the default queue when the trace is already in it', async () => {
    // The default queue is a USER queue ('default'); a USER membership seeds the
    // pinned default option as checked (and on no-auth it stays removable).
    mockMemberQueues = [
      { queue_id: 'rq-default', queue_type: 'USER', name: 'default', created_by: 'default', schema_ids: [] },
    ];
    renderDropdown({ open: true });

    await waitFor(() => expect(screen.getByRole('checkbox', { name: 'Default queue' })).toBeChecked());
  });

  it("locks a user's queue the trace is in unless the reviewer is a manager", async () => {
    // On an auth server, a per-user queue membership shows that user checked.
    // A personal queue is prunable only by a manager, so a non-manager sees it
    // locked (its accessible name carries the disabled reason).
    mockAuthAvailable = true;
    mockCanManage = false;
    mockMemberQueues = [
      { queue_id: 'rq-alice', queue_type: 'USER', name: 'alice', created_by: 'alice', schema_ids: [] },
    ];
    renderDropdown({ open: true });

    const option = await screen.findByRole('checkbox', { name: /alice/ });
    expect(option).toBeChecked();
    expect(option).toBeDisabled();
    expect(mockRemoveItems).not.toHaveBeenCalled();
  });

  it("disables queue options until a single trace's memberships finish loading", () => {
    // While membership is unknown we can't tell which rows are checked/locked,
    // so toggling must be blocked to avoid adding to a queue meant for removal.
    mockMembersLoading = true;
    renderDropdown({ open: true });

    expect(screen.getByRole('checkbox', { name: 'Relevance' })).toBeDisabled();
  });

  it('filters checked member rows by the search query', async () => {
    mockAuthAvailable = true;
    mockMemberQueues = [
      { queue_id: 'rq-alice', queue_type: 'USER', name: 'alice', created_by: 'alice', schema_ids: [] },
    ];
    renderDropdown({ open: true });

    // The membership shows by default...
    expect(await screen.findByRole('checkbox', { name: /alice/ })).toBeInTheDocument();

    // ...but a non-matching search hides it (the filter must apply to members too).
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'zzz' } });
    expect(screen.queryByRole('checkbox', { name: /alice/ })).not.toBeInTheDocument();

    // A matching search brings it back.
    fireEvent.change(screen.getByPlaceholderText(/search/i), { target: { value: 'ali' } });
    expect(screen.getByRole('checkbox', { name: /alice/ })).toBeInTheDocument();
  });
});
