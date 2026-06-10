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
jest.mock('./hooks/useReviewer', () => ({ useReviewer: () => 'default', DEFAULT_REVIEWER: 'default' }));
jest.mock('./hooks/useCanManageReviews', () => ({ useCanManageReviews: () => false }));
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
jest.mock('./hooks/useListReviewQueuesQuery', () => ({
  useListReviewQueuesQuery: () => ({ reviewQueues: [], isLoading: false, error: null }),
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
    mockAddItems.mockReset();
    mockAddItems.mockResolvedValue({});
    mockGetOrCreateUserQueue.mockReset();
    mockGetOrCreateUserQueue.mockResolvedValue({ review_queue: { queue_id: 'rq-default' } });
    jest.spyOn(Utils, 'displayGlobalInfoNotification').mockImplementation(() => {});
  });

  it('routes the traces and shows a confirmation toast when a destination is selected', async () => {
    renderModal();

    // Open the destination picker and select the (no-auth) default queue.
    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Default queue' }));

    // Confirm.
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));

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
});
