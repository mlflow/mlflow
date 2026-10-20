import { describe, jest, it, expect, beforeEach, afterEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import type { ModelTraceExplorerDisplayMode } from '@databricks/web-shared/model-trace-explorer';

import ExperimentReviewQueuePage from './ExperimentReviewQueuePage';
import type { ReviewQueue, ReviewQueueItem } from './types';

const queue: ReviewQueue = {
  queue_id: 'rq-1',
  experiment_id: 'exp-1',
  name: 'Team queue',
  queue_type: 'CUSTOM',
  creation_time_ms: 1,
  last_update_time_ms: 1,
  users: ['alice'],
};
const makeItem = (itemId: string): ReviewQueueItem => ({
  queue_id: 'rq-1',
  item_type: 'TRACE',
  item_id: itemId,
  status: 'PENDING',
  creation_time_ms: 1,
  last_update_time_ms: 1,
});

jest.mock('./hooks/useListReviewQueuesQuery', () => ({
  useListReviewQueuesQuery: () => ({ reviewQueues: [queue], isLoading: false }),
}));
jest.mock('./hooks/useListReviewQueueItemsQuery', () => ({
  useListReviewQueueItemsQuery: () => ({ items: [makeItem('tr-1'), makeItem('tr-2')], isLoading: false }),
}));
jest.mock('../../components/label-schemas', () => ({
  ...jest.requireActual<typeof import('../../components/label-schemas')>('../../components/label-schemas'),
  useListLabelSchemasQuery: () => ({ labelSchemas: [] }),
}));
jest.mock('./hooks/useSetReviewQueueItemStatusMutation', () => ({
  useSetReviewQueueItemStatusMutation: () => ({ setReviewQueueItemStatusAsync: jest.fn(), isSettingStatus: false }),
}));
jest.mock('./hooks/useRemoveItemsFromReviewQueueMutation', () => ({
  useRemoveItemsFromReviewQueueMutation: () => ({ removeItemsFromReviewQueueAsync: jest.fn(), isRemovingItems: false }),
}));
jest.mock('./hooks/useDeleteReviewQueueMutation', () => ({
  useDeleteReviewQueueMutation: () => ({ deleteReviewQueueAsync: jest.fn(), isDeletingQueue: false }),
}));
jest.mock('./hooks/useUpdateReviewQueueMutation', () => ({
  useUpdateReviewQueueMutation: () => ({ updateReviewQueueAsync: jest.fn(), isUpdatingQueue: false }),
}));
jest.mock('./hooks/useGetOrCreateUserQueueMutation', () => ({
  useGetOrCreateUserQueueMutation: () => ({ getOrCreateUserQueueAsync: jest.fn(() => Promise.resolve()) }),
}));
let mockCanEdit = true;
jest.mock('./hooks/useCanManageReviews', () => ({
  useCanManageReviews: () => false,
  useCanEditReviews: () => mockCanEdit,
}));
jest.mock('../../../account/hooks', () => ({
  ...jest.requireActual<typeof import('../../../account/hooks')>('../../../account/hooks'),
  useIsAuthAvailable: () => true,
}));
jest.mock('./hooks/useReviewer', () => ({
  ...jest.requireActual<typeof import('./hooks/useReviewer')>('./hooks/useReviewer'),
  useReviewer: () => 'alice',
  useIsReviewerResolved: () => true,
}));
jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/model-trace-explorer')>(
    '@databricks/web-shared/model-trace-explorer',
  ),
  useGetTracesById: () => ({ data: [], isLoading: false }),
}));
// The sidebar has its own tests; it is stubbed so this test only needs the page's own data.
jest.mock('./ReviewQueueSidebar', () => ({ ReviewQueueSidebar: () => null }));
// The provider's loading and read-only behavior is covered by ExperimentCustomViewProvider.test.tsx;
// the stub shows whether the page mounted it, and for which experiment.
const mockProviderMounts = jest.fn();
jest.mock('../../components/experiment-page/components/traces-v3/ExperimentCustomViewDefinitionProvider', () => ({
  ExperimentCustomViewDefinitionProvider: ({ experimentId, children }: { experimentId?: string; children: any }) => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports, global-require
    const { useEffect } = require('react');
    useEffect(() => mockProviderMounts(), []);
    return (
      <div data-testid="custom-view-definitions" data-experiment-id={experimentId}>
        {children}
      </div>
    );
  },
}));
// FocusedReview's own view switching is covered by FocusedReview.customView.test.tsx; the stub exposes the
// props the page hands it so this test checks the page's wiring.
jest.mock('./FocusedReview', () => ({
  FocusedReview: ({
    item,
    traceViewDisplayMode,
    onTraceViewDisplayModeChange,
    onCreateCustomView,
    onSelect,
    onBack,
  }: {
    item: ReviewQueueItem;
    traceViewDisplayMode?: ModelTraceExplorerDisplayMode;
    onTraceViewDisplayModeChange?: (mode: ModelTraceExplorerDisplayMode) => void;
    onCreateCustomView?: () => void;
    onSelect: (itemId: string) => void;
    onBack: () => void;
  }) => (
    <div>
      <div>{`focused ${item.item_id} in ${traceViewDisplayMode} mode`}</div>
      <button onClick={() => onTraceViewDisplayModeChange?.('custom')}>Choose saved view</button>
      <button onClick={() => onSelect('tr-2')}>Go to next item</button>
      <button onClick={onBack}>Back to list</button>
      {onCreateCustomView && <button onClick={onCreateCustomView}>Create custom view</button>}
    </div>
  ),
}));

const renderPage = (search = '?selectedQueueId=rq-1&selectedItemId=tr-1') =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MemoryRouter initialEntries={[`/experiments/exp-1/review-queues${search}`]}>
          <Routes>
            <Route path="/experiments/:experimentId/*" element={<ExperimentReviewQueuePage />} />
          </Routes>
        </MemoryRouter>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ExperimentReviewQueuePage saved custom views', () => {
  const windowOpen = jest.spyOn(window, 'open').mockImplementation(() => null);

  beforeEach(() => {
    mockCanEdit = true;
    windowOpen.mockClear();
    mockProviderMounts.mockClear();
  });

  afterEach(() => {
    windowOpen.mockClear();
  });

  it('mounts the experiment saved views and keeps the selected mode across item navigation', async () => {
    renderPage();

    expect(await screen.findByTestId('custom-view-definitions')).toHaveAttribute('data-experiment-id', 'exp-1');
    expect(screen.getByText('focused tr-1 in default mode')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Choose saved view' }));
    expect(screen.getByText('focused tr-1 in custom mode')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Go to next item' }));
    expect(await screen.findByText('focused tr-2 in custom mode')).toBeInTheDocument();
    // The provider (and the chosen saved view it holds) is not remounted by item navigation.
    expect(mockProviderMounts).toHaveBeenCalledTimes(1);
  });

  it('does not load saved views for the queue list', async () => {
    renderPage('?selectedQueueId=rq-1');

    expect(await screen.findByText('Team queue')).toBeInTheDocument();
    expect(screen.queryByTestId('custom-view-definitions')).not.toBeInTheDocument();
  });

  it('starts the next focused review on Default view after going back to the list', async () => {
    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'Choose saved view' }));
    expect(screen.getByText('focused tr-1 in custom mode')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Back to list' }));
    expect(screen.queryByTestId('custom-view-definitions')).not.toBeInTheDocument();

    await userEvent.click(screen.getAllByRole('row')[1]);
    expect(await screen.findByText('focused tr-1 in default mode')).toBeInTheDocument();
  });

  it('opens custom-view creation for the trace in a new Traces tab for an experiment editor', async () => {
    renderPage('?selectedQueueId=rq-1&selectedItemId=tr-1&startReview=true&spanId=s-1&workspace=team-ws');

    await userEvent.click(await screen.findByRole('button', { name: 'Create custom view' }));

    expect(windowOpen).toHaveBeenCalledTimes(1);
    const [url, target, features] = windowOpen.mock.calls[0];
    expect(target).toBe('_blank');
    expect(features).toBe('noopener,noreferrer');
    const [path, query] = String(url).split('?');
    expect(path).toBe('/#/experiments/exp-1/traces');
    const params = new URLSearchParams(query);
    expect(params.get('selectedEvaluationId')).toBe('tr-1');
    expect(params.get('createCustomView')).toBe('true');
    expect(params.get('workspace')).toBe('team-ws');
    expect(params.has('selectedQueueId')).toBe(false);
    expect(params.has('selectedItemId')).toBe(false);
    expect(params.has('startReview')).toBe(false);
    expect(params.has('spanId')).toBe(false);
  });

  it('does not offer custom-view creation to a user without experiment EDIT', async () => {
    mockCanEdit = false;
    renderPage();

    expect(await screen.findByText('focused tr-1 in default mode')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Create custom view' })).not.toBeInTheDocument();
  });
});
