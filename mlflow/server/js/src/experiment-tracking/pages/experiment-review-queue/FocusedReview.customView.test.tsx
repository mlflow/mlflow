import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React, { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import type { ModelTraceExplorerDisplayMode } from '@databricks/web-shared/model-trace-explorer';
import {
  CustomViewDefinitionProvider,
  MAX_CUSTOM_VIEWS_PER_EXPERIMENT,
  type CustomView,
} from '@databricks/web-shared/model-trace-explorer/custom-view';

import { FocusedReview } from './FocusedReview';
import type { ReviewQueueItem } from './types';

let mockTraceResult: { data: unknown[]; isLoading: boolean } = { data: [], isLoading: false };
jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/model-trace-explorer')>(
    '@databricks/web-shared/model-trace-explorer',
  ),
  useGetTracesById: () => mockTraceResult,
  ModelTraceExplorer: () => <div>Full trace explorer</div>,
}));

jest.mock('@databricks/web-shared/model-trace-explorer/custom-view', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/model-trace-explorer/custom-view')>(
    '@databricks/web-shared/model-trace-explorer/custom-view',
  ),
  // The shared renderer has its own coverage (ModelTraceExplorerCustomView.test.tsx and
  // ReviewQueueCustomViewContent.test.tsx); this test isolates the review panel's view switching.
  ModelTraceExplorerCustomViewRenderer: ({ enableFeedback }: { enableFeedback?: boolean }) => (
    <div>{enableFeedback ? 'Rendered saved custom view' : 'Rendered read-only saved custom view'}</div>
  ),
}));

jest.mock('./hooks/useCreateReviewAssessmentMutation', () => ({
  useCreateReviewAssessmentMutation: () => ({ createReviewAssessmentAsync: jest.fn(), isCreatingAssessment: false }),
}));
jest.mock('./hooks/useTraceAssessmentsQuery', () => ({
  useTraceAssessmentsQuery: () => ({ priorAnswers: [], isLoading: false, isFetching: false }),
}));

const makeItem = (itemId: string): ReviewQueueItem => ({
  queue_id: 'rq-1',
  item_type: 'TRACE',
  item_id: itemId,
  status: 'PENDING',
  creation_time_ms: 1_780_000_000_000,
  last_update_time_ms: 1_780_000_000_000,
});
const firstItem = makeItem('tr-1');
const secondItem = makeItem('tr-2');

const makeTrace = (traceId: string) => ({
  info: { trace_id: traceId, request_preview: `input of ${traceId}`, response_preview: `output of ${traceId}` },
  data: { spans: [] },
});

const savedView: CustomView = {
  id: 'saved-view-1',
  name: 'Reviewer view',
  label: 'Reviewer view',
  instruction: '',
  template: [
    {
      version: 'v0.9',
      updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'hi' }] },
    },
  ],
  createdAtMs: 1,
};
const maxSavedViews = Array.from({ length: MAX_CUSTOM_VIEWS_PER_EXPERIMENT }, (_unused, index) => ({
  ...savedView,
  id: `saved-view-${index}`,
  name: `Reviewer view ${index}`,
}));

const FocusedReviewHost = ({
  views,
  canReview,
  initialDisplayMode,
  onCreateCustomView,
}: {
  views: CustomView[];
  canReview: boolean;
  initialDisplayMode: ModelTraceExplorerDisplayMode;
  onCreateCustomView?: () => void;
}) => {
  const [displayMode, setDisplayMode] = useState<ModelTraceExplorerDisplayMode>(initialDisplayMode);
  const [selectedItemId, setSelectedItemId] = useState(firstItem.item_id);
  const item = selectedItemId === secondItem.item_id ? secondItem : firstItem;
  return (
    <CustomViewDefinitionProvider views={views} isLoaded>
      <FocusedReview
        // Remount per item like the page does, so the host-owned mode is what persists.
        key={item.item_id}
        item={item}
        items={[firstItem, secondItem]}
        schemas={[]}
        completedBy="tester"
        isSettingStatus={false}
        canReview={canReview}
        traceViewDisplayMode={displayMode}
        onTraceViewDisplayModeChange={setDisplayMode}
        onCreateCustomView={onCreateCustomView}
        onBack={jest.fn()}
        onSelect={setSelectedItemId}
        onSetStatus={() => Promise.resolve()}
      />
    </CustomViewDefinitionProvider>
  );
};

const renderFocusedReview = ({
  views = [],
  canReview = true,
  initialDisplayMode = 'default',
  onCreateCustomView,
}: {
  views?: CustomView[];
  canReview?: boolean;
  initialDisplayMode?: ModelTraceExplorerDisplayMode;
  onCreateCustomView?: () => void;
} = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <FocusedReviewHost
          views={views}
          canReview={canReview}
          initialDisplayMode={initialDisplayMode}
          onCreateCustomView={onCreateCustomView}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

const openPicker = async (currentLabel: RegExp = /Default view/) =>
  userEvent.click(await screen.findByRole('button', { name: currentLabel }));

describe('FocusedReview saved custom views', () => {
  beforeEach(() => {
    mockTraceResult = { data: [makeTrace('tr-1')], isLoading: false };
  });

  it('keeps the input/output view and a Default-only picker when no saved views are available', async () => {
    renderFocusedReview();

    expect(screen.getByText('input of tr-1')).toBeInTheDocument();
    await openPicker();
    expect(screen.getByRole('menuitemradio', { name: 'Default view' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitemradio', { name: 'Reviewer view' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Create custom view' })).not.toBeInTheDocument();
    expect(screen.getByText('View full trace')).toBeInTheDocument();
  });

  it('switches between Default view and a saved view without exposing creation', async () => {
    renderFocusedReview({ views: [savedView] });

    await openPicker();
    expect(screen.queryByRole('menuitem', { name: 'Create custom view' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Reviewer view' }));

    expect(await screen.findByText('Rendered saved custom view')).toBeInTheDocument();
    expect(screen.queryByText('input of tr-1')).not.toBeInTheDocument();
    expect(screen.getByText('View full trace')).toBeInTheDocument();

    await openPicker(/Reviewer view/);
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Default view' }));
    expect(await screen.findByText('input of tr-1')).toBeInTheDocument();
  });

  it('keeps the chosen saved view when moving to the next trace', async () => {
    mockTraceResult = { data: [makeTrace('tr-1')], isLoading: false };
    renderFocusedReview({ views: [savedView] });

    await openPicker();
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Reviewer view' }));
    expect(await screen.findByText('Rendered saved custom view')).toBeInTheDocument();

    mockTraceResult = { data: [makeTrace('tr-2')], isLoading: false };
    await userEvent.click(screen.getByRole('button', { name: 'Next trace' }));

    expect(await screen.findByText('Rendered saved custom view')).toBeInTheDocument();
    expect(screen.queryByText('input of tr-2')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Reviewer view/ })).toBeInTheDocument();
  });

  it('can return to Default view when custom mode has no saved views', async () => {
    renderFocusedReview({ initialDisplayMode: 'custom' });

    await openPicker(/Custom view/);
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Default view' }));

    expect(await screen.findByText('input of tr-1')).toBeInTheDocument();
  });

  it('shows an unavailable state when a saved view cannot load the full trace', async () => {
    mockTraceResult = { data: [], isLoading: false };
    renderFocusedReview({ views: [savedView] });

    await openPicker();
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Reviewer view' }));

    expect(await screen.findByText('The full trace could not be loaded for this custom view.')).toBeInTheDocument();
  });

  it('renders saved custom views without feedback writes when the reviewer cannot review', async () => {
    renderFocusedReview({ views: [savedView], canReview: false });

    await openPicker();
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Reviewer view' }));

    expect(await screen.findByText('Rendered read-only saved custom view')).toBeInTheDocument();
  });

  it('offers custom-view creation with no saved views when the host provides the action', async () => {
    const onCreateCustomView = jest.fn();
    renderFocusedReview({ onCreateCustomView });

    expect(screen.getByText('input of tr-1')).toBeInTheDocument();
    await openPicker();
    expect(screen.getByRole('menuitemradio', { name: 'Default view' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('menuitem', { name: 'Create custom view' }));

    expect(onCreateCustomView).toHaveBeenCalledTimes(1);
  });

  it('offers custom-view creation when the host provides the action', async () => {
    const onCreateCustomView = jest.fn();
    renderFocusedReview({ views: [savedView], onCreateCustomView });

    await openPicker();
    await userEvent.click(screen.getByRole('menuitem', { name: 'Create custom view' }));

    expect(onCreateCustomView).toHaveBeenCalledTimes(1);
  });

  it('hides custom-view creation when the trace fails to load', async () => {
    mockTraceResult = { data: [], isLoading: false };
    const onCreateCustomView = jest.fn();
    renderFocusedReview({ views: [savedView], onCreateCustomView });

    expect(await screen.findByText('No input or output recorded for this trace.')).toBeInTheDocument();
    await openPicker();

    expect(screen.getByRole('menuitemradio', { name: 'Reviewer view' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Create custom view' })).not.toBeInTheDocument();
  });

  it('hides custom-view creation when the experiment is at the saved-view limit', async () => {
    renderFocusedReview({ views: maxSavedViews, onCreateCustomView: jest.fn() });

    await openPicker();

    expect(screen.queryByRole('menuitem', { name: 'Create custom view' })).not.toBeInTheDocument();
  });
});
