import { describe, it, expect, jest } from '@jest/globals';
import { act, screen } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { MOCK_TRACE } from '../../ModelTraceExplorer.test-utils';
import type { ModelTrace } from '../ModelTrace.types';
import {
  ModelTraceExplorerPreferencesProvider,
  useModelTraceExplorerPreferences,
} from '../ModelTraceExplorerPreferencesContext';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from '../ModelTraceExplorerViewStateContext';
import { TEST_SPAN_FILTER_STATE } from './TimelineTree.test-utils';
import { TimelineTreeHeader } from './TimelineTreeHeader';

interface TestWrapperProps {
  refreshTrace?: () => Promise<void>;
  showGraph?: boolean;
  onToggleGraph?: () => void;
  useClippingPopupContainer?: boolean;
}

const MetricSelectionReader = () => {
  const { timelineTreeMetrics } = useModelTraceExplorerPreferences();
  return <output aria-label="Selected timeline metrics">{timelineTreeMetrics.join(',')}</output>;
};

const TestWrapper = ({ refreshTrace, showGraph, onToggleGraph, useClippingPopupContainer }: TestWrapperProps) => {
  const [showTimelineInfo, setShowTimelineInfo] = useState<boolean>(false);
  const [spanFilterState, setSpanFilterState] = useState(TEST_SPAN_FILTER_STATE);
  const [isRefreshingTrace, setIsRefreshingTrace] = useState(false);
  const [popupContainer, setPopupContainer] = useState<HTMLDivElement | null>(null);

  const handleRefresh = async () => {
    if (!refreshTrace) return;
    setIsRefreshingTrace(true);
    try {
      await refreshTrace();
    } finally {
      setIsRefreshingTrace(false);
    }
  };

  return (
    <IntlProvider locale="en">
      <DesignSystemProvider
        getPopupContainer={useClippingPopupContainer ? () => popupContainer ?? document.body : undefined}
      >
        <div data-testid="clipping-popup-container" ref={setPopupContainer} css={{ overflow: 'hidden' }} />
        <ModelTraceExplorerPreferencesProvider>
          <ModelTraceExplorerViewStateProvider
            modelTrace={MOCK_TRACE}
            assessmentsPaneEnabled={false}
            refreshTrace={refreshTrace ? handleRefresh : undefined}
            isRefreshingTrace={isRefreshingTrace}
          >
            <TimelineTreeHeader
              showTimelineInfo={showTimelineInfo}
              setShowTimelineInfo={setShowTimelineInfo}
              spanFilterState={spanFilterState}
              setSpanFilterState={setSpanFilterState}
              showGraph={showGraph}
              onToggleGraph={onToggleGraph}
            />
            <MetricSelectionReader />
            <span>{String(showTimelineInfo)}</span>
          </ModelTraceExplorerViewStateProvider>
        </ModelTraceExplorerPreferencesProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('TimelineTreeHeader', () => {
  it('should switch the timeline tree view state', async () => {
    render(<TestWrapper />);

    expect(screen.getByText('false')).toBeInTheDocument();

    const showTimelineButton = screen.getByRole('button', { name: 'Show execution timeline' });
    await userEvent.click(showTimelineButton);
    expect(await screen.findByText('true')).toBeInTheDocument();

    await userEvent.click(showTimelineButton);
    expect(await screen.findByText('false')).toBeInTheDocument();
  });

  it('keeps the graph toggle inside the settings menu', async () => {
    const onToggleGraph = jest.fn();
    render(<TestWrapper showGraph={false} onToggleGraph={onToggleGraph} />);

    const timelineButton = screen.getByRole('button', { name: 'Show execution timeline' });
    const settingsButton = screen.getByRole('button', { name: 'Trace display settings' });
    expect(timelineButton.compareDocumentPosition(settingsButton)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(screen.queryByRole('button', { name: 'Show graph' })).not.toBeInTheDocument();

    await userEvent.click(settingsButton);
    const graphToggle = await screen.findByRole('menuitemcheckbox', { name: 'Show graph' });
    expect(graphToggle).not.toBeChecked();
    await userEvent.click(graphToggle);
    expect(onToggleGraph).toHaveBeenCalledTimes(1);
  });

  it('shows only the supported metric options with their default selection', async () => {
    render(<TestWrapper />);

    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await userEvent.hover(screen.getByRole('menuitem', { name: 'Display metric types' }));

    expect(await screen.findByRole('menuitemcheckbox', { name: 'Duration' })).toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'Total tokens' })).toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'Estimated LLM cost' })).toBeChecked();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Status' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Time to first token' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Cache hit rate' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Refresh trace' })).not.toBeInTheDocument();
  });

  it('portals the settings menu outside a clipping drawer container', async () => {
    render(<TestWrapper useClippingPopupContainer />);

    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));

    const settingsMenu = await screen.findByRole('menu');
    expect(screen.getByTestId('clipping-popup-container')).not.toContainElement(settingsMenu);
    expect(document.body).toContainElement(settingsMenu);
  });

  it('updates a metric selection while preserving the defaults', async () => {
    render(<TestWrapper />);

    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await userEvent.hover(screen.getByRole('menuitem', { name: 'Display metric types' }));
    const costMetric = await screen.findByRole('menuitemcheckbox', { name: 'Estimated LLM cost' });
    await userEvent.click(costMetric);

    expect(costMetric).not.toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'Duration' })).toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'Total tokens' })).toBeChecked();
  });

  it('preserves canonical metric order when a metric is re-enabled', async () => {
    render(<TestWrapper />);

    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await userEvent.hover(screen.getByRole('menuitem', { name: 'Display metric types' }));
    const tokenMetric = await screen.findByRole('menuitemcheckbox', { name: 'Total tokens' });
    await userEvent.click(tokenMetric);
    await userEvent.click(tokenMetric);

    expect(screen.getByLabelText('Selected timeline metrics')).toHaveTextContent('duration,tokens,cost');
  });

  it('keeps the settings menu open with a disabled action while refresh is pending', async () => {
    let finishRefresh = () => {};
    const refreshPromise = new Promise<void>((resolve) => {
      finishRefresh = resolve;
    });
    const refreshTrace = jest.fn(() => refreshPromise);
    render(<TestWrapper refreshTrace={refreshTrace} />);

    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Refresh trace' }));

    expect(await screen.findByRole('menuitem', { name: 'Refreshing…' })).toHaveAttribute('aria-disabled', 'true');
    expect(refreshTrace).toHaveBeenCalledTimes(1);
    await act(async () => {
      finishRefresh();
      await refreshPromise;
    });
  });
});

const ShowTimelineTreeGanttReader = () => {
  const { showTimelineTreeGantt } = useModelTraceExplorerViewState();
  return <output aria-label="Timeline tree Gantt visibility">{String(showTimelineTreeGantt)}</output>;
};

const renderWithViewStateProvider = (initialShowTimelineTreeGantt?: boolean) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ModelTraceExplorerPreferencesProvider>
          <ModelTraceExplorerViewStateProvider
            modelTrace={MOCK_TRACE}
            assessmentsPaneEnabled={false}
            initialShowTimelineTreeGantt={initialShowTimelineTreeGantt}
          >
            <ShowTimelineTreeGanttReader />
          </ModelTraceExplorerViewStateProvider>
        </ModelTraceExplorerPreferencesProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ModelTraceExplorerViewStateProvider initialShowTimelineTreeGantt', () => {
  it('defaults showTimelineTreeGantt to false when the prop is omitted', () => {
    renderWithViewStateProvider();
    expect(screen.getByLabelText('Timeline tree Gantt visibility')).toHaveTextContent('false');
  });

  it('seeds showTimelineTreeGantt to true when initialShowTimelineTreeGantt is true', () => {
    renderWithViewStateProvider(true);
    expect(screen.getByLabelText('Timeline tree Gantt visibility')).toHaveTextContent('true');
  });
});

const SelectedSpanReader = () => {
  const { selectedNode } = useModelTraceExplorerViewState();
  return <output aria-label="Selected span">{selectedNode?.title}</output>;
};

const SelectedSpanWrapper = ({ modelTrace }: { modelTrace: ModelTrace }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <ModelTraceExplorerPreferencesProvider>
        <ModelTraceExplorerViewStateProvider modelTrace={modelTrace} assessmentsPaneEnabled={false}>
          <SelectedSpanReader />
        </ModelTraceExplorerViewStateProvider>
      </ModelTraceExplorerPreferencesProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

describe('ModelTraceExplorerViewStateProvider refreshed selection', () => {
  it('retains the selected span by key when refreshed span data replaces the tree', () => {
    const { rerender } = render(<SelectedSpanWrapper modelTrace={MOCK_TRACE} />);
    expect(screen.getByLabelText('Selected span')).toHaveTextContent('document-qa-chain');
    const refreshedTrace: ModelTrace = {
      ...MOCK_TRACE,
      data: {
        spans: MOCK_TRACE.data.spans.map((span) => ({ ...span, name: `${span.name} refreshed` })),
      },
    };

    rerender(<SelectedSpanWrapper modelTrace={refreshedTrace} />);

    expect(screen.getByLabelText('Selected span')).toHaveTextContent('document-qa-chain refreshed');
  });

  it('selects the root when an in-progress multi-root trace becomes complete', () => {
    const inProgressTrace: ModelTrace = {
      ...MOCK_TRACE,
      data: {
        spans: MOCK_TRACE.data.spans.map((span, index) =>
          index === 1 ? { ...span, parent_id: 'missing-parent' } : span,
        ),
      },
    };
    const { rerender } = render(<SelectedSpanWrapper modelTrace={inProgressTrace} />);
    expect(screen.getByLabelText('Selected span')).toBeEmptyDOMElement();

    rerender(<SelectedSpanWrapper modelTrace={MOCK_TRACE} />);

    expect(screen.getByLabelText('Selected span')).toHaveTextContent('document-qa-chain');
  });
});
