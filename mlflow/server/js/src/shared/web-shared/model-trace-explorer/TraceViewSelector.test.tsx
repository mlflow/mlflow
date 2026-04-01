import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TraceViewSelector } from './TraceViewSelector';
import type { TraceView } from './hooks/useTraceViews';
import { QueryClient, QueryClientProvider } from '../query-client/queryClient';

const MOCK_TRACE_VIEW: TraceView = {
  view_id: 'tv-abc123',
  name: 'Agent Reasoning',
  trace_id: 'tr-001',
  span_filter: { span_name: 'plan_action' },
  output_path: '$.reasoning',
};

const MOCK_EXPERIMENT_VIEW: TraceView = {
  view_id: 'tv-exp456',
  name: 'Tool Calls',
  experiment_id: '1',
  span_filter: { span_type: 'TOOL' },
};

jest.mock('./hooks/useTraceViews', () => ({
  useTraceViews: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { useTraceViews } = require('./hooks/useTraceViews');

const Wrapper = ({
  views,
  activeViewId: initialViewId = null,
}: {
  views: TraceView[];
  activeViewId?: string | null;
}) => {
  const [activeViewId, setActiveViewId] = useState<string | null>(initialViewId);
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <TraceViewSelector
            traceId="tr-001"
            activeViewId={activeViewId}
            onViewChange={(view) => setActiveViewId(view?.view_id ?? null)}
          />
          <div data-testid="active-view">{activeViewId ?? 'none'}</div>
        </DesignSystemProvider>
      </IntlProvider>
    </QueryClientProvider>
  );
};

describe('TraceViewSelector', () => {
  it('renders nothing when there are no views', () => {
    useTraceViews.mockReturnValue({ data: [], isLoading: false });
    const { container } = render(<Wrapper views={[]} />);
    expect(container.querySelector('[role="combobox"]')).toBeNull();
  });

  it('renders nothing while loading with no data', () => {
    useTraceViews.mockReturnValue({ data: undefined, isLoading: true });
    const { container } = render(<Wrapper views={[]} />);
    expect(container.querySelector('[role="combobox"]')).toBeNull();
  });

  it('shows "Raw Trace" as default label when no view is active', () => {
    useTraceViews.mockReturnValue({ data: [MOCK_TRACE_VIEW], isLoading: false });
    render(<Wrapper views={[MOCK_TRACE_VIEW]} />);
    expect(screen.getByText('Raw Trace')).toBeInTheDocument();
  });

  it('shows the active view name when a view is selected', () => {
    useTraceViews.mockReturnValue({ data: [MOCK_TRACE_VIEW], isLoading: false });
    render(<Wrapper views={[MOCK_TRACE_VIEW]} activeViewId="tv-abc123" />);
    expect(screen.getByText('Agent Reasoning')).toBeInTheDocument();
  });

  it('opens dropdown and shows views grouped by scope', async () => {
    useTraceViews.mockReturnValue({
      data: [MOCK_TRACE_VIEW, MOCK_EXPERIMENT_VIEW],
      isLoading: false,
    });
    render(<Wrapper views={[MOCK_TRACE_VIEW, MOCK_EXPERIMENT_VIEW]} />);

    // Click to open dropdown
    await userEvent.click(screen.getByText('Raw Trace'));

    // Should show both group labels and view names
    expect(screen.getByText('Trace Views')).toBeInTheDocument();
    expect(screen.getByText('Experiment Views')).toBeInTheDocument();
    expect(screen.getByText('Agent Reasoning')).toBeInTheDocument();
    expect(screen.getByText('Tool Calls')).toBeInTheDocument();
  });

  it('calls onViewChange with selected view', async () => {
    useTraceViews.mockReturnValue({
      data: [MOCK_TRACE_VIEW],
      isLoading: false,
    });
    render(<Wrapper views={[MOCK_TRACE_VIEW]} />);

    await userEvent.click(screen.getByText('Raw Trace'));
    await userEvent.click(screen.getByText('Agent Reasoning'));

    expect(screen.getByTestId('active-view').textContent).toBe('tv-abc123');
  });

  it('calls onViewChange with null when switching back to Raw Trace', async () => {
    useTraceViews.mockReturnValue({
      data: [MOCK_TRACE_VIEW],
      isLoading: false,
    });
    render(<Wrapper views={[MOCK_TRACE_VIEW]} activeViewId="tv-abc123" />);

    await userEvent.click(screen.getByText('Agent Reasoning'));
    await userEvent.click(screen.getByText('Raw Trace'));

    expect(screen.getByTestId('active-view').textContent).toBe('none');
  });
});
