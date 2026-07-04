import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { AggregationType } from '@databricks/web-shared/model-trace-explorer';

import { EvalRunsEmptyStateCard } from './EvalRunsEmptyStateCard';

const mockUseTraceMetricsQuery = jest.fn();
jest.mock('../experiment-overview/hooks/useTraceMetricsQuery', () => ({
  __esModule: true,
  useTraceMetricsQuery: (...args: unknown[]) => mockUseTraceMetricsQuery(...args),
}));

const mockTraceMetricsCount = (count: number) =>
  mockUseTraceMetricsQuery.mockReturnValue({
    data: { data_points: [{ values: { [AggregationType.COUNT]: count } }] },
    isSuccess: true,
  });

const mockOpenPanel = jest.fn();
const mockPrefillPrompt = jest.fn();

jest.mock('@mlflow/mlflow/src/assistant', () => ({
  __esModule: true,
  useAssistant: () => ({
    openPanel: mockOpenPanel,
    sendMessage: jest.fn(),
    prefillPrompt: mockPrefillPrompt,
    clearPendingPrompt: jest.fn(),
    pendingPrompt: null,
    reset: jest.fn(),
    setupComplete: true,
    isPanelOpen: false,
    sessionId: null,
    messages: [],
    isStreaming: false,
    error: null,
    currentStatus: null,
    activeTools: [],
    isLoadingConfig: false,
    isLocalServer: true,
    closePanel: jest.fn(),
    regenerateLastMessage: jest.fn(),
    cancelSession: jest.fn(),
    refreshConfig: jest.fn(),
    completeSetup: jest.fn(),
  }),
  AssistantSparkleIcon: ({ iconSize }: { iconSize?: number }) => (
    <span data-testid="assistant-sparkle-icon" data-icon-size={iconSize} />
  ),
}));

// RunEvaluationButton renders a primary "Evaluate traces" button. We stub it so we don't
// have to set up the trace-search + scorer-fetch hooks just to test the empty-state shell.
jest.mock('./RunEvaluationButton', () => ({
  __esModule: true,
  RunEvaluationButton: () => (
    <button data-testid="run-evaluation-button" data-button-type="primary">
      Evaluate traces
    </button>
  ),
}));

const renderCard = () =>
  renderWithIntl(
    <DesignSystemProvider>
      <EvalRunsEmptyStateCard experimentId="42" />
    </DesignSystemProvider>,
  );

beforeEach(() => {
  mockOpenPanel.mockClear();
  mockPrefillPrompt.mockClear();
  mockUseTraceMetricsQuery.mockReset();
  mockTraceMetricsCount(5); // default: experiment has traces
});

describe('EvalRunsEmptyStateCard', () => {
  it('renders the primary "Evaluate traces" CTA at the top', () => {
    renderCard();
    const btn = screen.getByTestId('run-evaluation-button');
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveAttribute('data-button-type', 'primary');
    expect(btn).toHaveTextContent('Evaluate traces');
  });

  it('renders the three tabs in order: Copy for coding agent → Python → MLflow assistant', () => {
    renderCard();
    const tabs = screen.getAllByRole('tab').map((t) => t.textContent?.trim() ?? '');
    expect(tabs).toEqual(['Copy for coding agent', 'Python', 'MLflow assistant']);
  });

  it('renders the sparkle icon on the MLflow assistant tab', () => {
    renderCard();
    expect(screen.getByTestId('assistant-sparkle-icon')).toBeInTheDocument();
  });

  it('clicking "Open assistant" opens the panel and prefills the chat input with the eval prompt', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard();

    await user.click(screen.getByRole('tab', { name: /MLflow assistant/ }));
    await user.click(screen.getByRole('button', { name: /Open assistant/ }));

    expect(mockOpenPanel).toHaveBeenCalledTimes(1);
    expect(mockPrefillPrompt).toHaveBeenCalledTimes(1);
    expect(mockPrefillPrompt.mock.calls[0][0]).toContain('Target experiment ID: 42');
  });

  it('renders the trace-based snippet when the experiment has traces', async () => {
    mockTraceMetricsCount(5);
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard();

    await user.click(screen.getByRole('tab', { name: /Python/ }));
    // CodeSnippet splits text across syntax-highlighted spans, so we read textContent
    // of the active tab panel and check for substrings instead of using getByText.
    const panel = screen.getByRole('tabpanel');
    expect(panel.textContent).toContain('mlflow.search_traces');
    expect(panel.textContent).not.toContain('eval_dataset = [{');
  });

  it('renders the dataset-based snippet when the experiment has no traces', async () => {
    mockTraceMetricsCount(0);
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard();

    await user.click(screen.getByRole('tab', { name: /Python/ }));
    const panel = screen.getByRole('tabpanel');
    expect(panel.textContent).toContain('eval_dataset = [{');
    expect(panel.textContent).not.toContain('mlflow.search_traces');
  });

  it('does not fire the trace-count query until the Python tab is opened (lazy fetch)', async () => {
    mockUseTraceMetricsQuery.mockReset();
    mockUseTraceMetricsQuery.mockImplementation((args: unknown) => {
      // Mirror the real hook: if disabled, no data; once enabled, return our fixture.
      const enabled = (args as { enabled?: boolean })?.enabled !== false;
      return enabled
        ? { data: { data_points: [{ values: { [AggregationType.COUNT]: 5 } }] }, isSuccess: true }
        : { data: undefined, isSuccess: false };
    });

    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderCard();

    // Before any interaction, the hook is called with enabled=false.
    const firstCallArgs = mockUseTraceMetricsQuery.mock.calls[0]?.[0] as { enabled?: boolean };
    expect(firstCallArgs.enabled).toBe(false);

    // After opening the Python tab, the hook is re-called with enabled=true.
    await user.click(screen.getByRole('tab', { name: /Python/ }));
    const lastCallArgs = mockUseTraceMetricsQuery.mock.calls.at(-1)?.[0] as { enabled?: boolean };
    expect(lastCallArgs.enabled).toBe(true);
  });
});
