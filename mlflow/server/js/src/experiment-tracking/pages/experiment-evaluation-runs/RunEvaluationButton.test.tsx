import { describe, beforeAll, beforeEach, afterEach, afterAll, jest, it, expect } from '@jest/globals';
import React from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AggregationType } from '@databricks/web-shared/model-trace-explorer';

import { RunEvaluationButton } from './RunEvaluationButton';

const mockUseTraceMetricsQuery = jest.fn();
jest.mock('../experiment-overview/hooks/useTraceMetricsQuery', () => ({
  __esModule: true,
  useTraceMetricsQuery: (...args: unknown[]) => mockUseTraceMetricsQuery(...args),
}));

const mockUseSearchMlflowTraces = jest.fn();
jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  __esModule: true,
  ...jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  ),
  useSearchMlflowTraces: (...args: unknown[]) => mockUseSearchMlflowTraces(...args),
}));

const mockUseGetScheduledScorers = jest.fn();
jest.mock('../experiment-scorers/hooks/useGetScheduledScorers', () => ({
  __esModule: true,
  useGetScheduledScorers: (...args: unknown[]) => mockUseGetScheduledScorers(...args),
}));

const mockUseEndpointsQuery = jest.fn();
jest.mock('../../../gateway/hooks/useEndpointsQuery', () => ({
  __esModule: true,
  useEndpointsQuery: (...args: unknown[]) => mockUseEndpointsQuery(...args),
}));

jest.mock('../../../gateway/components/endpoint-form', () => ({
  __esModule: true,
  CreateEndpointModal: ({ open }: { open: boolean }) =>
    open ? <div data-testid="create-endpoint-modal">Create Endpoint Modal</div> : null,
}));

const mockTraceMetricsCount = (count: number) =>
  mockUseTraceMetricsQuery.mockReturnValue({
    data: { data_points: [{ values: { [AggregationType.COUNT]: count } }] },
    isSuccess: true,
  });

const COPY_BUTTON_SELECTOR = '[data-component-id="mlflow.eval-runs.start-run-modal.copy-snippet"]';

describe('RunEvaluationButton', () => {
  let originalClipboard: typeof navigator.clipboard;

  beforeAll(() => {
    originalClipboard = navigator.clipboard;
  });

  beforeEach(() => {
    Object.defineProperty(global.navigator, 'clipboard', {
      value: { writeText: jest.fn(() => Promise.resolve()) },
      writable: true,
    });
    mockUseTraceMetricsQuery.mockReset();
    mockTraceMetricsCount(1);
    mockUseSearchMlflowTraces.mockReset();
    mockUseSearchMlflowTraces.mockReturnValue({ data: [], isLoading: false, isFetching: false });
    mockUseGetScheduledScorers.mockReset();
    mockUseGetScheduledScorers.mockReturnValue({
      data: { experimentId: 'exp-1', scheduledScorers: [] },
      isLoading: false,
    });
    mockUseEndpointsQuery.mockReset();
    mockUseEndpointsQuery.mockReturnValue({
      data: [],
      error: undefined,
      isLoading: false,
      refetch: jest.fn(),
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  afterAll(() => {
    Object.defineProperty(global.navigator, 'clipboard', {
      value: originalClipboard,
      writable: true,
    });
  });

  const renderButton = (experimentId = 'exp-1') =>
    renderWithIntl(
      <DesignSystemProvider>
        <RunEvaluationButton experimentId={experimentId} />
      </DesignSystemProvider>,
    );

  // Click the copy-snippet button and return the string that was passed to clipboard.writeText.
  const copyCurrentSnippet = async (): Promise<string> => {
    const copyButton = await waitFor(() => {
      const el = document.querySelector<HTMLElement>(COPY_BUTTON_SELECTOR);
      expect(el).not.toBeNull();
      return el!;
    });
    await userEvent.click(copyButton);

    expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
    return jest.mocked(navigator.clipboard.writeText).mock.calls[0][0] as string;
  };

  it('renders the dataset-based snippet and copies it when the experiment has no traces', async () => {
    mockTraceMetricsCount(0);

    renderButton('exp-1');
    // No traces → Start an Evaluation tab is hidden, modal lands on Code Snippet directly.
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    const copied = await copyCurrentSnippet();

    expect(copied).toContain('mlflow.set_experiment(experiment_id="exp-1")');
    expect(copied).toContain('eval_dataset = [{');
    expect(copied).toContain('predict_fn=predict');
    expect(copied).not.toContain('mlflow.search_traces');
  });

  it('renders the trace-based snippet and copies it when the experiment has traces', async () => {
    mockTraceMetricsCount(5);

    renderButton('exp-7');
    // Trace experiments auto-land on Start an Evaluation; navigate to Code Snippet
    // before copying — otherwise the copy button isn't rendered yet.
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await userEvent.click(screen.getByRole('tab', { name: 'Code Snippet' }));

    const copied = await copyCurrentSnippet();

    expect(copied).toContain('mlflow.set_experiment(experiment_id="exp-7")');
    expect(copied).toContain('mlflow.search_traces(max_results=20)');
    expect(copied).not.toContain('eval_dataset');
    expect(copied).not.toContain('predict_fn=predict');
  });

  it('shows a spinner instead of the tabs until the trace count query resolves', async () => {
    // Simulate the trace count query still in flight after the modal opens.
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isSuccess: false });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    // Neither tab renders yet — we wait for the trace count before deciding the layout,
    // otherwise users would see Code Snippet flash and then jump to Start an Evaluation.
    expect(screen.queryByRole('tab', { name: 'Code Snippet' })).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Start an Evaluation' })).not.toBeInTheDocument();
    expect(document.querySelector(COPY_BUTTON_SELECTOR)).toBeNull();
    expect(screen.queryByText(/eval_dataset = \[\{/)).not.toBeInTheDocument();
    expect(screen.queryByText(/mlflow\.search_traces/)).not.toBeInTheDocument();
  });

  it('hides the Start an Evaluation tab when the experiment has no traces', async () => {
    mockTraceMetricsCount(0);

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.getByRole('tab', { name: 'Code Snippet' })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Start an Evaluation' })).not.toBeInTheDocument();
  });

  it('auto-selects the Start an Evaluation tab when the experiment has traces', async () => {
    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    // Both tabs render, with the trace tab listed first and auto-selected. We confirm
    // it's the active tab by asserting its content (the Judges section) is shown.
    expect(screen.getByRole('tab', { name: 'Start an Evaluation' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Code Snippet' })).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search judges')).toBeInTheDocument();
  });

  it('defaults the Select traces button to "all traces selected" when traces are available', async () => {
    mockTraceMetricsCount(3);
    mockUseSearchMlflowTraces.mockReturnValue({
      data: [{ trace_id: 't-1' }, { trace_id: 't-2' }, { trace_id: 't-3' }],
      isLoading: false,
      isFetching: false,
    });

    renderButton('exp-7');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '3 traces selected' })).toBeInTheDocument();
    });
  });

  it('shows the judges section with empty-state when the experiment has no custom scorers', async () => {
    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.getByPlaceholderText('Search judges')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Custom LLM-as-a-judge \(0\)/ })).toBeInTheDocument();
    expect(screen.getByText('No custom LLM-as-a-judge scorers found')).toBeInTheDocument();
  });

  it('lists custom LLM-as-a-judge scorers returned by the API and reflects the count in the pill', async () => {
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [
          { name: 'My Custom Judge', type: 'llm', isSessionLevelScorer: false },
          { name: 'Another Judge', type: 'llm', isSessionLevelScorer: false },
          // Session-level and non-LLM scorers must be filtered out of the trace-level picker.
          { name: 'Session Judge', type: 'llm', isSessionLevelScorer: true },
          { name: 'Code Scorer', type: 'custom-code', isSessionLevelScorer: false },
        ],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.getByRole('radio', { name: /Custom LLM-as-a-judge \(2\)/ })).toBeInTheDocument();
    expect(screen.getByText('My Custom Judge')).toBeInTheDocument();
    expect(screen.getByText('Another Judge')).toBeInTheDocument();
    expect(screen.queryByText('Session Judge')).not.toBeInTheDocument();
    expect(screen.queryByText('Code Scorer')).not.toBeInTheDocument();
  });

  it('switches to the pre-built tab and renders pre-built LLM-as-a-judge templates', async () => {
    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await userEvent.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));

    // The "Custom" and "Guidelines" templates are intentionally hidden from this picker.
    expect(screen.queryByText('Guidelines')).not.toBeInTheDocument();
    // At least one well-known pre-built template should render in its place.
    expect(screen.getByText('Safety')).toBeInTheDocument();
  });

  it('filters the judge list by the search input', async () => {
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [
          { name: 'Alpha Judge', type: 'llm', isSessionLevelScorer: false },
          { name: 'Beta Judge', type: 'llm', isSessionLevelScorer: false },
        ],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await userEvent.type(screen.getByPlaceholderText('Search judges'), 'alpha');

    await waitFor(() => {
      expect(screen.getByText('Alpha Judge')).toBeInTheDocument();
    });
    expect(screen.queryByText('Beta Judge')).not.toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Custom LLM-as-a-judge \(1\)/ })).toBeInTheDocument();
  });

  it('toggles selection state with a single click on the judge label text', async () => {
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [{ name: 'My Custom Judge', type: 'llm', isSessionLevelScorer: false }],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    const labelText = screen.getByText('My Custom Judge');
    const row = labelText.closest('[role="checkbox"]') as HTMLElement;
    expect(row).not.toBeNull();
    expect(row).toHaveAttribute('aria-checked', 'false');

    await userEvent.click(labelText);
    expect(row).toHaveAttribute('aria-checked', 'true');

    await userEvent.click(labelText);
    expect(row).toHaveAttribute('aria-checked', 'false');
  });

  it('does not show the endpoint section until a pre-built template is selected', async () => {
    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.queryByText('Endpoint')).not.toBeInTheDocument();
  });

  it('hides the Cancel and Run judge footer buttons when the experiment has no traces', async () => {
    mockTraceMetricsCount(0);

    renderButton('exp-1');
    // With no traces, the Start an Evaluation tab is hidden entirely and the modal
    // lands on Code Snippet, so neither footer button should render.
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run judge' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run judges' })).not.toBeInTheDocument();
  });

  it('renders the Cancel and Run judge footer buttons on the Start an Evaluation tab', async () => {
    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run judge' })).toBeInTheDocument();
  });

  it('hides the footer again when switching from Start an Evaluation to Code Snippet', async () => {
    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: 'Code Snippet' }));

    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run judge' })).not.toBeInTheDocument();
  });

  it('closes the modal when the Cancel button is clicked', async () => {
    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));

    await waitFor(() => {
      expect(screen.queryByRole('tab', { name: 'Start an Evaluation' })).not.toBeInTheDocument();
    });
  });

  it('disables the Run judge button when no judges are selected', async () => {
    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.getByRole('button', { name: 'Run judge' })).toBeDisabled();
  });

  it('enables the Run judge button for a single custom judge (no endpoint required)', async () => {
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [{ name: 'My Custom Judge', type: 'llm', isSessionLevelScorer: false }],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await userEvent.click(screen.getByText('My Custom Judge'));

    expect(screen.getByRole('button', { name: 'Run judge' })).toBeEnabled();
  });

  it('switches the button text to "Run judges" when multiple judges are selected', async () => {
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [
          { name: 'Alpha Judge', type: 'llm', isSessionLevelScorer: false },
          { name: 'Beta Judge', type: 'llm', isSessionLevelScorer: false },
        ],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await userEvent.click(screen.getByText('Alpha Judge'));
    await userEvent.click(screen.getByText('Beta Judge'));

    expect(screen.getByRole('button', { name: 'Run judges' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run judge' })).not.toBeInTheDocument();
  });

  it('keeps the Run judge button disabled when a pre-built template is selected without an endpoint', async () => {
    // No endpoints available → EndpointSelector renders no value → Run judge stays disabled.
    mockUseEndpointsQuery.mockReturnValue({
      data: [],
      error: undefined,
      isLoading: false,
      refetch: jest.fn(),
    });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await userEvent.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));
    await userEvent.click(screen.getByText('Safety'));

    expect(screen.getByRole('button', { name: 'Run judge' })).toBeDisabled();
  });

  it('reveals the endpoint section and selector after selecting a pre-built template', async () => {
    mockUseEndpointsQuery.mockReturnValue({
      data: [
        {
          endpoint_id: 'ep-1',
          name: 'my-chat-endpoint',
          created_at: 1,
          last_updated_at: 1,
          model_mappings: [
            {
              mapping_id: 'mm-1',
              endpoint_id: 'ep-1',
              model_definition_id: 'md-1',
              weight: 1,
              created_at: 1,
              model_definition: {
                model_definition_id: 'md-1',
                name: 'model-1',
                provider: 'openai',
                model_name: 'gpt-4o',
                secret_id: 'secret-1',
                secret_name: 'secret-1',
                created_at: 1,
                last_updated_at: 1,
                endpoint_count: 1,
              },
            },
          ],
        },
      ],
      error: undefined,
      isLoading: false,
      refetch: jest.fn(),
    });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await userEvent.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));

    const safetyRow = screen.getByText('Safety').closest('[role="checkbox"]') as HTMLElement;
    await userEvent.click(safetyRow);

    await waitFor(() => {
      expect(screen.getByText('Endpoint')).toBeInTheDocument();
    });
    // EndpointSelector auto-selects the first endpoint, so its name should be visible.
    await waitFor(() => {
      expect(screen.getByText('my-chat-endpoint')).toBeInTheDocument();
    });
    // With a pre-built template selected AND an endpoint auto-selected, Run judge becomes enabled.
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Run judge' })).toBeEnabled();
    });
  });
});
