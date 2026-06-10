import { describe, beforeEach, afterEach, jest, it, expect } from '@jest/globals';
import React from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';

import { RunEvaluationButton } from './RunEvaluationButton';

const setupUserEvent = () => userEvent.setup({ pointerEventsCheck: 0 });

const getJudgeCheckboxByName = (name: string): HTMLInputElement =>
  screen.getByRole('checkbox', { name: new RegExp(name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')) }) as HTMLInputElement;

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

const mockNavigate = jest.fn();
jest.mock('../../../common/utils/RoutingUtils', () => ({
  __esModule: true,
  ...jest.requireActual<typeof import('../../../common/utils/RoutingUtils')>('../../../common/utils/RoutingUtils'),
  useNavigate: () => mockNavigate,
}));

const mockInvokeMutate = jest.fn();
const mockResetSubmit = jest.fn();
let mockInvokeState: {
  isLoading: boolean;
  error: Error | null;
} = { isLoading: false, error: null };

jest.mock('./hooks/useInvokeGenAIEvaluation', () => ({
  __esModule: true,
  useInvokeGenAIEvaluation: () => ({
    mutate: mockInvokeMutate,
    isLoading: mockInvokeState.isLoading,
    error: mockInvokeState.error,
    reset: mockResetSubmit,
  }),
}));

describe('RunEvaluationButton', () => {
  let user: ReturnType<typeof setupUserEvent>;

  beforeEach(() => {
    user = setupUserEvent();
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
    mockNavigate.mockReset();
    mockInvokeMutate.mockReset();
    mockResetSubmit.mockReset();
    mockInvokeState = { isLoading: false, error: null };
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  const renderButton = (experimentId = 'exp-1') => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <DesignSystemProvider>
            <RunEvaluationButton experimentId={experimentId} />
          </DesignSystemProvider>
        </MemoryRouter>
      </QueryClientProvider>,
    );
  };

  it('opens straight into the trace eval flow (no tabs, no code-snippet content)', async () => {
    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

    // After the tab refactor, the modal is single-purpose: no tablist, no snippet content.
    expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
    expect(screen.queryByRole('tab')).not.toBeInTheDocument();
    expect(screen.queryByText(/mlflow\.search_traces/)).not.toBeInTheDocument();
    expect(screen.queryByText(/eval_dataset = \[\{/)).not.toBeInTheDocument();

    // …and the eval-configuration UI is shown immediately.
    expect(screen.getByPlaceholderText('Search judges')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run judge' })).toBeInTheDocument();
  });

  it('defaults the Select traces button to "all traces selected" when traces are available', async () => {
    mockUseSearchMlflowTraces.mockReturnValue({
      data: [{ trace_id: 't-1' }, { trace_id: 't-2' }, { trace_id: 't-3' }],
      isLoading: false,
      isFetching: false,
    });

    renderButton('exp-7');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '3 traces selected' })).toBeInTheDocument();
    });
  });

  it('shows the judges section with empty-state when the experiment has no custom scorers', async () => {
    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

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
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.getByRole('radio', { name: /Custom LLM-as-a-judge \(2\)/ })).toBeInTheDocument();
    expect(screen.getByText('My Custom Judge')).toBeInTheDocument();
    expect(screen.getByText('Another Judge')).toBeInTheDocument();
    expect(screen.queryByText('Session Judge')).not.toBeInTheDocument();
    expect(screen.queryByText('Code Scorer')).not.toBeInTheDocument();
  });

  it('switches to the pre-built tab and renders pre-built LLM-as-a-judge templates', async () => {
    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await user.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));

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
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await user.type(screen.getByPlaceholderText('Search judges'), 'alpha');

    await waitFor(() => {
      expect(screen.getByText('Alpha Judge')).toBeInTheDocument();
    });
    expect(screen.queryByText('Beta Judge')).not.toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Custom LLM-as-a-judge \(1\)/ })).toBeInTheDocument();
  });

  it('toggles selection state with a single click on the judge row', async () => {
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [{ name: 'My Custom Judge', type: 'llm', isSessionLevelScorer: false }],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

    const checkbox = getJudgeCheckboxByName('My Custom Judge');
    expect(checkbox).not.toBeChecked();

    await user.click(checkbox);
    expect(checkbox).toBeChecked();

    await user.click(checkbox);
    expect(checkbox).not.toBeChecked();
  });

  it('renders exactly one checkbox-role element per judge row', async () => {
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [{ name: 'My Custom Judge', type: 'llm', isSessionLevelScorer: false }],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.getAllByRole('checkbox', { name: /My Custom Judge/ })).toHaveLength(1);
  });

  it('does not show the endpoint section until a pre-built template is selected', async () => {
    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.queryByText('Endpoint')).not.toBeInTheDocument();
  });

  it('closes the modal when the Cancel button is clicked', async () => {
    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    expect(screen.getByRole('dialog', { name: 'Run evaluation' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: 'Run evaluation' })).not.toBeInTheDocument();
    });
  });

  it('clears judge / template / endpoint selections when the modal is cancelled and reopened', async () => {
    mockUseSearchMlflowTraces.mockReturnValue({
      data: [{ trace_id: 't-1' }],
      isLoading: false,
      isFetching: false,
    });
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [{ name: 'My Custom Judge', type: 'llm', isSessionLevelScorer: false }],
      },
      isLoading: false,
    });
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

    // First session: pick a custom judge, switch to pre-built and pick Safety, the
    // endpoint section auto-selects an endpoint → Run judge is enabled.
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await user.click(getJudgeCheckboxByName('My Custom Judge'));
    await user.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));
    await user.click(getJudgeCheckboxByName('Safety'));
    await waitFor(() => {
      expect(screen.getByText('Endpoint')).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: 'Run judges' })).toBeEnabled();

    // Cancel — the close handler should wipe judge/template/endpoint state.
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: 'Run evaluation' })).not.toBeInTheDocument();
    });

    // Second session: re-open. The button is back to "Run judge" (singular, disabled),
    // the Endpoint section is hidden (no template selected), and once we visit each
    // pill the previously-checked rows are unchecked.
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    expect(screen.queryByText('Endpoint')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run judge' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Run judges' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));
    expect(await screen.findByText('Safety')).toBeInTheDocument();
    expect(getJudgeCheckboxByName('Safety')).not.toBeChecked();

    await user.click(screen.getByRole('radio', { name: /Custom LLM-as-a-judge/ }));
    expect(await screen.findByText('My Custom Judge')).toBeInTheDocument();
    expect(getJudgeCheckboxByName('My Custom Judge')).not.toBeChecked();
  });

  it('disables the Run judge button when no judges are selected', async () => {
    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

    expect(screen.getByRole('button', { name: 'Run judge' })).toBeDisabled();
  });

  it('keeps the Run judge button disabled when judges are selected but no traces are', async () => {
    mockUseSearchMlflowTraces.mockReturnValue({ data: [], isLoading: false, isFetching: false });
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [{ name: 'My Custom Judge', type: 'llm', isSessionLevelScorer: false }],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await user.click(getJudgeCheckboxByName('My Custom Judge'));

    expect(screen.getByRole('button', { name: 'Run judge' })).toBeDisabled();
  });

  it('enables the Run judge button for a single custom judge (no endpoint required)', async () => {
    // Custom LLM judges carry their own model, so this flow only needs traces + judge.
    mockUseSearchMlflowTraces.mockReturnValue({
      data: [{ trace_id: 't-1' }],
      isLoading: false,
      isFetching: false,
    });
    mockUseGetScheduledScorers.mockReturnValue({
      data: {
        experimentId: 'exp-1',
        scheduledScorers: [{ name: 'My Custom Judge', type: 'llm', isSessionLevelScorer: false }],
      },
      isLoading: false,
    });

    renderButton('exp-1');
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await user.click(getJudgeCheckboxByName('My Custom Judge'));

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
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await user.click(getJudgeCheckboxByName('Alpha Judge'));
    await user.click(getJudgeCheckboxByName('Beta Judge'));

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
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await user.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));
    await user.click(getJudgeCheckboxByName('Safety'));

    expect(screen.getByRole('button', { name: 'Run judge' })).toBeDisabled();
  });

  it('reveals the endpoint section and selector after selecting a pre-built template', async () => {
    mockUseSearchMlflowTraces.mockReturnValue({
      data: [{ trace_id: 't-1' }],
      isLoading: false,
      isFetching: false,
    });
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
    await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
    await user.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));

    await user.click(getJudgeCheckboxByName('Safety'));

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

  describe('submission', () => {
    const setupReadyToSubmit = () => {
      mockUseSearchMlflowTraces.mockReturnValue({
        data: [{ trace_id: 't-1' }, { trace_id: 't-2' }],
        isLoading: false,
        isFetching: false,
      });
      mockUseGetScheduledScorers.mockReturnValue({
        data: {
          experimentId: 'exp-1',
          scheduledScorers: [
            {
              name: 'My Custom Judge',
              type: 'llm',
              isSessionLevelScorer: false,
              llmTemplate: 'Custom',
              instructions: 'Be strict.',
              model: 'gateway:/custom-endpoint',
              is_instructions_judge: true,
            },
          ],
        },
        isLoading: false,
      });
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
    };

    it('posts the right payload when both a custom judge and a pre-built template are selected', async () => {
      setupReadyToSubmit();

      renderButton('exp-1');
      await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
      await user.click(getJudgeCheckboxByName('My Custom Judge'));
      await user.click(screen.getByRole('radio', { name: /Pre-built LLM-as-a-judge/ }));
      await user.click(getJudgeCheckboxByName('Safety'));
      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Run judges' })).toBeEnabled();
      });

      await user.click(screen.getByRole('button', { name: 'Run judges' }));

      expect(mockInvokeMutate).toHaveBeenCalledTimes(1);
      const [params] = mockInvokeMutate.mock.calls[0] as [
        { experimentId: string; traceIds: string[]; serializedScorers: string[] },
        unknown,
      ];

      expect(params.experimentId).toBe('exp-1');
      expect(params.traceIds).toEqual(['t-1', 't-2']);
      expect(params.serializedScorers).toHaveLength(2);

      // The custom judge keeps its own model, never the modal's endpoint.
      const customSerialized = JSON.parse(params.serializedScorers[0]);
      expect(customSerialized.name).toBe('My Custom Judge');
      expect(customSerialized.instructions_judge_pydantic_data.model).toBe('gateway:/custom-endpoint');

      // Pre-built templates with default instructions (like Safety) are serialized as
      // ad-hoc instructions judges — the modal's endpoint is stamped onto their model
      // field and the canonical template prompt is baked into the instructions.
      const templateSerialized = JSON.parse(params.serializedScorers[1]);
      expect(templateSerialized.name).toBe('Safety');
      expect(templateSerialized.instructions_judge_pydantic_data.model).toBe('gateway:/my-chat-endpoint');
      expect(templateSerialized.instructions_judge_pydantic_data.instructions).toContain('safety classifier');
    });

    it('opens the new run in the split-view side panel and resets state on success', async () => {
      setupReadyToSubmit();
      mockInvokeMutate.mockImplementation((...args: unknown[]) => {
        const opts = args[1] as { onSuccess: (data: unknown) => void };
        opts.onSuccess({ job_id: 'job-9', run_id: 'run-42' });
      });

      renderButton('exp-1');
      await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
      await user.click(getJudgeCheckboxByName('My Custom Judge'));
      await user.click(screen.getByRole('button', { name: 'Run judge' }));

      expect(mockNavigate).toHaveBeenCalledWith({
        pathname: '/experiments/exp-1/evaluation-runs',
        search: '?selectedRunUuid=run-42',
      });
      await waitFor(() => {
        expect(screen.queryByRole('dialog', { name: 'Run evaluation' })).not.toBeInTheDocument();
      });

      // Re-opening shows a fresh form (no Custom Judge pre-checked).
      await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
      expect(getJudgeCheckboxByName('My Custom Judge')).not.toBeChecked();
    });

    it('shows a loading spinner on Run judge and disables Cancel while the request is in flight', async () => {
      setupReadyToSubmit();
      mockInvokeState = { isLoading: true, error: null };

      renderButton('exp-1');
      await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
      // Select a judge so the OK button isn't disabled by runJudgeDisabled; this isolates
      // the loading: isSubmitting branch as the only thing affecting the button.
      await user.click(getJudgeCheckboxByName('My Custom Judge'));

      const okButton = document.querySelector<HTMLButtonElement>(
        '[data-component-id="mlflow.eval-runs.start-run-modal.footer.ok"]',
      );
      expect(okButton).not.toBeNull();
      // The design-system Button reflects `loading` via the btn-loading class / loading attr
      // (it does not set the DOM disabled attribute), and swallows clicks while loading.
      expect(okButton?.className).toContain('btn-loading');
      expect(okButton).toHaveAttribute('loading', 'true');
      await user.click(okButton as HTMLButtonElement);
      expect(mockInvokeMutate).not.toHaveBeenCalled();

      expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
    });

    it('renders an inline error alert when the API call fails', async () => {
      setupReadyToSubmit();
      mockInvokeState = { isLoading: false, error: new Error('Boom: scorer rejected.') };

      renderButton('exp-1');
      await user.click(screen.getByRole('button', { name: 'Run evaluation' }));

      expect(screen.getByText('Boom: scorer rejected.')).toBeInTheDocument();
    });

    it('does not call the mutation if the user closes the modal without submitting', async () => {
      setupReadyToSubmit();

      renderButton('exp-1');
      await user.click(screen.getByRole('button', { name: 'Run evaluation' }));
      await user.click(screen.getByRole('button', { name: 'Cancel' }));

      expect(mockInvokeMutate).not.toHaveBeenCalled();
      expect(mockNavigate).not.toHaveBeenCalled();
    });
  });
});
