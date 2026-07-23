import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModal } from './IssueDetectionModal';
import { useInvokeIssueDetection } from './hooks/useInvokeIssueDetection';
import { clearSubmittedIssueDetectionJob, getSubmittedIssueDetectionJob } from './IssueDetectionJobNotifications';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import { useLogTelemetryEvent } from '../../../../../telemetry/hooks/useLogTelemetryEvent';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';

jest.mock('./hooks/useInvokeIssueDetection');
jest.mock('../../../../../gateway/hooks/useCreateSecret');
jest.mock('../../../../../telemetry/hooks/useLogTelemetryEvent', () => ({
  useLogTelemetryEvent: jest.fn(),
}));
jest.mock('../../../../../gateway/hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: jest.fn(),
}));
jest.mock('../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration', () => ({
  useApiKeyConfiguration: jest.fn(),
}));

jest.mock('../../../SelectTracesModal', () => ({
  SelectTracesModal: ({ onClose, onSuccess }: { onClose: () => void; onSuccess: (traceIds: string[]) => void }) => (
    <div data-testid="select-traces-modal">
      <button data-testid="select-traces-cancel" onClick={onClose}>
        Cancel
      </button>
      <button data-testid="select-traces-confirm" onClick={() => onSuccess(['trace-1', 'trace-2'])}>
        Select
      </button>
    </div>
  ),
}));

jest.mock('./IssueDetectionModelDropdown', () => ({
  ...jest.requireActual<typeof import('./IssueDetectionModelDropdown')>('./IssueDetectionModelDropdown'),
  IssueDetectionModelDropdown: ({
    value,
    onChange,
  }: {
    value: { mode: string; provider: string; model: string; endpointName?: string };
    onChange: (value: unknown) => void;
  }) => (
    <div data-testid="model-dropdown">
      <span data-testid="model-dropdown-value">
        {value.mode === 'endpoint' ? value.endpointName : `${value.provider}/${value.model}`}
      </span>
      <button
        data-testid="pick-anthropic"
        onClick={() => onChange({ mode: 'direct', provider: 'anthropic', model: 'claude-sonnet-4-6' })}
      >
        anthropic
      </button>
    </div>
  ),
}));

describe('IssueDetectionModal', () => {
  const defaultProps = {
    onClose: jest.fn(),
    experimentId: 'exp-123',
  };

  let mockInvokeIssueDetection: jest.Mock;
  let mockCreateSecret: jest.Mock;
  let mockLogTelemetryEvent: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    clearSubmittedIssueDetectionJob();
    mockLogTelemetryEvent = jest.fn();
    jest.mocked(useLogTelemetryEvent).mockReturnValue(mockLogTelemetryEvent as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [], isLoading: false, refetch: jest.fn() } as any);
    jest.mocked(useApiKeyConfiguration).mockReturnValue({
      existingSecrets: [{ secret_id: 'secret-123', secret_name: 'my-key' }],
      hasExistingSecrets: true,
      isLoadingSecrets: false,
      authModes: [],
      defaultAuthMode: '',
      selectedAuthMode: undefined,
      isLoadingProviderConfig: false,
    } as any);

    mockInvokeIssueDetection = jest.fn((_request, options) => {
      (options as { onSuccess?: (response: { job_id: string; run_id: string }) => void })?.onSuccess?.({
        job_id: 'job-123',
        run_id: 'run-456',
      });
    });
    jest.mocked(useInvokeIssueDetection).mockReturnValue({
      mutate: mockInvokeIssueDetection,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);

    mockCreateSecret = jest.fn((_request, options) => {
      (options as { onSuccess?: (response: { secret: { secret_id: string } }) => void })?.onSuccess?.({
        secret: { secret_id: 'new-secret-123' },
      });
    });
    jest.mocked(useCreateSecret).mockReturnValue({
      mutate: mockCreateSecret,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
  });

  test('renders hero, provider summary, trace count, and Run button', () => {
    const availableTraceIds = Array.from({ length: 40 }, (_, i) => `trace-${i}`);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} availableTraceIds={availableTraceIds} />);

    expect(screen.getByText('Detect Issues')).toBeInTheDocument();
    expect(screen.getByText('Find failure patterns hiding in your traces, automatically.')).toBeInTheDocument();
    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByTestId('model-dropdown-value')).toHaveTextContent('openai/gpt-5.5');
    expect(screen.getByText('40 traces selected')).toBeInTheDocument();
    expect(screen.getByText(/Estimated cost: ~\$0\.10-\$0\.40/)).toBeInTheDocument();
    expect(screen.getByText('Run Analysis')).toBeInTheDocument();
    // No API key input anywhere
    expect(screen.queryByText(/API key/i)).not.toBeInTheDocument();
  });

  test('defaults to the 50 most recent traces when none are explicitly selected', () => {
    const availableTraceIds = Array.from({ length: 80 }, (_, i) => `trace-${i}`);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} availableTraceIds={availableTraceIds} />);

    expect(screen.getByText('50 traces selected')).toBeInTheDocument();
  });

  test('tells the user to log traces first when the experiment has none', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('No traces yet. Log traces to this experiment first.')).toBeInTheDocument();
    expect(screen.getByText('Run Analysis').closest('button')).toBeDisabled();
  });

  test('clicking the traces card opens trace selection and updates the count', async () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1', 'trace-2', 'trace-3']} />,
    );

    await userEvent.click(screen.getByTestId('traces-card'));
    expect(screen.getByTestId('select-traces-modal')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('select-traces-confirm'));
    expect(screen.queryByTestId('select-traces-modal')).not.toBeInTheDocument();
    expect(screen.getByText('2 traces selected')).toBeInTheDocument();
  });

  test('defaults to the first gateway endpoint when endpoints exist', async () => {
    jest
      .mocked(useEndpointsQuery)
      .mockReturnValue({ data: [{ name: 'my-endpoint' }], isLoading: false, refetch: jest.fn() } as any);

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(screen.getByTestId('model-dropdown-value')).toHaveTextContent('my-endpoint');

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);
    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ endpoint_name: 'my-endpoint', secret_id: undefined }),
        expect.any(Object),
      );
    });
  });

  test('changing the model in the dropdown updates the selection and submission', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(screen.getByTestId('model-dropdown-value')).toHaveTextContent('openai/gpt-5.5');

    await userEvent.click(screen.getByTestId('pick-anthropic'));
    expect(screen.getByTestId('model-dropdown-value')).toHaveTextContent('anthropic/claude-sonnet-4-6');

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);
    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ provider: 'anthropic', model: 'claude-sonnet-4-6' }),
        expect.any(Object),
      );
    });
  });

  test('missing API key turns into the one-last-step view and Continue and run submits', async () => {
    jest.mocked(useInvokeIssueDetection).mockReturnValue({
      mutate: mockInvokeIssueDetection,
      isLoading: false,
      error: new Error(
        "No API key available for provider 'openai'. Save an API key in AI Gateway, or set the OPENAI_API_KEY environment variable on the MLflow server.",
      ),
      reset: jest.fn(),
    } as any);

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(await screen.findByTestId('api-key-view')).toBeInTheDocument();
    expect(screen.getByText('One last step to run issue detection')).toBeInTheDocument();
    expect(screen.queryByText(/No API key available/)).not.toBeInTheDocument();

    const continueButton = screen.getByText('Continue and run').closest('button')!;
    expect(continueButton).toBeDisabled();

    await userEvent.type(screen.getByTestId('api-key-input'), 'sk-test-key');
    expect(continueButton).not.toBeDisabled();

    await userEvent.click(continueButton);
    await waitFor(() => {
      expect(mockCreateSecret).toHaveBeenCalledWith(
        expect.objectContaining({ provider: 'openai', secret_value: { api_key: 'sk-test-key' } }),
        expect.any(Object),
      );
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ secret_id: 'new-secret-123' }),
        expect.any(Object),
      );
    });
  });

  test('generic submission errors show an inline alert', () => {
    jest.mocked(useInvokeIssueDetection).mockReturnValue({
      mutate: mockInvokeIssueDetection,
      isLoading: false,
      error: new Error('Job execution is disabled on this server'),
      reset: jest.fn(),
    } as any);

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(screen.getByText('Job execution is disabled on this server')).toBeInTheDocument();
    expect(screen.queryByTestId('api-key-view')).not.toBeInTheDocument();
  });

  test('submits all categories and the saved secret for direct providers', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({
          categories: ['correctness', 'latency', 'execution', 'adherence', 'relevance', 'safety'],
          provider: 'openai',
          model: 'gpt-5.5',
          secret_id: 'secret-123',
          endpoint_name: undefined,
        }),
        expect.any(Object),
      );
    });
  });

  test('shows a low-trace tip when fewer than 10 traces are selected', () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1', 'trace-2', 'trace-3']} />,
    );

    const tip = screen.getByTestId('low-trace-warning');
    expect(tip).toHaveTextContent('You selected only 3 traces. Analyze at least 10 for more accurate results.');
  });

  test('does not show the low-trace tip when at least 10 traces are selected', () => {
    const ids = Array.from({ length: 10 }, (_, i) => `trace-${i}`);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={ids} />);

    expect(screen.queryByTestId('low-trace-warning')).not.toBeInTheDocument();
  });

  test('logs submit context telemetry on submit', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockLogTelemetryEvent).toHaveBeenCalledWith(
        expect.objectContaining({
          componentId: 'mlflow.traces.issue-detection-modal.submit-context',
          value: JSON.stringify({
            selectedTraceCount: 1,
            lowTraceWarningShown: true,
          }),
        }),
      );
    });
  });

  test('records submitted background job when form is submitted', async () => {
    const onClose = jest.fn();
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });

    expect(getSubmittedIssueDetectionJob()).toEqual({
      experimentId: 'exp-123',
      jobId: 'job-123',
      runId: 'run-456',
      traceCount: 1,
    });
  });
});
