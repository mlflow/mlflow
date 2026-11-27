import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModal } from './IssueDetectionModal';
import { useInvokeIssueDetection } from './hooks/useInvokeIssueDetection';
import { clearSubmittedIssueDetectionJob, getSubmittedIssueDetectionJob } from './IssueDetectionJobNotifications';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { useModelsQuery } from '../../../../../gateway/hooks/useModelsQuery';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import { useIssueDetectionSettings } from '../../../../pages/experiment-issues/useIssueDetectionSettings';
import { defaultIssueDetectionSettings } from '../../../../pages/experiment-issues/issueDetectionSettings';

jest.mock('./hooks/useInvokeIssueDetection');
jest.mock('../../../../pages/experiment-issues/useIssueDetectionSettings', () => ({
  useIssueDetectionSettings: jest.fn(),
}));
jest.mock('../../../../../gateway/hooks/useCreateSecret');
jest.mock('../../../../../gateway/hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: jest.fn(),
}));
jest.mock('../../../../../gateway/hooks/useModelsQuery', () => ({
  useModelsQuery: jest.fn(),
}));
jest.mock('../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration', () => ({
  useApiKeyConfiguration: jest.fn(),
}));

jest.mock('../../../SelectTracesModal', () => ({
  SelectTracesModal: ({
    onClose,
    onSuccess,
    defaultGroupBySession,
  }: {
    onClose: () => void;
    onSuccess: (traceIds: string[]) => void;
    defaultGroupBySession?: boolean;
  }) => (
    <div data-testid="select-traces-modal">
      <div data-testid="default-group-by-session">{String(defaultGroupBySession)}</div>
      <button data-testid="select-traces-cancel" onClick={onClose}>
        Cancel
      </button>
      <button data-testid="select-traces-confirm" onClick={() => onSuccess(['trace-1', 'trace-2'])}>
        Select
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

  const changeModelToAnthropic = async () => {
    await userEvent.click(screen.getByTestId('model-dropdown-trigger'));
    await userEvent.click(screen.getByTestId('model-provider-anthropic'));
    await userEvent.click(screen.getByTestId('model-option-anthropic-claude-opus-4-8'));
  };

  const mockSettings = (settings = defaultIssueDetectionSettings()) =>
    jest.mocked(useIssueDetectionSettings).mockReturnValue({ data: settings, isLoading: false, error: null } as any);

  beforeEach(() => {
    jest.clearAllMocks();
    clearSubmittedIssueDetectionJob();
    mockSettings();
    jest.mocked(useEndpointsQuery).mockReturnValue({ data: [], isLoading: false, refetch: jest.fn() } as any);
    jest.mocked(useModelsQuery).mockImplementation(
      ({ provider } = {}) =>
        ({
          data: provider === 'anthropic' ? [{ model: 'claude-opus-4-8' }, { model: 'claude-sonnet-4-6' }] : undefined,
          isLoading: false,
          refetch: jest.fn(),
        }) as any,
    );
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
    expect(screen.getByText('gpt-5.6-sol')).toBeInTheDocument();
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

    expect(screen.getByText('my-endpoint')).toBeInTheDocument();

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

    expect(screen.getByText('gpt-5.6-sol')).toBeInTheDocument();

    await changeModelToAnthropic();
    expect(screen.getByText('claude-opus-4-8')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);
    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ provider: 'anthropic', model: 'claude-opus-4-8' }),
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

  test('submits every category by default, with the saved secret for direct providers', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({
          categories: ['correctness', 'latency', 'execution', 'adherence', 'relevance', 'safety', 'cost'],
          provider: 'openai',
          model: 'gpt-5.6-sol',
          secret_id: 'secret-123',
          endpoint_name: undefined,
        }),
        expect.any(Object),
      );
    });
  });

  test("submits only the experiment's selected focus areas", async () => {
    // The settings exist to narrow a run; a modal that kept sending every category would make
    // saving them look like it worked while changing nothing.
    mockSettings({
      instructions: '',
      categories: {
        correctness: true,
        latency: false,
        execution: false,
        adherence: false,
        relevance: false,
        safety: true,
        cost: false,
      },
    });
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ categories: ['correctness', 'safety'] }),
        expect.any(Object),
      );
    });
  });

  test("submits the experiment's saved detection priorities", async () => {
    mockSettings({ ...defaultIssueDetectionSettings(), instructions: 'Only report high severity issues.' });
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ instructions: 'Only report high severity issues.' }),
        expect.any(Object),
      );
    });
  });

  test('sends no priorities when the experiment has none, rather than an empty string', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ instructions: undefined }),
        expect.any(Object),
      );
    });
  });

  test('analysis cannot start while the settings are still loading', async () => {
    // Reported in review: submitting here would silently use the defaults and drop the saved
    // priorities -- an expensive run whose results ignore the experiment's configuration.
    jest.mocked(useIssueDetectionSettings).mockReturnValue({ data: undefined, isLoading: true, error: null } as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(screen.getByText('Run Analysis').closest('button')).toBeDisabled();
    expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
  });

  test('saved settings that select nothing disclose the fallback where the run is started', async () => {
    // Raised in review: the stored value and the sent value differ in this one case, and the
    // settings modal is not open when someone launches a run a week after saving.
    mockSettings({
      instructions: '',
      categories: {
        correctness: false,
        latency: false,
        execution: false,
        adherence: false,
        relevance: false,
        safety: false,
        cost: false,
      },
    });
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(
      screen.getByText(
        "This experiment's issue detection settings have no focus area selected, so this run considers all of them.",
      ),
    ).toBeInTheDocument();

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({
          categories: ['correctness', 'latency', 'execution', 'adherence', 'relevance', 'safety', 'cost'],
        }),
        expect.any(Object),
      );
    });
  });

  test('ordinary settings do not claim a fallback that is not happening', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(screen.queryByText(/no focus area selected/)).not.toBeInTheDocument();
  });

  test('a failed refresh over loaded settings says what the run will actually use', async () => {
    // React Query keeps the cached data alongside a failed background refetch, and the run is
    // submitted with it -- so claiming "the defaults will be used" here would be false.
    jest.mocked(useIssueDetectionSettings).mockReturnValue({
      data: { instructions: 'Only report high severity.', categories: { safety: true } },
      isLoading: false,
      error: new Error('nope'),
    } as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(screen.getByText(/could not be refreshed, so this run uses the last ones loaded/)).toBeInTheDocument();
    expect(screen.queryByText(/will use the defaults/)).not.toBeInTheDocument();

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({
          categories: ['safety'],
          instructions: 'Only report high severity.',
        }),
        expect.any(Object),
      );
    });
  });

  test('settings that failed to load say so rather than quietly using the defaults', async () => {
    // Blocking on a tag read would strand a feature that worked before settings existed, so
    // the run is still allowed -- but not silently.
    jest
      .mocked(useIssueDetectionSettings)
      .mockReturnValue({ data: undefined, isLoading: false, error: new Error('nope') } as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    expect(
      screen.getByText(
        "This experiment's issue detection settings could not be read, so this run will use the defaults: all focus areas and no saved priorities.",
      ),
    ).toBeInTheDocument();

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({
          categories: ['correctness', 'latency', 'execution', 'adherence', 'relevance', 'safety', 'cost'],
          instructions: undefined,
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

  test('records submitted background job when form is submitted', async () => {
    const onClose = jest.fn(() => expect(getSubmittedIssueDetectionJob()?.jobId).toBe('job-123'));
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await userEvent.click(screen.getByText('Run Analysis').closest('button')!);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });

    expect(getSubmittedIssueDetectionJob()).toEqual({
      experimentId: 'exp-123',
      workspace: null,
      jobId: 'job-123',
      runId: 'run-456',
      traceCount: 1,
      submittedAtMs: expect.any(Number),
    });
  });

  test('passes defaultGroupBySession prop to SelectTracesModal when set to true', async () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} defaultGroupBySession />,
    );

    await userEvent.click(screen.getByTestId('traces-card'));

    expect(screen.getByTestId('default-group-by-session')).toHaveTextContent('true');
  });

  test('passes defaultGroupBySession prop to SelectTracesModal when set to false', async () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} defaultGroupBySession={false} />,
    );

    await userEvent.click(screen.getByTestId('traces-card'));

    expect(screen.getByTestId('default-group-by-session')).toHaveTextContent('false');
  });
});
