import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { useExperimentPermission } from '../../../../hooks/useExperimentPermission';
import { IssueDetectionModal } from './IssueDetectionModal';
import { useInvokeIssueDetection } from './hooks/useInvokeIssueDetection';
import { clearSubmittedIssueDetectionJob, getSubmittedIssueDetectionJob } from './IssueDetectionJobNotifications';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { useModelsQuery } from '../../../../../gateway/hooks/useModelsQuery';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import { useIssueDetectionSettings } from '../../../../pages/experiment-issues/useIssueDetectionSettings';
import { defaultIssueDetectionSettings } from '../../../../pages/experiment-issues/issueDetectionSettings';
import { useHasExperimentIssues } from '../../../../pages/experiment-issues/useExperimentIssues';
import { useNormalizeIssueDetectionTraceScope } from '../../../../pages/experiment-issues/IssueDetectionTraceScope';
import { InvalidIssueDetectionTraceScopeError } from '../../../../pages/experiment-issues/issueDetectionTraceScope';
import { useIssueDetectionTraceScope } from '../../../../pages/experiment-issues/useIssueDetectionTraceScope';
import { searchMlflowTracesQueryFn } from '../../../../../shared/web-shared/genai-traces-table/hooks/useMlflowTraces';

jest.mock('./hooks/useInvokeIssueDetection');
jest.mock('../../../../hooks/useExperimentPermission');
jest.mock('../../../../pages/experiment-issues/useExperimentIssues', () => ({ useHasExperimentIssues: jest.fn() }));
jest.mock('../../../../../shared/web-shared/genai-traces-table/hooks/useMlflowTraces', () => ({
  searchMlflowTracesQueryFn: jest.fn(),
}));
jest.mock('../../../../pages/experiment-issues/useIssueDetectionTraceScope', () => ({
  useIssueDetectionTraceScope: jest.fn(),
}));
jest.mock('../../../../pages/experiment-issues/IssueDetectionTraceScope', () => ({
  useNormalizeIssueDetectionTraceScope: jest.fn(() => (scope: unknown) => scope),
  IssueDetectionTraceScope: ({ scope, onChange }: any) => (
    <input
      aria-label="Target traces"
      value={scope.searchQuery}
      onChange={(event) => onChange({ searchQuery: event.target.value, filters: scope.filters })}
    />
  ),
}));

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
    jest.mocked(useNormalizeIssueDetectionTraceScope).mockReturnValue((scope) => scope);
    jest.mocked(useExperimentPermission).mockReturnValue(true);
    clearSubmittedIssueDetectionJob();
    jest.mocked(searchMlflowTracesQueryFn).mockReset();
    mockSettings();
    jest
      .mocked(useHasExperimentIssues)
      .mockReturnValue({ data: { issues: [{ issue_id: 'existing' }] }, isLoading: false, error: null } as any);
    jest.mocked(useIssueDetectionTraceScope).mockReturnValue({
      data: { searchQuery: '', filters: [] },
      isLoading: false,
      error: null,
      refetch: jest.fn(),
      save: { mutateAsync: jest.fn() },
    } as any);
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

  test.each([
    ['  Only high severity.  ', 'Only high severity.'],
    ['   ', undefined],
  ])('run priorities are normalized like persisted settings: %s', async (instructions, expected) => {
    mockSettings({ ...defaultIssueDetectionSettings(), instructions });
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await waitFor(() =>
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ instructions: expected }),
        expect.any(Object),
      ),
    );
  });

  test.each([true, false])('invalid scope recovery is explicit and edit-gated (canEdit=%s)', async (canEdit) => {
    jest.mocked(useExperimentPermission).mockReturnValue(canEdit);
    const reset = jest.fn();
    jest.mocked(useIssueDetectionTraceScope).mockReturnValue({
      error: new InvalidIssueDetectionTraceScopeError('Saved scope is invalid; reset before analysis.'),
      isLoading: false,
      save: { mutate: reset },
      refetch: jest.fn(),
    } as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    expect(screen.getByText('Saved scope is invalid; reset before analysis.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run Analysis' })).toBeDisabled();
    const button = screen.queryByRole('button', { name: 'Reset trace scope to all traces' });
    if (canEdit) {
      await userEvent.click(button!);
      expect(reset).toHaveBeenCalledWith({ searchQuery: '', filters: [] });
    } else expect(button).not.toBeInTheDocument();
    expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
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
        'Previously saved settings have no focus area selected. This run considers all areas; select at least one when editing settings.',
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

  test.each([undefined, { instructions: 'Only report high severity.', categories: { safety: true } }])(
    'an unreadable configuration blocks analysis even with cached settings %j',
    async (data) => {
      const refetch = jest.fn();
      jest
        .mocked(useIssueDetectionSettings)
        .mockReturnValue({ data, isLoading: false, error: new Error('offline'), refetch } as any);
      renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
      expect(
        screen.getByText('Could not load issue detection settings. Retry before starting analysis.'),
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Run Analysis' })).toBeDisabled();
      await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
      expect(refetch).toHaveBeenCalledTimes(1);
      expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
    },
  );

  test('customization keeps a draft until Run and Cancel discards edits', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Customize' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Detection priorities' }), 'Unsaved priorities');
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await userEvent.click(screen.getByRole('button', { name: 'Customize' }));
    expect(screen.getByRole('textbox', { name: 'Detection priorities' })).toHaveValue('');
    expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
  });

  test.each([null, new Error('history offline')])(
    'first runs offer setup and defaults make no settings writes with advisory history %s',
    async (error) => {
      jest.mocked(useHasExperimentIssues).mockReturnValue({ data: { issues: [] }, isLoading: false, error } as any);
      const save = jest.fn();
      jest.mocked(useIssueDetectionSettings).mockReturnValue({
        data: defaultIssueDetectionSettings(),
        isLoading: false,
        error: null,
        save: { mutateAsync: save },
      } as any);
      renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
      expect(await screen.findByRole('textbox', { name: 'Detection priorities' })).toBeInTheDocument();
      await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
      await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
      await waitFor(() => expect(mockInvokeIssueDetection).toHaveBeenCalledTimes(1));
      expect(save).not.toHaveBeenCalled();
      expect(searchMlflowTracesQueryFn).not.toHaveBeenCalled();
    },
  );

  test('invalid customized scope never persists priorities before rejecting the run', async () => {
    const save = jest.fn();
    jest.mocked(useIssueDetectionSettings).mockReturnValue({
      data: defaultIssueDetectionSettings(),
      isLoading: false,
      error: null,
      save: { mutateAsync: save },
    } as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Customize' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Detection priorities' }), 'New priorities');
    await userEvent.type(screen.getByRole('textbox', { name: 'Target traces' }), "can't");
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await screen.findByText('Single quotes are not supported in trace filter values or text search.');
    expect(save).not.toHaveBeenCalled();
    expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
  });

  test('equivalent applied scope and trimmed priorities do not rewrite tags on retry', async () => {
    const saveSettings = jest.fn();
    const saveScope = jest.fn();
    const data = { searchQuery: 'feedback', filters: [], filterString: "trace.text ILIKE '%feedback%'" };
    jest.mocked(useIssueDetectionSettings).mockReturnValue({
      data: { ...defaultIssueDetectionSettings(), instructions: 'Same priorities' },
      isLoading: false,
      error: null,
      save: { mutateAsync: saveSettings },
    } as any);
    jest
      .mocked(useIssueDetectionTraceScope)
      .mockReturnValue({ data, isLoading: false, error: null, save: { mutateAsync: saveScope } } as any);
    jest
      .mocked(useNormalizeIssueDetectionTraceScope)
      .mockReturnValue(({ searchQuery, filters }) => ({ searchQuery, filters }));
    jest.mocked(searchMlflowTracesQueryFn).mockResolvedValue([{ trace_id: 'trace-1' }] as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Customize' }));
    const priorities = screen.getByRole('textbox', { name: 'Detection priorities' });
    await userEvent.clear(priorities);
    await userEvent.type(priorities, '  Same priorities  ');
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await waitFor(() => expect(mockInvokeIssueDetection).toHaveBeenCalledTimes(1));
    expect(saveSettings).not.toHaveBeenCalled();
    expect(saveScope).not.toHaveBeenCalled();
  });

  test('saved scope intersects an explicit selection before starting analysis', async () => {
    jest
      .mocked(useIssueDetectionTraceScope)
      .mockReturnValue({ data: { searchQuery: 'feedback', filters: [] }, isLoading: false, error: null } as any);
    jest.mocked(searchMlflowTracesQueryFn).mockResolvedValue([{ trace_id: 'trace-1' }] as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1', 'trace-2']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await waitFor(() =>
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ traceIds: ['trace-1'] }),
        expect.any(Object),
      ),
    );
    expect(searchMlflowTracesQueryFn).toHaveBeenCalledWith(
      expect.objectContaining({
        filter: "attributes.request_id IN ('trace-1','trace-2') AND trace.text ILIKE '%feedback%'",
      }),
    );
  });

  test('saved scope re-searches up to 50 recent traces without an explicit selection', async () => {
    jest
      .mocked(useIssueDetectionTraceScope)
      .mockReturnValue({ data: { searchQuery: 'feedback', filters: [] }, isLoading: false } as any);
    jest.mocked(searchMlflowTracesQueryFn).mockResolvedValue([{ trace_id: 'matching-trace' }] as any);
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} availableTraceIds={Array.from({ length: 60 }, (_, i) => `trace-${i}`)} />,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await waitFor(() =>
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ traceIds: ['matching-trace'] }),
        expect.any(Object),
      ),
    );
    expect(searchMlflowTracesQueryFn).toHaveBeenCalledWith(
      expect.objectContaining({
        filter: "trace.text ILIKE '%feedback%'",
        pageSize: 50,
        limit: 50,
        orderBy: ['timestamp DESC'],
      }),
    );
  });

  test('an empty saved-scope intersection blocks the run with an inline error', async () => {
    jest
      .mocked(useIssueDetectionTraceScope)
      .mockReturnValue({ data: { searchQuery: 'feedback', filters: [] }, isLoading: false } as any);
    jest.mocked(searchMlflowTracesQueryFn).mockResolvedValue([]);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    expect(await screen.findByText(/No selected traces match the saved trace scope/)).toBeInTheDocument();
    expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
  });

  test('Apply normalizes a complete structured query without Enter before saving or launching', async () => {
    const actual = jest.requireActual<typeof import('../../../../pages/experiment-issues/IssueDetectionTraceScope')>(
      '../../../../pages/experiment-issues/IssueDetectionTraceScope',
    );
    jest.mocked(useNormalizeIssueDetectionTraceScope).mockImplementation(actual.useNormalizeIssueDetectionTraceScope);
    const save = jest.fn<any>().mockResolvedValue({ searchQuery: '', filters: [] });
    jest.mocked(useIssueDetectionTraceScope).mockReturnValue({
      data: { searchQuery: '', filters: [] },
      isLoading: false,
      save: { mutateAsync: save },
    } as any);
    jest.mocked(searchMlflowTracesQueryFn).mockResolvedValue([{ trace_id: 'trace-1' }] as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Customize' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Target traces' }), `tag."team" = 'support'`);
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith({
        searchQuery: '',
        filters: [{ field: 'tag', key: 'team', operator: '=', value: 'support' }],
      }),
    );
    await waitFor(() => expect(mockInvokeIssueDetection).toHaveBeenCalledTimes(1));
  });

  test('a failed customization save keeps the draft and permits retry without starting a run', async () => {
    const save = jest
      .fn<any>()
      .mockRejectedValueOnce(new Error('save failed'))
      .mockResolvedValue(defaultIssueDetectionSettings());
    jest.mocked(useIssueDetectionSettings).mockReturnValue({
      data: defaultIssueDetectionSettings(),
      isLoading: false,
      error: null,
      save: { mutateAsync: save },
    } as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Customize' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Detection priorities' }), 'Focus on feedback');
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    expect(await screen.findByText('save failed')).toBeInTheDocument();
    expect(mockInvokeIssueDetection).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await waitFor(() =>
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ instructions: 'Focus on feedback' }),
        expect.any(Object),
      ),
    );
  });

  test('read-only setup uses saved settings without writes and disables editing', async () => {
    jest.mocked(useExperimentPermission).mockReturnValue(false);
    jest.mocked(useHasExperimentIssues).mockReturnValue({ data: { issues: [] }, isLoading: false } as any);
    const save = jest.fn();
    const saveScope = jest.fn();
    jest.mocked(useIssueDetectionTraceScope).mockReturnValue({
      data: { searchQuery: '', filters: [] },
      isLoading: false,
      error: null,
      save: { mutateAsync: saveScope },
    } as any);
    const settings = { ...defaultIssueDetectionSettings(), instructions: 'Saved priorities' };
    jest.mocked(useIssueDetectionSettings).mockReturnValue({
      data: settings,
      isLoading: false,
      error: null,
      save: { mutateAsync: save },
    } as any);
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);
    expect(await screen.findByRole('textbox', { name: 'Detection priorities' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(screen.getByRole('button', { name: 'Customize' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await waitFor(() =>
      expect(mockInvokeIssueDetection).toHaveBeenCalledWith(
        expect.objectContaining({ instructions: 'Saved priorities' }),
        expect.any(Object),
      ),
    );
    expect(save).not.toHaveBeenCalled();
    expect(saveScope).not.toHaveBeenCalled();
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
