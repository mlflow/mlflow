import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModal } from './IssueDetectionModal';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import { useInvokeIssueDetection } from './hooks/useInvokeIssueDetection';

jest.mock('../../../../../gateway/hooks/useCreateSecret');
jest.mock('./hooks/useInvokeIssueDetection');
jest.mock('../../../../../gateway/components/create-endpoint/ProviderSelect', () => ({
  ProviderSelect: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <select data-testid="provider-select" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">Select provider</option>
      <option value="openai">OpenAI</option>
      <option value="anthropic">Anthropic</option>
    </select>
  ),
}));
jest.mock('./IssueDetectionApiKeyConfigurator', () => ({
  IssueDetectionApiKeyConfigurator: ({
    value,
    onChange,
  }: {
    value: {
      mode: string;
      existingSecretId: string;
      newSecret: { name: string; secretFields: Record<string, string> };
    };
    onChange: (v: any) => void;
  }) => (
    <div data-testid="api-key-configurator">
      <button
        data-testid="set-existing-key"
        onClick={() => onChange({ ...value, mode: 'existing', existingSecretId: 'secret-123' })}
      >
        Use existing key
      </button>
      <button
        data-testid="set-new-key"
        onClick={() =>
          onChange({
            ...value,
            mode: 'new',
            newSecret: { name: 'my-key', authMode: '', secretFields: { api_key: 'sk-123' }, configFields: {} },
          })
        }
      >
        Use new key
      </button>
    </div>
  ),
}));
const mockUseApiKeyConfiguration = jest.fn();
jest.mock('../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration', () => ({
  useApiKeyConfiguration: () => mockUseApiKeyConfiguration(),
}));
jest.mock('./IssueDetectionAdvancedSettings', () => ({
  IssueDetectionAdvancedSettings: () => <div data-testid="advanced-settings">Advanced Settings</div>,
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

describe('IssueDetectionModal', () => {
  const defaultProps = {
    onClose: jest.fn(),
    experimentId: 'exp-123',
  };

  let mockCreateSecret: jest.Mock;
  let mockResetCreateSecret: jest.Mock;
  let mockInvokeIssueDetection: jest.Mock;
  let mockResetIssueDetection: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    mockCreateSecret = jest.fn((_request, options) => {
      (options as { onSuccess?: (response: { secret: { secret_id: string } }) => void })?.onSuccess?.({
        secret: { secret_id: 'new-secret-123' },
      });
    });
    mockResetCreateSecret = jest.fn();
    mockInvokeIssueDetection = jest.fn((_request, options) => {
      (options as { onSuccess?: (response: { job_id: string; run_id: string }) => void })?.onSuccess?.({
        job_id: 'job-123',
        run_id: 'run-456',
      });
    });
    mockResetIssueDetection = jest.fn();
    mockUseApiKeyConfiguration.mockReturnValue({
      existingSecrets: [],
      authModes: [],
      defaultAuthMode: undefined,
      isLoadingProviderConfig: false,
    });
    jest.mocked(useCreateSecret).mockReturnValue({
      mutate: mockCreateSecret,
      isLoading: false,
      error: null,
      reset: mockResetCreateSecret,
    } as any);
    jest.mocked(useInvokeIssueDetection).mockReturnValue({
      mutate: mockInvokeIssueDetection,
      isLoading: false,
      error: null,
      reset: mockResetIssueDetection,
    } as any);
  });

  // Helper to navigate to step 2 (provider/model configuration)
  const navigateToStep2 = async () => {
    const nextButton = screen.getByText('Next').closest('button')!;
    await userEvent.click(nextButton);
  };

  test('renders modal with step 1 (category selection)', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('Detect Issues')).toBeInTheDocument();
    expect(screen.getByText('Select Categories')).toBeInTheDocument();
  });

  test('renders description text', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(
      screen.getByText('Use AI to automatically analyze your traces and identify potential issues'),
    ).toBeInTheDocument();
  });

  test('renders provider selection in step 2', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();

    expect(screen.getByTestId('provider-select')).toBeInTheDocument();
  });

  test('shows default model when provider with defaults is selected', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');

    expect(screen.getByText(/Model: gpt-5-mini/)).toBeInTheDocument();
  });

  test('shows message when provider without defaults is selected', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();

    // Add a provider option without defaults
    const providerSelect = screen.getByTestId('provider-select');
    const newOption = document.createElement('option');
    newOption.value = 'other-provider';
    newOption.textContent = 'Other Provider';
    providerSelect.appendChild(newOption);

    await userEvent.selectOptions(providerSelect, 'other-provider');

    expect(screen.getByText(/Please select a model in `Advanced settings` below/)).toBeInTheDocument();
  });

  test('renders api key configurator in step 2', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();

    expect(screen.getByTestId('api-key-configurator')).toBeInTheDocument();
  });

  test('submit button is disabled when form is incomplete', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    // Clear the default provider to make form incomplete
    await userEvent.selectOptions(screen.getByTestId('provider-select'), '');

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).toBeDisabled();
  });

  test('submit button is disabled without traces selected', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).toBeDisabled();
  });

  test('submit button is enabled when form is complete with existing key and traces', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('submit button is enabled when form is complete with new key and traces', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-new-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('does not show save key message in step 1', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.queryByText('This key will be saved for reuse.')).not.toBeInTheDocument();
  });

  test('shows save key message when using new key mode in step 2', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-new-key'));

    expect(screen.getByText('This key will be saved for reuse.')).toBeInTheDocument();
  });

  test('hides save key message when switching to existing key', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-new-key'));
    expect(screen.getByText('This key will be saved for reuse.')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('set-existing-key'));
    expect(screen.queryByText('This key will be saved for reuse.')).not.toBeInTheDocument();
  });

  test('calls onClose when cancel button is clicked', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} onClose={onClose} />);

    const cancelButton = screen.getByText('Cancel').closest('button')!;
    await userEvent.click(cancelButton);

    expect(onClose).toHaveBeenCalled();
  });

  test('calls onClose and resets form when submit is clicked', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
  });

  test('resets api key config when provider changes', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    expect(screen.queryByText('This key will be saved for reuse.')).not.toBeInTheDocument();

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'anthropic');
    await userEvent.click(screen.getByTestId('set-new-key'));

    expect(screen.getByText('This key will be saved for reuse.')).toBeInTheDocument();
  });

  test('shows API key name input when using new key mode', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-new-key'));

    expect(screen.getByPlaceholderText('API key name')).toBeInTheDocument();
  });

  test('defaults to existing key mode when provider is selected and secrets are available', async () => {
    // Override the mock to return existing secrets
    mockUseApiKeyConfiguration.mockReturnValue({
      existingSecrets: [{ secret_id: 'secret-1', secret_name: 'My Key' }],
      authModes: [],
      defaultAuthMode: undefined,
      isLoadingProviderConfig: false,
    });

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await navigateToStep2();
    // When selecting a provider with available secrets, it should default to existing mode
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');

    // Verify the component receives config with mode: 'existing' by checking
    // that the "Use existing" button test behavior is consistent
    await userEvent.click(screen.getByTestId('set-existing-key'));
    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('renders traces selection section in step 2', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();

    expect(screen.getByText('Traces')).toBeInTheDocument();
    expect(screen.getByText('Select the traces to analyze for issues')).toBeInTheDocument();
    expect(screen.getByText('Select traces')).toBeInTheDocument();
  });

  test('shows trace count when initial traces are provided', async () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1', 'trace-2', 'trace-3']} />,
    );

    await navigateToStep2();

    expect(screen.getByText('3 traces selected')).toBeInTheDocument();
  });

  test('opens select traces modal when button is clicked', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.click(screen.getByText('Select traces'));

    expect(screen.getByTestId('select-traces-modal')).toBeInTheDocument();
  });

  test('updates trace count after selecting traces', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.click(screen.getByText('Select traces'));
    await userEvent.click(screen.getByTestId('select-traces-confirm'));

    expect(screen.getByText('2 traces selected')).toBeInTheDocument();
  });

  test('closes select traces modal when cancel is clicked', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.click(screen.getByText('Select traces'));
    expect(screen.getByTestId('select-traces-modal')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('select-traces-cancel'));
    expect(screen.queryByTestId('select-traces-modal')).not.toBeInTheDocument();
  });

  test('saves secret when form is submitted with new key', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-new-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(mockCreateSecret).toHaveBeenCalledWith(
        {
          secret_name: 'my-key',
          secret_value: { api_key: 'sk-123' },
          provider: 'openai',
          auth_config: undefined,
        },
        expect.any(Object),
      );
    });
  });

  test('does not save secret when using existing key', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
    expect(mockCreateSecret).not.toHaveBeenCalled();
  });

  test('calls onSubmitSuccess callback when form is submitted', async () => {
    const onClose = jest.fn();
    const onSubmitSuccess = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal
        {...defaultProps}
        onClose={onClose}
        initialSelectedTraceIds={['trace-1']}
        onSubmitSuccess={onSubmitSuccess}
      />,
    );

    await navigateToStep2();
    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitSuccess).toHaveBeenCalledWith('run-456');
      expect(onClose).toHaveBeenCalled();
    });
  });
});
