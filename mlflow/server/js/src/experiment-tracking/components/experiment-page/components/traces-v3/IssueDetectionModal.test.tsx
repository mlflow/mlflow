import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModal } from './IssueDetectionModal';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';

jest.mock('../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration');
jest.mock('../../../../../gateway/hooks/useCreateSecret');
jest.mock('../../../../../gateway/components/create-endpoint/ProviderSelect', () => ({
  ProviderSelect: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <select data-testid="provider-select" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">Select provider</option>
      <option value="openai">OpenAI</option>
      <option value="anthropic">Anthropic</option>
    </select>
  ),
}));
jest.mock('../../../../../gateway/components/create-endpoint/ModelSelect', () => ({
  ModelSelect: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <select data-testid="model-select" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">Select model</option>
      <option value="gpt-4">gpt-4</option>
      <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
    </select>
  ),
}));
jest.mock('../../../../../gateway/components/model-configuration/components/ApiKeyConfigurator', () => ({
  ApiKeyConfigurator: ({
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
    visible: true,
    onClose: jest.fn(),
    experimentId: 'exp-123',
  };

  let mockCreateSecret: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    mockCreateSecret = jest.fn(() => Promise.resolve());
    jest.mocked(useApiKeyConfiguration).mockReturnValue({
      existingSecrets: [],
      isLoadingSecrets: false,
      authModes: [],
      defaultAuthMode: undefined,
      selectedAuthMode: undefined,
      hasExistingSecrets: false,
      isLoadingProviderConfig: false,
    });
    jest.mocked(useCreateSecret).mockReturnValue({
      mutateAsync: mockCreateSecret,
    } as any);
  });

  test('renders modal when visible', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('Detect Issues')).toBeInTheDocument();
  });

  test('does not render modal when not visible', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} visible={false} />);

    expect(screen.queryByText('Detect Issues')).not.toBeInTheDocument();
  });

  test('renders description text', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('Connect an LLM to run an AI-powered issue analysis on your traces')).toBeInTheDocument();
  });

  test('renders model selection section', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByTestId('provider-select')).toBeInTheDocument();
    expect(screen.getByTestId('model-select')).toBeInTheDocument();
  });

  test('renders connections section', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('Connections')).toBeInTheDocument();
    expect(screen.getByTestId('api-key-configurator')).toBeInTheDocument();
  });

  test('submit button is disabled when form is incomplete', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).toBeDisabled();
  });

  test('submit button is disabled without traces selected', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).toBeDisabled();
  });

  test('submit button is enabled when form is complete with existing key and traces', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('submit button is enabled when form is complete with new key and traces', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-new-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('does not show save key checkbox without provider selected', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.queryByText('Save this key for reuse')).not.toBeInTheDocument();
  });

  test('shows save key checkbox when provider selected and using new key mode', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-new-key'));

    expect(screen.getByText('Save this key for reuse')).toBeInTheDocument();
  });

  test('hides save key checkbox when switching to existing key', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.click(screen.getByTestId('set-new-key'));
    expect(screen.getByText('Save this key for reuse')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('set-existing-key'));
    expect(screen.queryByText('Save this key for reuse')).not.toBeInTheDocument();
  });

  test('calls onClose when cancel button is clicked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const onClose = jest.fn();

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} onClose={onClose} />);

    const cancelButton = screen.getByRole('button', { name: /close/i });
    await userEvent.click(cancelButton);

    expect(onClose).toHaveBeenCalled();
  });

  test('calls onClose and resets form when submit is clicked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
  });

  test('resets api key config when provider changes', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    expect(screen.queryByText('Save this key for reuse')).not.toBeInTheDocument();

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'anthropic');
    await userEvent.click(screen.getByTestId('set-new-key'));

    expect(screen.getByText('Save this key for reuse')).toBeInTheDocument();
  });

  test('renders traces selection section', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('Traces')).toBeInTheDocument();
    expect(screen.getByText('Select the traces to analyze for issues')).toBeInTheDocument();
    expect(screen.getByText('Select traces')).toBeInTheDocument();
  });

  test('shows trace count when initial traces are provided', () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1', 'trace-2', 'trace-3']} />,
    );

    expect(screen.getByText('3 traces selected')).toBeInTheDocument();
  });

  test('opens select traces modal when button is clicked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Select traces'));

    expect(screen.getByTestId('select-traces-modal')).toBeInTheDocument();
  });

  test('updates trace count after selecting traces', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Select traces'));
    await userEvent.click(screen.getByTestId('select-traces-confirm'));

    expect(screen.getByText('2 traces selected')).toBeInTheDocument();
  });

  test('closes select traces modal when cancel is clicked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Select traces'));
    expect(screen.getByTestId('select-traces-modal')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('select-traces-cancel'));
    expect(screen.queryByTestId('select-traces-modal')).not.toBeInTheDocument();
  });

  test('saves secret when save key checkbox is checked and form is submitted with new key', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-new-key'));
    await userEvent.click(screen.getByText('Save this key for reuse'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(mockCreateSecret).toHaveBeenCalledWith({
        secret_name: 'my-key',
        secret_value: { api_key: 'sk-123' },
        provider: 'openai',
        auth_config: undefined,
      });
    });
  });

  test('does not save secret when save key checkbox is not checked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-new-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
    expect(mockCreateSecret).not.toHaveBeenCalled();
  });

  test('does not save secret when using existing key', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
    expect(mockCreateSecret).not.toHaveBeenCalled();
  });
});
