import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModal } from './IssueDetectionModal';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';

jest.mock('../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration');
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

describe('IssueDetectionModal', () => {
  const defaultProps = {
    visible: true,
    onClose: jest.fn(),
    experimentId: 'exp-123',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useApiKeyConfiguration).mockReturnValue({
      existingSecrets: [],
      isLoadingSecrets: false,
      authModes: [],
      defaultAuthMode: undefined,
      selectedAuthMode: undefined,
      hasExistingSecrets: false,
      isLoadingProviderConfig: false,
    });
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

  test('submit button is enabled when form is complete with existing key', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('submit button is enabled when form is complete with new key', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');
    await userEvent.selectOptions(screen.getByTestId('model-select'), 'gpt-4');
    await userEvent.click(screen.getByTestId('set-new-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('shows save key checkbox when using new key mode', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('Save this key for reuse')).toBeInTheDocument();
  });

  test('hides save key checkbox when switching to existing key', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

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

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} onClose={onClose} />);

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

    expect(screen.getByText('Save this key for reuse')).toBeInTheDocument();
  });
});
