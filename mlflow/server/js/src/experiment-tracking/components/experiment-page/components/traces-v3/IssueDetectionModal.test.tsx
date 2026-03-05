import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModal } from './IssueDetectionModal';

jest.mock('../../../../../gateway/components/create-endpoint/ProviderSelect', () => ({
  ProviderSelect: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <select data-testid="provider-select" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">Select provider</option>
      <option value="openai">OpenAI</option>
      <option value="anthropic">Anthropic</option>
    </select>
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

  test('renders provider selection', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByTestId('provider-select')).toBeInTheDocument();
  });

  test('shows default models when provider with defaults is selected', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');

    expect(screen.getByText(/Analysis model: gpt-5 · Judge model: gpt-5-mini/)).toBeInTheDocument();
  });

  test('shows message when provider without defaults is selected', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    // Add a provider option without defaults
    const providerSelect = screen.getByTestId('provider-select');
    const newOption = document.createElement('option');
    newOption.value = 'other-provider';
    newOption.textContent = 'Other Provider';
    providerSelect.appendChild(newOption);

    await userEvent.selectOptions(providerSelect, 'other-provider');

    expect(screen.getByText(/Please select models in `Advanced settings` below/)).toBeInTheDocument();
  });

  test('submit button is disabled when form is incomplete', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).toBeDisabled();
  });

  test('submit button is enabled when provider is selected', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
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

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
  });
});
