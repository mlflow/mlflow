import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { CustomGuardrailForm } from './CustomGuardrailForm';

jest.mock('../../api', () => ({
  GatewayApi: {
    createGuardrail: jest.fn<any>().mockResolvedValue({
      guardrail: { guardrail_id: 'g-new123' },
    }),
    addGuardrailToEndpoint: jest.fn<any>().mockResolvedValue({ config: {} }),
  },
}));

jest.mock('../../../experiment-tracking/pages/experiment-scorers/api', () => ({
  registerScorer: jest.fn<any>().mockResolvedValue({
    scorer_id: 'sc-custom123',
    version: 1,
    name: 'pii-check',
  }),
}));

jest.mock('../../../experiment-tracking/components/EndpointSelector', () => ({
  EndpointSelector: ({ onEndpointSelect }: any) => (
    <button type="button" onClick={() => onEndpointSelect('my-model-endpoint')}>
      Mock Endpoint Selector
    </button>
  ),
}));

describe('CustomGuardrailForm', () => {
  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    onBack: jest.fn(),
    onSuccess: jest.fn(),
    endpointId: 'e-123',
    experimentId: '0',
    stage: 'BEFORE' as const,
    action: 'VALIDATION' as const,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders form fields when open', () => {
    renderWithDesignSystem(<CustomGuardrailForm {...defaultProps} />);

    expect(screen.getByText('Create Custom Guardrail')).toBeInTheDocument();
    expect(screen.getByText('Guardrail name')).toBeInTheDocument();
    expect(screen.getByText('Judge instructions')).toBeInTheDocument();
    expect(screen.getByText('Judge model (optional)')).toBeInTheDocument();
  });

  test('Create button is disabled when form is empty', () => {
    renderWithDesignSystem(<CustomGuardrailForm {...defaultProps} />);

    const createButton = screen.getByRole('button', { name: 'Create' });
    expect(createButton).toBeDisabled();
  });

  test('Create button is enabled when name and prompt are filled', async () => {
    renderWithDesignSystem(<CustomGuardrailForm {...defaultProps} />);

    const nameInput = screen.getByPlaceholderText('e.g., pii-detector');
    await userEvent.type(nameInput, 'my-guardrail');

    const promptInput = screen.getByPlaceholderText(/Check if the text contains/);
    await userEvent.type(promptInput, 'Block harmful content');

    const createButton = screen.getByRole('button', { name: 'Create' });
    expect(createButton).not.toBeDisabled();
  });

  test('Back button calls onBack', async () => {
    renderWithDesignSystem(<CustomGuardrailForm {...defaultProps} />);

    const backButton = screen.getByRole('button', { name: /Back/ });
    await userEvent.click(backButton);

    expect(defaultProps.onBack).toHaveBeenCalled();
  });

  test('does not render when closed', () => {
    renderWithDesignSystem(<CustomGuardrailForm {...defaultProps} open={false} />);
    expect(screen.queryByText('Create Custom Guardrail')).not.toBeInTheDocument();
  });

  test('submits form: registers scorer, creates guardrail, adds to endpoint', async () => {
    const { GatewayApi } = await import('../../api');
    const { registerScorer } = await import('../../../experiment-tracking/pages/experiment-scorers/api');

    renderWithDesignSystem(<CustomGuardrailForm {...defaultProps} />);

    await userEvent.type(screen.getByPlaceholderText('e.g., pii-detector'), 'pii-check');
    await userEvent.type(screen.getByPlaceholderText(/Check if the text contains/), 'Detect PII');

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    // Step 1: Registers the scorer
    expect(registerScorer).toHaveBeenCalledWith('0', expect.objectContaining({ name: 'pii-check' }));

    // Step 2: Creates the guardrail with the registered scorer_id
    expect(GatewayApi.createGuardrail).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'pii-check',
        scorer_id: 'sc-custom123',
        scorer_version: 1,
        stage: 'BEFORE',
        action: 'VALIDATION',
      }),
    );

    // Step 3: Adds to endpoint
    expect(GatewayApi.addGuardrailToEndpoint).toHaveBeenCalledWith({
      endpoint_id: 'e-123',
      guardrail_id: 'g-new123',
    });

    expect(defaultProps.onSuccess).toHaveBeenCalled();
    expect(defaultProps.onClose).toHaveBeenCalled();
  });
});
