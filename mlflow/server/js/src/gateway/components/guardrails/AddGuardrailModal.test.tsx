import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { GuardrailModal } from './AddGuardrailModal';
import { useAddGuardrail } from '../../hooks/useAddGuardrail';
import { GatewayApi } from '../../api';

jest.mock('../../hooks/useAddGuardrail');
jest.mock('../../api', () => ({
  GatewayApi: {
    addGuardrailToEndpoint: jest.fn(),
  },
}));

const mockRegisterScorer = jest.fn<any>();
jest.mock('../../../experiment-tracking/pages/experiment-scorers/api', () => ({
  registerScorer: (...args: any[]) => mockRegisterScorer(...args),
}));

const mockCreateGuardrail = jest.fn<any>();

describe('GuardrailModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useAddGuardrail).mockReturnValue({
      mutateAsync: mockCreateGuardrail,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
  });

  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    onSuccess: jest.fn(),
    endpointName: 'my-endpoint',
    endpointId: 'e-123',
    experimentId: '0',
  };

  // ─── Step 1: Type selection ───────────────────────────────────────────

  test('renders step 1 with guardrail type cards', () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    expect(screen.getByText('Add Guardrail')).toBeInTheDocument();
    expect(screen.getByText('Guardrail Type')).toBeInTheDocument();
    expect(screen.getByText('Safety')).toBeInTheDocument();
    expect(screen.getByText('Custom Guardrail')).toBeInTheDocument();
  });

  test('Next button is disabled until a type is selected', () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    const nextButton = screen.getByRole('button', { name: 'Next' });
    expect(nextButton).toBeDisabled();
  });

  test('selecting a type enables Next button', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    const safetyCard = screen.getByText('Safety').closest('[role="option"]')!;
    await userEvent.click(safetyCard);

    expect(screen.getByRole('button', { name: 'Next' })).not.toBeDisabled();
  });

  // ─── Step 2: Configuration ────────────────────────────────────────────

  test('clicking Next moves to step 2 with configuration', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));

    expect(screen.getByText('Guardrail Details')).toBeInTheDocument();
    expect(screen.getByText('Placement')).toBeInTheDocument();
    expect(screen.getByText('Action')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).toBeInTheDocument();
  });

  test('Safety type pre-fills name', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));

    expect(screen.getByDisplayValue('Safety')).toBeInTheDocument();
  });

  test('Custom type shows description field', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));

    expect(screen.getByText('Description')).toBeInTheDocument();
  });

  test('Safety type shows judge instructions pre-filled', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));

    expect(screen.getByText('Judge Instructions')).toBeInTheDocument();
    // Safety template prompt is pre-filled
    const textarea = screen.getByPlaceholderText('Enter judge instructions...');
    expect((textarea as HTMLTextAreaElement).value).toContain('content safety classifier');
  });

  test('Custom type shows judge instructions empty', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));

    expect(screen.getByText('Judge Instructions')).toBeInTheDocument();
    const textarea = screen.getByPlaceholderText('Enter judge instructions...');
    expect(textarea).toHaveValue('');
  });

  test('Back button returns to step 1', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    await userEvent.click(screen.getByRole('button', { name: /Back/ }));

    expect(screen.getByText('Select the type of guardrail you want to add.')).toBeInTheDocument();
  });

  // ─── Submission ───────────────────────────────────────────────────────

  test('creates guardrail with register + create + add flow', async () => {
    mockRegisterScorer.mockResolvedValue({ scorer_id: 'sc-safety', version: 1 });
    mockCreateGuardrail.mockResolvedValue({ guardrail: { guardrail_id: 'g-abc123' } });
    jest.mocked(GatewayApi.addGuardrailToEndpoint).mockResolvedValue({ config: {} as any });

    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    await userEvent.click(screen.getByRole('button', { name: 'Create Guardrail' }));

    expect(mockRegisterScorer).toHaveBeenCalledWith('0', expect.objectContaining({ name: 'safety' }));
    expect(mockCreateGuardrail).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'Safety',
        scorer_id: 'sc-safety',
        stage: 'BEFORE',
        action: 'VALIDATION',
      }),
    );
    expect(GatewayApi.addGuardrailToEndpoint).toHaveBeenCalledWith({
      endpoint_id: 'e-123',
      guardrail_id: 'g-abc123',
    });
    expect(defaultProps.onSuccess).toHaveBeenCalled();
  });

  test('does not render when closed', () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} open={false} />);
    expect(screen.queryByText('Add Guardrail')).not.toBeInTheDocument();
  });
});
