import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { fireEvent } from '@testing-library/react';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { AddGuardrailModal } from './AddGuardrailModal';
import { useCreateGuardrail } from '../../hooks/useCreateGuardrail';
import { GatewayApi } from '../../api';

jest.mock('../../hooks/useCreateGuardrail');
jest.mock('../../hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: () => ({ data: [] }),
}));
jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));
jest.mock('../../../experiment-tracking/components/EndpointSelector', () => ({
  EndpointSelector: () => null,
}));
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

describe('AddGuardrailModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useCreateGuardrail).mockReturnValue({
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

  const instructionsPlaceholder = 'Describe what this guardrail should check for...';

  // ─── Step 1: Type selection ───────────────────────────────────────────

  test('renders step 1 with guardrail type cards', () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    expect(screen.getByText('Create Guardrail')).toBeInTheDocument();
    expect(screen.getByText('Safety')).toBeInTheDocument();
    expect(screen.getByText('Custom Guardrail')).toBeInTheDocument();
    expect(screen.getByText('Select the type of guardrail you want to create.')).toBeInTheDocument();
  });

  test('clicking a type card advances directly to step 2', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);

    expect(screen.getByText('Instructions')).toBeInTheDocument();
    expect(screen.getByText('Stage')).toBeInTheDocument();
    expect(screen.getByText('Action')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).toBeInTheDocument();
  });

  // ─── Step 2: Configuration ────────────────────────────────────────────

  test('Safety type pre-fills name', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);

    expect(screen.getByDisplayValue('Safety')).toBeInTheDocument();
  });

  test('Custom type shows instructions field', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    expect(screen.getByText('Instructions')).toBeInTheDocument();
  });

  test('Safety type pre-fills instructions with safety prompt', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);

    const textarea = screen.getByPlaceholderText(instructionsPlaceholder);
    expect((textarea as HTMLTextAreaElement).value).toContain('content safety classifier');
  });

  test('Custom type shows instructions field empty', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    const textarea = screen.getByPlaceholderText(instructionsPlaceholder);
    expect(textarea).toHaveValue('');
  });

  test('Back button returns to step 1', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: /Back/ }));

    expect(screen.getByText('Select the type of guardrail you want to create.')).toBeInTheDocument();
  });

  // ─── Submission ───────────────────────────────────────────────────────

  test('creates guardrail with register + create + add flow', async () => {
    mockRegisterScorer.mockResolvedValue({ scorer_id: 'sc-safety', version: 1 });
    mockCreateGuardrail.mockResolvedValue({ guardrail: { guardrail_id: 'g-abc123' } });
    jest.mocked(GatewayApi.addGuardrailToEndpoint).mockResolvedValue({ config: {} as any });

    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);
    await userEvent.click(screen.getByRole('button', { name: 'Create Guardrail' }));

    expect(mockRegisterScorer).toHaveBeenCalledWith('0', expect.objectContaining({ name: 'safety' }));
    expect(mockCreateGuardrail).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'Safety',
        scorer_id: 'sc-safety',
        stage: 'AFTER',
        action: 'VALIDATION',
      }),
    );
    expect(GatewayApi.addGuardrailToEndpoint).toHaveBeenCalledWith({
      endpoint_id: 'e-123',
      guardrail_id: 'g-abc123',
    });
    expect(defaultProps.onSuccess).toHaveBeenCalled();
  });

  // ─── Stage-variable validation ────────────────────────────────────────

  test('shows BEFORE-stage hint on instructions field', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    // Default stage is BEFORE
    expect(screen.getByText(/Receives {{ inputs }}/)).toBeInTheDocument();
  });

  test('shows AFTER-stage hint when AFTER stage is selected', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);
    await userEvent.click(screen.getByText('After Guardrails'));

    expect(screen.getByText(/Receives {{ inputs }}.*{{ outputs }}/s)).toBeInTheDocument();
  });

  test('shows error and disables Create when BEFORE-stage instructions lack {{ inputs }}', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    const nameInput = screen.getByPlaceholderText('e.g., PII Detection & Redaction');
    await userEvent.type(nameInput, 'My Guard');

    const textarea = screen.getByPlaceholderText(instructionsPlaceholder);
    await userEvent.type(textarea, 'Is this safe?');

    expect(screen.getByText('BEFORE-stage instructions must reference {{ inputs }}')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).toBeDisabled();
  });

  test('enables Create when BEFORE-stage instructions contain {{ inputs }}', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    const nameInput = screen.getByPlaceholderText('e.g., PII Detection & Redaction');
    await userEvent.type(nameInput, 'My Guard');

    // userEvent.type treats { as special — use fireEvent.change for template variable syntax
    const textarea = screen.getByPlaceholderText(instructionsPlaceholder);
    fireEvent.change(textarea, { target: { value: 'Is {{ inputs }} free of profanity?' } });

    expect(screen.queryByText(/must reference|not available/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).not.toBeDisabled();
  });

  test('shows error when BEFORE-stage instructions reference {{ outputs }}', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    const nameInput = screen.getByPlaceholderText('e.g., PII Detection & Redaction');
    await userEvent.type(nameInput, 'My Guard');

    const textarea = screen.getByPlaceholderText(instructionsPlaceholder);
    fireEvent.change(textarea, { target: { value: 'Is {{ outputs }} appropriate?' } });

    expect(screen.getByText(/{{ outputs }} is not available in BEFORE stage/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).toBeDisabled();
  });

  test('does not render when closed', () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} open={false} />);
    expect(screen.queryByText('Select the type of guardrail you want to create.')).not.toBeInTheDocument();
  });
});
