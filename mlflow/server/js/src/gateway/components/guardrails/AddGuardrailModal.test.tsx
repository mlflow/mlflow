import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { fireEvent } from '@testing-library/react';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { AddGuardrailModal } from './AddGuardrailModal';
import { useCreateGuardrail } from '../../hooks/useCreateGuardrail';
import { GatewayApi } from '../../api';

const mockEndpoints: Array<{ endpoint_id: string; name: string }> = [];

jest.mock('../../hooks/useCreateGuardrail');
let mockEndpointsLoading = false;
let mockEndpointsError: Error | undefined = undefined;

jest.mock('../../hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: () => ({ data: mockEndpoints, isLoading: mockEndpointsLoading, error: mockEndpointsError }),
}));
jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));
jest.mock('../../../experiment-tracking/components/EndpointSelector', () => ({
  EndpointSelector: ({
    onEndpointSelect,
    disabled,
  }: {
    onEndpointSelect: (value: string) => void;
    disabled?: boolean;
  }) => (
    <button type="button" onClick={() => onEndpointSelect('judge-endpoint')} disabled={disabled}>
      Select guardrail model
    </button>
  ),
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
const setMockEndpoints = (endpoints: Array<{ endpoint_id: string; name: string }>) => {
  mockEndpoints.length = 0;
  mockEndpoints.push(...endpoints);
};

describe('AddGuardrailModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockEndpointsLoading = false;
    mockEndpointsError = undefined;
    setMockEndpoints([{ endpoint_id: 'e-456', name: 'judge-endpoint' }]);
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

  const selectGuardrailModel = async () => {
    await userEvent.click(screen.getByRole('button', { name: 'Select guardrail model' }));
  };

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
    await selectGuardrailModel();
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

  test('shows Pre-LLM stage hint on instructions field', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    // Default stage is BEFORE (Pre-LLM)
    expect(screen.getByText(/Receives {{ inputs }}/)).toBeInTheDocument();
  });

  test('shows Post-LLM stage hint when Post-LLM stage is selected', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);
    await userEvent.click(screen.getByText('Post-LLM Guardrails'));

    expect(screen.getByText(/Receives {{ inputs }}.*{{ outputs }}/s)).toBeInTheDocument();
  });

  test('shows error and disables Create when Pre-LLM stage instructions lack {{ inputs }}', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    const nameInput = screen.getByPlaceholderText('e.g., PII Detection & Redaction');
    await userEvent.type(nameInput, 'My Guard');

    const textarea = screen.getByPlaceholderText(instructionsPlaceholder);
    await userEvent.type(textarea, 'Is this safe?');

    expect(screen.getByText('Pre-LLM Guardrails instructions must reference {{ inputs }}')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).toBeDisabled();
  });

  test('enables Create when Pre-LLM stage instructions contain {{ inputs }}', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    const nameInput = screen.getByPlaceholderText('e.g., PII Detection & Redaction');
    await userEvent.type(nameInput, 'My Guard');

    // userEvent.type treats { as special — use fireEvent.change for template variable syntax
    const textarea = screen.getByPlaceholderText(instructionsPlaceholder);
    fireEvent.change(textarea, { target: { value: 'Is {{ inputs }} free of profanity?' } });
    await selectGuardrailModel();

    expect(screen.queryByText(/must reference|not available/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).not.toBeDisabled();
  });

  test('disables Create when Guardrail Model is not selected', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);

    expect(screen.getByRole('button', { name: 'Create Guardrail' })).toBeDisabled();
  });

  test('shows no-endpoint guidance when no alternate endpoint is available', async () => {
    setMockEndpoints([{ endpoint_id: 'e-123', name: 'my-endpoint' }]);
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);

    expect(screen.getByText('You need another endpoint to use guardrails.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Select guardrail model' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).toBeDisabled();
  });

  test('does not show no-endpoint guidance while endpoints are loading', async () => {
    mockEndpointsLoading = true;
    setMockEndpoints([]);
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);

    expect(screen.queryByText('You need another endpoint to use guardrails.')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Select guardrail model' })).not.toBeDisabled();
  });

  test('does not show no-endpoint guidance when endpoints query fails', async () => {
    mockEndpointsError = new Error('Network error');
    setMockEndpoints([]);
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Safety').closest('[role="option"]')!);

    expect(screen.queryByText('You need another endpoint to use guardrails.')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Select guardrail model' })).not.toBeDisabled();
  });

  test('shows error when Pre-LLM stage instructions reference {{ outputs }}', async () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} />);

    await userEvent.click(screen.getByText('Custom Guardrail').closest('[role="option"]')!);

    const nameInput = screen.getByPlaceholderText('e.g., PII Detection & Redaction');
    await userEvent.type(nameInput, 'My Guard');

    const textarea = screen.getByPlaceholderText(instructionsPlaceholder);
    fireEvent.change(textarea, { target: { value: 'Is {{ outputs }} appropriate?' } });

    expect(screen.getByText(/{{ outputs }} is not available in Pre-LLM Guardrails/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Guardrail' })).toBeDisabled();
  });

  test('does not render when closed', () => {
    renderWithDesignSystem(<AddGuardrailModal {...defaultProps} open={false} />);
    expect(screen.queryByText('Select the type of guardrail you want to create.')).not.toBeInTheDocument();
  });
});
