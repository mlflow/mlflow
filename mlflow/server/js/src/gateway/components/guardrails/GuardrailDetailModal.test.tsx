import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { fireEvent } from '@testing-library/react';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { GuardrailDetailModal } from './GuardrailDetailModal';
import { GatewayApi } from '../../api';

jest.mock('../../api', () => ({
  GatewayApi: {
    createGuardrail: jest.fn(),
    addGuardrailToEndpoint: jest.fn(),
    removeGuardrailFromEndpoint: jest.fn(),
  },
}));
jest.mock('../../hooks/useEndpointsQuery', () => {
  const data: never[] = [];
  return { useEndpointsQuery: () => ({ data }) };
});
jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));
jest.mock('../../../experiment-tracking/components/EndpointSelector', () => ({
  EndpointSelector: () => null,
}));

const mockRegisterScorer = jest.fn<any>();
jest.mock('../../../experiment-tracking/pages/experiment-scorers/api', () => ({
  registerScorer: (...args: any[]) => mockRegisterScorer(...args),
}));

describe('GuardrailDetailModal', () => {
  const mockConfig = {
    endpoint_id: 'e-123',
    guardrail_id: 'gr-abc',
    execution_order: 1,
    created_at: 1700000000000,
    guardrail: {
      guardrail_id: 'gr-abc',
      name: 'Safety',
      stage: 'BEFORE' as const,
      action: 'VALIDATION' as const,
      created_at: 1700000000000,
      last_updated_at: 1700000000000,
      scorer: { scorer_id: 'sc-123', scorer_version: 2 },
    },
  };

  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    onDelete: jest.fn(),
    onSuccess: jest.fn(),
    endpointId: 'e-123',
    guardrailConfig: mockConfig,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders guardrail title and version', () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    expect(screen.getByText('Guardrail: Safety')).toBeInTheDocument();
    expect(screen.getByText('v2')).toBeInTheDocument();
  });

  test('renders stage and action selects', () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    // Stage and action are custom div selectors, not <select> elements
    expect(screen.getByText('Before Guardrails')).toBeInTheDocument();
    expect(screen.getByText('After Guardrails')).toBeInTheDocument();
    expect(screen.getByText('Block')).toBeInTheDocument();
    expect(screen.getByText('Sanitize')).toBeInTheDocument();
  });

  test('Save button is disabled when there are no changes', () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    const saveButton = screen.getByRole('button', { name: /Save/ });
    expect(saveButton).toBeDisabled();
  });

  test('Save button enables after editing the prompt with valid instructions', async () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    // Use fireEvent.change to set value with template variables (userEvent.type treats { as special)
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Is {{ inputs }} safe?' } });

    expect(screen.getByRole('button', { name: /Save/ })).not.toBeDisabled();
  });

  test('Remove button calls onDelete and onClose', async () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    await userEvent.click(screen.getByRole('button', { name: /Delete/ }));

    expect(defaultProps.onDelete).toHaveBeenCalledWith('gr-abc');
    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  test('does not render when guardrailConfig is null', () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} guardrailConfig={null} />);
    expect(screen.queryByText(/Guardrail:/)).not.toBeInTheDocument();
  });

  test('does not render when not open', () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} open={false} />);
    expect(screen.queryByText('Guardrail: Safety')).not.toBeInTheDocument();
  });

  test('save flow creates new guardrail before removing old one', async () => {
    const newGuardrailId = 'gr-new';
    jest.mocked(GatewayApi.createGuardrail).mockResolvedValue({
      guardrail: { guardrail_id: newGuardrailId } as any,
    });
    jest.mocked(GatewayApi.addGuardrailToEndpoint).mockResolvedValue({ config: {} as any });
    jest.mocked(GatewayApi.removeGuardrailFromEndpoint).mockResolvedValue({} as any);

    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    // Change stage by clicking the custom div selector
    await userEvent.click(screen.getByText('After Guardrails'));

    await userEvent.click(screen.getByRole('button', { name: /Save/ }));

    // Verify order: create → add → remove
    const createOrder = jest.mocked(GatewayApi.createGuardrail).mock.invocationCallOrder[0];
    const addOrder = jest.mocked(GatewayApi.addGuardrailToEndpoint).mock.invocationCallOrder[0];
    const removeOrder = jest.mocked(GatewayApi.removeGuardrailFromEndpoint).mock.invocationCallOrder[0];
    expect(createOrder).toBeLessThan(addOrder!);
    expect(addOrder).toBeLessThan(removeOrder!);

    expect(GatewayApi.removeGuardrailFromEndpoint).toHaveBeenCalledWith({
      endpoint_id: 'e-123',
      guardrail_id: 'gr-abc',
    });
    expect(defaultProps.onSuccess).toHaveBeenCalled();
  });

  // ─── Stage-variable validation ────────────────────────────────────────

  test('shows BEFORE-stage hint below instructions field', () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    expect(screen.getByText(/Receives {{ inputs }}/)).toBeInTheDocument();
  });

  test('shows error when BEFORE-stage instructions reference {{ outputs }}', async () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Is {{ outputs }} appropriate?' } });

    expect(screen.getByText(/{{ outputs }} is not available in BEFORE stage/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Save/ })).toBeDisabled();
  });

  test('shows error when BEFORE-stage instructions lack {{ inputs }}', async () => {
    renderWithDesignSystem(<GuardrailDetailModal {...defaultProps} />);

    const textarea = screen.getByRole('textbox');
    await userEvent.type(textarea, 'Is this safe?');

    expect(screen.getByText('BEFORE-stage instructions must reference {{ inputs }}')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Save/ })).toBeDisabled();
  });
});
