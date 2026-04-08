import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, within } from '../../../common/utils/TestUtils.react18';
import { GuardrailModal } from './AddGuardrailModal';
import { useAddGuardrail } from '../../hooks/useAddGuardrail';
import { GatewayApi } from '../../api';

jest.mock('../../hooks/useAddGuardrail');
jest.mock('../../api', () => ({
  GatewayApi: {
    addGuardrailToEndpoint: jest.fn(),
  },
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
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ scorers: [] }),
      }),
    ) as any;
  });

  const defaultProps = {
    open: true,
    onClose: jest.fn(),
    onSuccess: jest.fn(),
    endpointName: 'my-endpoint',
    endpointId: 'e-123',
    experimentId: '0',
  };

  test('renders modal with title and selectors when open', () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    expect(screen.getByText('Add Guardrail')).toBeInTheDocument();
    expect(screen.getByText('When the guardrail runs')).toBeInTheDocument();
    expect(screen.getByText('Action')).toBeInTheDocument();
  });

  test('renders builtin scorers list', () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    expect(screen.getByText('Safety')).toBeInTheDocument();
  });

  test('filters scorers by search query', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    const searchInput = screen.getByPlaceholderText('Search guardrails...');
    await userEvent.type(searchInput, 'safety');

    expect(screen.getByText('Safety')).toBeInTheDocument();
  });

  test('shows empty state when search matches nothing', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    const searchInput = screen.getByPlaceholderText('Search guardrails...');
    await userEvent.type(searchInput, 'xyznonexistent');

    expect(screen.getByText('No scorers found matching your search.')).toBeInTheDocument();
  });

  test('selecting a scorer enables the Add button', async () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    const addButton = screen.getByRole('button', { name: 'Add' });
    expect(addButton).toBeDisabled();

    // Click on Safety scorer row
    const safetyRow = screen.getByText('Safety').closest('[role="option"]')!;
    await userEvent.click(safetyRow);

    expect(addButton).not.toBeDisabled();
  });

  test('submits create + add-to-endpoint on Add click', async () => {
    mockCreateGuardrail.mockResolvedValue({
      guardrail: { guardrail_id: 'g-abc123' },
    });
    jest.mocked(GatewayApi.addGuardrailToEndpoint).mockResolvedValue({
      config: {} as any,
    });

    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);

    // Select a scorer
    const safetyRow = screen.getByText('Safety').closest('[role="option"]')!;
    await userEvent.click(safetyRow);

    // Click Add
    const addButton = screen.getByRole('button', { name: 'Add' });
    await userEvent.click(addButton);

    expect(mockCreateGuardrail).toHaveBeenCalledWith({
      name: 'Safety',
      scorer_id: 'Safety',
      scorer_version: 1,
      stage: 'BEFORE',
      action: 'VALIDATION',
    });

    expect(GatewayApi.addGuardrailToEndpoint).toHaveBeenCalledWith({
      endpoint_id: 'e-123',
      guardrail_id: 'g-abc123',
    });

    expect(defaultProps.onSuccess).toHaveBeenCalled();
    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  test('does not render when closed', () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} open={false} />);
    expect(screen.queryByText('Add Guardrail')).not.toBeInTheDocument();
  });

  test('renders "+ Create custom guardrail" link', () => {
    renderWithDesignSystem(<GuardrailModal {...defaultProps} />);
    expect(screen.getByText('+ Create custom guardrail')).toBeInTheDocument();
  });
});
