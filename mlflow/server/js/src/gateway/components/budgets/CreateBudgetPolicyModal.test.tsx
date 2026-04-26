import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { CreateBudgetPolicyModal } from './CreateBudgetPolicyModal';
import { useCreateBudgetPolicy } from '../../hooks/useCreateBudgetPolicy';

jest.mock('../../hooks/useCreateBudgetPolicy');
jest.mock('../../../experiment-tracking/hooks/useServerInfo', () => ({
  getWorkspacesEnabledSync: () => false,
}));

const mockMutateAsync = jest.fn().mockReturnValue(Promise.resolve());

describe('CreateBudgetPolicyModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useCreateBudgetPolicy).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
  });

  test('renders form fields when open', () => {
    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={jest.fn()} />);

    expect(screen.getByText('Create Budget Policy')).toBeInTheDocument();
    expect(screen.getByText('Budget amount (USD)')).toBeInTheDocument();
    expect(screen.getByText('Reset period')).toBeInTheDocument();
    expect(screen.getByText('On exceeded')).toBeInTheDocument();
  });

  test('has Create button disabled initially (empty amount)', () => {
    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={jest.fn()} />);

    const createButton = screen.getByRole('button', { name: 'Create' });
    expect(createButton).toBeDisabled();
  });

  test('enables Create button when amount is entered', async () => {
    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={jest.fn()} />);

    const amountInput = screen.getByPlaceholderText('e.g., 100.00');
    await userEvent.type(amountInput, '50');

    const createButton = screen.getByRole('button', { name: 'Create' });
    expect(createButton).not.toBeDisabled();
  });

  test('submits correct payload with duration preset mapping', async () => {
    const onClose = jest.fn();
    const onSuccess = jest.fn();

    mockMutateAsync.mockReturnValue(Promise.resolve());

    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={onClose} onSuccess={onSuccess} />);

    const amountInput = screen.getByPlaceholderText('e.g., 100.00');
    await userEvent.type(amountInput, '100');

    const createButton = screen.getByRole('button', { name: 'Create' });
    await userEvent.click(createButton);

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_unit: 'USD',
      budget_amount: 100,
      duration: { unit: 'MONTHS', value: 1 },
      target_scope: 'GLOBAL',
      budget_action: 'REJECT',
    });
  });

  test('displays error message on mutation failure', () => {
    jest.mocked(useCreateBudgetPolicy).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isLoading: false,
      error: new Error('Budget limit reached'),
      reset: jest.fn(),
    } as any);

    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={jest.fn()} />);

    expect(screen.getByText('Budget limit reached')).toBeInTheDocument();
  });

  test('does not propagate unhandled rejection when submit fails (e.g. 403)', async () => {
    const onClose = jest.fn();
    const onSuccess = jest.fn();
    // Lazy-construct the rejection so it isn't created at mock-setup time
    // (which would briefly look unhandled before the modal awaits it).
    const rejectingMutateAsync = jest
      .fn()
      .mockImplementation(() => Promise.reject(new Error('You do not have permission to access this resource.')));
    jest.mocked(useCreateBudgetPolicy).mockReturnValue({
      mutateAsync: rejectingMutateAsync,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);

    const unhandledRejections: unknown[] = [];
    const handler = (event: PromiseRejectionEvent) => {
      unhandledRejections.push(event.reason);
    };
    window.addEventListener('unhandledrejection', handler);

    try {
      renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={onClose} onSuccess={onSuccess} />);

      const amountInput = screen.getByPlaceholderText('e.g., 100.00');
      await userEvent.type(amountInput, '50');

      const createButton = screen.getByRole('button', { name: 'Create' });
      await userEvent.click(createButton);

      expect(rejectingMutateAsync).toHaveBeenCalledTimes(1);
      // Modal stays open and onSuccess does not fire on failure.
      expect(onClose).not.toHaveBeenCalled();
      expect(onSuccess).not.toHaveBeenCalled();
      // The rejection must not surface as an unhandled promise rejection
      // (which would render as the dev-server runtime error overlay).
      expect(unhandledRejections).toEqual([]);
    } finally {
      window.removeEventListener('unhandledrejection', handler);
    }
  });
});
