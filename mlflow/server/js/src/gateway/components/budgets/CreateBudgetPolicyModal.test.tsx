import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { CreateBudgetPolicyModal } from './CreateBudgetPolicyModal';
import { useCreateBudgetPolicy } from '../../hooks/useCreateBudgetPolicy';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';

jest.mock('../../hooks/useCreateBudgetPolicy');
jest.mock('../../hooks/useEndpointsQuery');
jest.mock('../../../experiment-tracking/hooks/useServerInfo', () => ({
  getWorkspacesEnabledSync: () => false,
}));

const mockMutateAsync = jest.fn().mockReturnValue(Promise.resolve());

const mockEndpoints = [
  { endpoint_id: 'e-1', name: 'my-endpoint' },
  { endpoint_id: 'e-2', name: 'other-endpoint' },
];

describe('CreateBudgetPolicyModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useCreateBudgetPolicy).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: mockEndpoints,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
  });

  test('renders form fields when open', () => {
    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={jest.fn()} />);

    expect(screen.getByText('Create Budget Policy')).toBeInTheDocument();
    expect(screen.getByText('Applies to')).toBeInTheDocument();
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

  test('defaults to all-endpoints scope without endpoint picker', () => {
    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={jest.fn()} />);

    expect(screen.getByText('Applies to')).toBeInTheDocument();
    expect(screen.queryByText('Select an endpoint')).not.toBeInTheDocument();
  });

  test('requires an endpoint selection when scope is a specific endpoint', async () => {
    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={jest.fn()} />);

    const amountInput = screen.getByPlaceholderText('e.g., 100.00');
    await userEvent.type(amountInput, '50');

    const createButton = screen.getByRole('button', { name: 'Create' });
    expect(createButton).not.toBeDisabled();

    const [scopeSelect] = screen.getAllByRole('combobox');
    await userEvent.click(scopeSelect);
    await userEvent.click(screen.getByRole('option', { name: 'Specific endpoint' }));

    // No endpoint chosen yet, so Create stays disabled.
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();

    const endpointSelect = screen.getAllByRole('combobox')[1];
    await userEvent.click(endpointSelect);
    await userEvent.click(screen.getByRole('option', { name: 'my-endpoint' }));

    expect(screen.getByRole('button', { name: 'Create' })).not.toBeDisabled();
  });

  test('submits ENDPOINT payload with target_value', async () => {
    const onClose = jest.fn();
    const onSuccess = jest.fn();

    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={onClose} onSuccess={onSuccess} />);

    const amountInput = screen.getByPlaceholderText('e.g., 100.00');
    await userEvent.type(amountInput, '25');

    const [scopeSelect] = screen.getAllByRole('combobox');
    await userEvent.click(scopeSelect);
    await userEvent.click(screen.getByRole('option', { name: 'Specific endpoint' }));

    const endpointSelect = screen.getAllByRole('combobox')[1];
    await userEvent.click(endpointSelect);
    await userEvent.click(screen.getByRole('option', { name: 'other-endpoint' }));

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_unit: 'USD',
      budget_amount: 25,
      duration: { unit: 'MONTHS', value: 1 },
      target_scope: 'ENDPOINT',
      target_value: 'e-2',
      budget_action: 'REJECT',
    });
  });

  test('requires a username when Specific user scope is selected', async () => {
    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={jest.fn()} />);

    const amountInput = screen.getByPlaceholderText('e.g., 100.00');
    await userEvent.type(amountInput, '50');
    expect(screen.getByRole('button', { name: 'Create' })).not.toBeDisabled();

    // The scope select is the first combobox in the modal
    const scopeSelect = screen.getAllByRole('combobox')[0];
    await userEvent.click(scopeSelect);
    await userEvent.click(screen.getByRole('option', { name: 'Specific user' }));

    // No username entered yet, so Create stays disabled.
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();

    await userEvent.type(screen.getByPlaceholderText('Username, e.g., alice'), 'alice');
    expect(screen.getByRole('button', { name: 'Create' })).not.toBeDisabled();
  });

  test('submits USER-scoped payload with principal as target_value', async () => {
    const onClose = jest.fn();
    const onSuccess = jest.fn();

    mockMutateAsync.mockReturnValue(Promise.resolve());

    renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={onClose} onSuccess={onSuccess} />);

    await userEvent.type(screen.getByPlaceholderText('e.g., 100.00'), '25');

    const scopeSelect = screen.getAllByRole('combobox')[0];
    await userEvent.click(scopeSelect);
    await userEvent.click(screen.getByRole('option', { name: 'Specific user' }));
    await userEvent.type(screen.getByPlaceholderText('Username, e.g., alice'), '  alice  ');

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_unit: 'USD',
      budget_amount: 25,
      duration: { unit: 'MONTHS', value: 1 },
      target_scope: 'USER',
      target_value: 'alice',
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
      // Suppress the runtime's default behavior so a reintroduced unhandled
      // rejection surfaces as a clean assertion failure below rather than
      // crashing the whole jest worker.
      event.preventDefault();
      unhandledRejections.push(event.reason);
    };
    window.addEventListener('unhandledrejection', handler);

    try {
      renderWithDesignSystem(<CreateBudgetPolicyModal open onClose={onClose} onSuccess={onSuccess} />);

      const amountInput = screen.getByPlaceholderText('e.g., 100.00');
      await userEvent.type(amountInput, '50');

      const createButton = screen.getByRole('button', { name: 'Create' });
      await userEvent.click(createButton);

      // `unhandledrejection` is dispatched asynchronously, so a regression
      // could fire it on a later turn. Wait for the rejected mutation to be
      // observed, then advance one macrotask so any pending rejection event
      // has had a chance to run before we assert.
      await waitFor(() => expect(rejectingMutateAsync).toHaveBeenCalledTimes(1));
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(unhandledRejections).toEqual([]);
      // Modal stays open and onSuccess does not fire on failure.
      expect(onClose).not.toHaveBeenCalled();
      expect(onSuccess).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener('unhandledrejection', handler);
    }
  });
});
