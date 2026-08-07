import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { EditBudgetPolicyModal } from './EditBudgetPolicyModal';
import { useUpdateBudgetPolicy } from '../../hooks/useUpdateBudgetPolicy';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import type { BudgetPolicy } from '../../types';

jest.mock('../../hooks/useUpdateBudgetPolicy');
jest.mock('../../hooks/useEndpointsQuery');
jest.mock('../../../experiment-tracking/hooks/useServerInfo', () => ({
  getWorkspacesEnabledSync: () => false,
}));

const mockMutateAsync = jest.fn().mockReturnValue(Promise.resolve());

const mockPolicy: BudgetPolicy = {
  budget_policy_id: 'bp-1',
  budget_unit: 'USD',
  budget_amount: 200,
  duration: { unit: 'WEEKS', value: 1 },
  target_scope: 'GLOBAL',
  budget_action: 'ALERT',
  created_at: Date.now() / 1000,
  last_updated_at: Date.now() / 1000,
};

const mockEndpointPolicy: BudgetPolicy = {
  ...mockPolicy,
  budget_policy_id: 'bp-ep',
  target_scope: 'ENDPOINT',
  target_value: 'e-1',
};

const mockUserPolicy: BudgetPolicy = {
  ...mockPolicy,
  budget_policy_id: 'bp-user',
  target_scope: 'USER',
  budget_action: 'REJECT',
  target_value: 'alice',
};

describe('EditBudgetPolicyModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useUpdateBudgetPolicy).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [
        { endpoint_id: 'e-1', name: 'my-endpoint' },
        { endpoint_id: 'e-2', name: 'other-endpoint' },
      ],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);
  });

  test('renders nothing when policy is null', () => {
    const { container } = renderWithDesignSystem(<EditBudgetPolicyModal open policy={null} onClose={jest.fn()} />);

    expect(container.innerHTML).toBe('');
  });

  test('renders form with policy data populated', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('Edit Budget Policy')).toBeInTheDocument();
    expect(screen.getByText('Budget amount (USD)')).toBeInTheDocument();
    expect(screen.getByDisplayValue('200')).toBeInTheDocument();
  });

  test('submits with correct payload mapping duration preset', async () => {
    const onClose = jest.fn();
    const onSuccess = jest.fn();

    mockMutateAsync.mockReturnValue(Promise.resolve());

    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={onClose} onSuccess={onSuccess} />);

    const saveButton = screen.getByRole('button', { name: 'Save Changes' });
    await userEvent.click(saveButton);

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_policy_id: 'bp-1',
      budget_unit: 'USD',
      budget_amount: 200,
      duration: { unit: 'WEEKS', value: 1 },
      target_scope: 'GLOBAL',
      budget_action: 'ALERT',
    });
  });

  test('initializes scope and endpoint from an ENDPOINT policy', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockEndpointPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('Specific endpoint')).toBeInTheDocument();
    expect(screen.getByText('my-endpoint')).toBeInTheDocument();
  });

  test('submits ENDPOINT payload preserving target_value', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockEndpointPolicy} onClose={jest.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_policy_id: 'bp-ep',
      budget_unit: 'USD',
      budget_amount: 200,
      duration: { unit: 'WEEKS', value: 1 },
      target_scope: 'ENDPOINT',
      target_value: 'e-1',
      budget_action: 'ALERT',
    });
  });

  test('switching an ENDPOINT policy back to all endpoints drops target_value', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockEndpointPolicy} onClose={jest.fn()} />);

    const [scopeSelect] = screen.getAllByRole('combobox');
    await userEvent.click(scopeSelect);
    await userEvent.click(screen.getByRole('option', { name: 'All endpoints and users' }));

    await userEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_policy_id: 'bp-ep',
      budget_unit: 'USD',
      budget_amount: 200,
      duration: { unit: 'WEEKS', value: 1 },
      target_scope: 'GLOBAL',
      budget_action: 'ALERT',
    });
  });

  test('renders principal field populated for a USER-scoped policy', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockUserPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('Specific user')).toBeInTheDocument();
    expect(screen.getByDisplayValue('alice')).toBeInTheDocument();
  });

  test('does not render principal field for a non-USER policy', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={jest.fn()} />);

    expect(screen.queryByPlaceholderText('Username, e.g., alice')).not.toBeInTheDocument();
  });

  test('preserves USER scope and submits edited principal', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockUserPolicy} onClose={jest.fn()} />);

    const principalInput = screen.getByDisplayValue('alice');
    await userEvent.clear(principalInput);
    await userEvent.type(principalInput, 'bob');

    await userEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_policy_id: 'bp-user',
      budget_unit: 'USD',
      budget_amount: 200,
      duration: { unit: 'WEEKS', value: 1 },
      target_scope: 'USER',
      target_value: 'bob',
      budget_action: 'REJECT',
    });
  });

  test('disables save when the principal of a USER policy is cleared', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockUserPolicy} onClose={jest.fn()} />);

    await userEvent.clear(screen.getByDisplayValue('alice'));

    expect(screen.getByRole('button', { name: 'Save Changes' })).toBeDisabled();
    expect(mockMutateAsync).not.toHaveBeenCalled();
  });

  test('does not propagate unhandled rejection when submit fails (e.g. USER scope without auth)', async () => {
    const onClose = jest.fn();
    const onSuccess = jest.fn();
    // Lazy-construct the rejection so it isn't created at mock-setup time
    // (which would briefly look unhandled before the modal awaits it).
    const rejectingMutateAsync = jest
      .fn()
      .mockImplementation(() =>
        Promise.reject(new Error('USER-scoped budget policies require server authentication to be enabled.')),
      );
    jest.mocked(useUpdateBudgetPolicy).mockReturnValue({
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
      renderWithDesignSystem(
        <EditBudgetPolicyModal open policy={mockPolicy} onClose={onClose} onSuccess={onSuccess} />,
      );

      await userEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

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

  test('displays error message on mutation failure', () => {
    jest.mocked(useUpdateBudgetPolicy).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isLoading: false,
      error: new Error('Update failed'),
      reset: jest.fn(),
    } as any);

    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('Update failed')).toBeInTheDocument();
  });
});
