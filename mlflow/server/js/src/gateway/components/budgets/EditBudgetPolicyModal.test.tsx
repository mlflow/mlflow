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

  test('renders username field populated for a USER-scoped policy', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockUserPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('Specific user')).toBeInTheDocument();
    expect(screen.getByDisplayValue('alice')).toBeInTheDocument();
  });

  test('does not render username field for a non-USER policy', () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockPolicy} onClose={jest.fn()} />);

    expect(screen.queryByPlaceholderText('Username, e.g., alice')).not.toBeInTheDocument();
  });

  test('preserves USER scope and submits edited username', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockUserPolicy} onClose={jest.fn()} />);

    const usernameInput = screen.getByDisplayValue('alice');
    await userEvent.clear(usernameInput);
    await userEvent.type(usernameInput, 'bob');

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

  test('disables save when the username of a USER policy is cleared', async () => {
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

  describe('when the target endpoint has been deleted', () => {
    // Endpoint deletion doesn't cascade to budget policies, so a policy can
    // outlive its endpoint. The picker has no option for the stale id, so it
    // renders its placeholder and the selection is unrecoverable from the UI.
    const stalePolicy: BudgetPolicy = {
      ...mockEndpointPolicy,
      budget_policy_id: 'bp-stale',
      target_value: 'e-deleted',
    };

    test('disables save and explains the missing endpoint', () => {
      renderWithDesignSystem(<EditBudgetPolicyModal open policy={stalePolicy} onClose={jest.fn()} />);

      expect(screen.getByRole('button', { name: 'Save Changes' })).toBeDisabled();
      expect(
        screen.getByText(
          'The endpoint this policy applied to (e-deleted) no longer exists. Select another endpoint, or change this policy to apply to all endpoints and users.',
        ),
      ).toBeInTheDocument();
      expect(mockMutateAsync).not.toHaveBeenCalled();
    });

    test('re-enables save once a live endpoint is selected', async () => {
      renderWithDesignSystem(<EditBudgetPolicyModal open policy={stalePolicy} onClose={jest.fn()} />);

      const [, endpointSelect] = screen.getAllByRole('combobox');
      await userEvent.click(endpointSelect);
      await userEvent.click(screen.getByRole('option', { name: 'my-endpoint' }));

      await userEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

      expect(mockMutateAsync).toHaveBeenCalledWith({
        budget_policy_id: 'bp-stale',
        budget_unit: 'USD',
        budget_amount: 200,
        duration: { unit: 'WEEKS', value: 1 },
        target_scope: 'ENDPOINT',
        target_value: 'e-1',
        budget_action: 'ALERT',
      });
    });

    test('re-enables save when switching the policy to all endpoints and users', async () => {
      renderWithDesignSystem(<EditBudgetPolicyModal open policy={stalePolicy} onClose={jest.fn()} />);

      const [scopeSelect] = screen.getAllByRole('combobox');
      await userEvent.click(scopeSelect);
      await userEvent.click(screen.getByRole('option', { name: 'All endpoints and users' }));

      await userEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

      expect(mockMutateAsync).toHaveBeenCalledWith({
        budget_policy_id: 'bp-stale',
        budget_unit: 'USD',
        budget_amount: 200,
        duration: { unit: 'WEEKS', value: 1 },
        target_scope: 'GLOBAL',
        budget_action: 'ALERT',
      });
    });

    test('keeps save enabled while endpoints are still loading', () => {
      jest.mocked(useEndpointsQuery).mockReturnValue({
        data: [],
        isLoading: true,
        error: undefined,
        refetch: jest.fn(),
      } as any);

      renderWithDesignSystem(<EditBudgetPolicyModal open policy={stalePolicy} onClose={jest.fn()} />);

      // An in-flight endpoints request must not be mistaken for "deleted".
      expect(screen.getByRole('button', { name: 'Save Changes' })).not.toBeDisabled();
      expect(screen.queryByText('No endpoints available')).not.toBeInTheDocument();
    });
  });

  test('does not report a live endpoint as deleted when the endpoints request fails', () => {
    // `useEndpointsQuery` yields `data: []` on failure just as it does while
    // loading, so a failed list must not be read as "the endpoint was deleted" —
    // that would declare a valid policy broken and block editing it.
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: new Error('You do not have permission to access this resource.'),
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockEndpointPolicy} onClose={jest.fn()} />);

    expect(
      screen.queryByText(
        'The endpoint this policy applied to (e-1) no longer exists. Select another endpoint, or change this policy to apply to all endpoints and users.',
      ),
    ).not.toBeInTheDocument();
    expect(screen.queryByText('No endpoints available')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save Changes' })).not.toBeDisabled();
  });

  test('shows an empty-state placeholder when no endpoints exist', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockEndpointPolicy} onClose={jest.fn()} />);

    expect(screen.getByText('No endpoints available')).toBeInTheDocument();
    expect(screen.queryByText('Select an endpoint')).not.toBeInTheDocument();
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
