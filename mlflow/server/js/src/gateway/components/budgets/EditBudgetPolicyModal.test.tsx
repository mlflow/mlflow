import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
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
  endpoint_id: 'e-1',
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

  test('submits ENDPOINT payload preserving endpoint_id', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockEndpointPolicy} onClose={jest.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    expect(mockMutateAsync).toHaveBeenCalledWith({
      budget_policy_id: 'bp-ep',
      budget_unit: 'USD',
      budget_amount: 200,
      duration: { unit: 'WEEKS', value: 1 },
      target_scope: 'ENDPOINT',
      endpoint_id: 'e-1',
      budget_action: 'ALERT',
    });
  });

  test('switching an ENDPOINT policy back to all endpoints drops endpoint_id', async () => {
    renderWithDesignSystem(<EditBudgetPolicyModal open policy={mockEndpointPolicy} onClose={jest.fn()} />);

    const [scopeSelect] = screen.getAllByRole('combobox');
    await userEvent.click(scopeSelect);
    await userEvent.click(screen.getByRole('option', { name: 'All endpoints' }));

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
