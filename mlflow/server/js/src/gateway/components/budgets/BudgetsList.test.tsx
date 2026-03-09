import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { BudgetsList } from './BudgetsList';
import { useBudgetPoliciesQuery } from '../../hooks/useBudgetPoliciesQuery';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';

jest.mock('../../hooks/useBudgetPoliciesQuery');

const now = Date.now() / 1000;

const mockPolicies = [
  {
    budget_policy_id: 'bp-1',
    budget_unit: 'USD' as const,
    budget_amount: 100,
    duration_unit: 'DAYS' as const,
    duration_value: 1,
    target_scope: 'GLOBAL' as const,
    budget_action: 'REJECT' as const,
    created_at: now,
    last_updated_at: now,
  },
  {
    budget_policy_id: 'bp-2',
    budget_unit: 'USD' as const,
    budget_amount: 500.5,
    duration_unit: 'MONTHS' as const,
    duration_value: 1,
    target_scope: 'WORKSPACE' as const,
    budget_action: 'ALERT' as const,
    created_at: now,
    last_updated_at: now,
  },
];

describe('BudgetsList', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state', () => {
    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: [],
      isLoading: true,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('Loading budget policies...')).toBeInTheDocument();
  });

  test('renders empty state when no policies', () => {
    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('No budget policies created')).toBeInTheDocument();
  });

  test('renders policies with formatted data', () => {
    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: mockPolicies,
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('$100.00')).toBeInTheDocument();
    expect(screen.getByText('$500.50')).toBeInTheDocument();
    expect(screen.getByText('Daily')).toBeInTheDocument();
    expect(screen.getByText('Monthly')).toBeInTheDocument();
    expect(screen.getByText('Reject')).toBeInTheDocument();
    expect(screen.getByText('Alert')).toBeInTheDocument();
  });

  test('calls onEditClick when edit button is clicked', async () => {
    const onEditClick = jest.fn();

    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: [mockPolicies[0]],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList onEditClick={onEditClick} />
      </MemoryRouter>,
    );

    await userEvent.click(screen.getByLabelText('Edit budget policy'));

    expect(onEditClick).toHaveBeenCalledWith(mockPolicies[0]);
  });

  test('calls onDeleteClick when delete button is clicked', async () => {
    const onDeleteClick = jest.fn();

    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: [mockPolicies[0]],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList onDeleteClick={onDeleteClick} />
      </MemoryRouter>,
    );

    await userEvent.click(screen.getByLabelText('Delete budget policy'));

    expect(onDeleteClick).toHaveBeenCalledWith(mockPolicies[0]);
  });
});
