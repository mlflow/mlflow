import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { BudgetsList } from './BudgetsList';
import { useBudgetPoliciesQuery } from '../../hooks/useBudgetPoliciesQuery';
import { useBudgetWindowsQuery } from '../../hooks/useBudgetWindowsQuery';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';

jest.mock('../../hooks/useBudgetPoliciesQuery');
jest.mock('../../hooks/useBudgetWindowsQuery');

const now = Date.now() / 1000;

const mockPolicies = [
  {
    budget_policy_id: 'bp-1',
    budget_unit: 'USD' as const,
    budget_amount: 100,
    duration: { unit: 'DAYS' as const, value: 1 },
    target_scope: 'GLOBAL' as const,
    budget_action: 'REJECT' as const,
    created_at: now,
    last_updated_at: now,
  },
  {
    budget_policy_id: 'bp-2',
    budget_unit: 'USD' as const,
    budget_amount: 500.5,
    duration: { unit: 'MONTHS' as const, value: 1 },
    target_scope: 'WORKSPACE' as const,
    budget_action: 'ALERT' as const,
    created_at: now,
    last_updated_at: now,
  },
];

describe('BudgetsList', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useBudgetWindowsQuery).mockReturnValue({
      data: {},
      isLoading: false,
      error: undefined,
    });
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

    expect(screen.getByText('$100')).toBeInTheDocument();
    expect(screen.getByText('$500.5')).toBeInTheDocument();
    expect(screen.getByText('Daily')).toBeInTheDocument();
    expect(screen.getByText('Monthly')).toBeInTheDocument();
    expect(screen.getByText('Reject')).toBeInTheDocument();
    expect(screen.getByText('Alert')).toBeInTheDocument();
  });

  test('renders window columns with spend data when available', () => {
    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: [mockPolicies[0]],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    jest.mocked(useBudgetWindowsQuery).mockReturnValue({
      data: {
        'bp-1': {
          budget_policy_id: 'bp-1',
          window_start_ms: new Date('2026-03-01T00:00:00Z').getTime(),
          window_end_ms: new Date('2026-03-02T00:00:00Z').getTime(),
          current_spend: 42.5,
        },
      },
      isLoading: false,
      error: undefined,
    });

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList />
      </MemoryRouter>,
    );

    // Current spend should be formatted as budget amount
    expect(screen.getByText('$42.5')).toBeInTheDocument();
  });

  test('shows violation indicator when current spend exceeds budget', () => {
    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: [mockPolicies[0]], // budget_amount: 100
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    jest.mocked(useBudgetWindowsQuery).mockReturnValue({
      data: {
        'bp-1': {
          budget_policy_id: 'bp-1',
          window_start_ms: new Date('2026-03-01T00:00:00Z').getTime(),
          window_end_ms: new Date('2026-03-02T00:00:00Z').getTime(),
          current_spend: 120, // exceeds budget_amount of 100
        },
      },
      isLoading: false,
      error: undefined,
    });

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('$120')).toBeInTheDocument();
    expect(screen.getByLabelText('Budget exceeded')).toBeInTheDocument();
  });

  test('does not show violation indicator when spend is within budget', () => {
    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: [mockPolicies[0]], // budget_amount: 100
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    jest.mocked(useBudgetWindowsQuery).mockReturnValue({
      data: {
        'bp-1': {
          budget_policy_id: 'bp-1',
          window_start_ms: new Date('2026-03-01T00:00:00Z').getTime(),
          window_end_ms: new Date('2026-03-02T00:00:00Z').getTime(),
          current_spend: 50, // within budget_amount of 100
        },
      },
      isLoading: false,
      error: undefined,
    });

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList />
      </MemoryRouter>,
    );

    expect(screen.getByText('$50')).toBeInTheDocument();
    expect(screen.queryByLabelText('Budget exceeded')).not.toBeInTheDocument();
  });

  test('renders dash placeholders when no window data exists', () => {
    jest.mocked(useBudgetPoliciesQuery).mockReturnValue({
      data: [mockPolicies[0]],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    jest.mocked(useBudgetWindowsQuery).mockReturnValue({
      data: {},
      isLoading: false,
      error: undefined,
    });

    renderWithDesignSystem(
      <MemoryRouter>
        <BudgetsList />
      </MemoryRouter>,
    );

    // Should show em-dash placeholders for missing window data
    const dashes = screen.getAllByText('—');
    expect(dashes.length).toBe(3); // Window Start, Window End, Current Spend
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
