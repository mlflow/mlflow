import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import BudgetsPage from './BudgetsPage';

// Mock useBudgetsPage hook
const mockUseBudgetsPage = jest.fn();
jest.mock('../hooks/useBudgetsPage', () => ({
  useBudgetsPage: () => mockUseBudgetsPage(),
}));

// Mock BudgetsList
jest.mock('../components/budgets/BudgetsList', () => ({
  BudgetsList: () => <div data-testid="budgets-list">Budget Policies List</div>,
}));

// Mock WebhooksSettings
jest.mock('../../settings/WebhooksSettings', () => ({
  __esModule: true,
  default: () => <div data-testid="webhooks-settings">Webhooks Settings</div>,
}));

// Mock budget modals
jest.mock('../components/budgets/CreateBudgetPolicyModal', () => ({
  CreateBudgetPolicyModal: () => null,
}));
jest.mock('../components/budgets/EditBudgetPolicyModal', () => ({
  EditBudgetPolicyModal: () => null,
}));
jest.mock('../components/budgets/DeleteBudgetPolicyModal', () => ({
  DeleteBudgetPolicyModal: () => null,
}));

describe('BudgetsPage', () => {
  const mockHandleCreateClick = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseBudgetsPage.mockReturnValue({
      isCreateModalOpen: false,
      editingPolicy: null,
      deletingPolicy: null,
      handleCreateClick: mockHandleCreateClick,
      handleCreateModalClose: jest.fn(),
      handleCreateSuccess: jest.fn(),
      handleEditClick: jest.fn(),
      handleEditModalClose: jest.fn(),
      handleEditSuccess: jest.fn(),
      handleDeleteClick: jest.fn(),
      handleDeleteModalClose: jest.fn(),
      handleDeleteSuccess: jest.fn(),
    });
  });

  const renderComponent = (initialEntry = '/gateway/budgets') => {
    return renderWithDesignSystem(
      <MemoryRouter initialEntries={[initialEntry]}>
        <BudgetsPage />
      </MemoryRouter>,
    );
  };

  test('renders page title and breadcrumb', () => {
    renderComponent();

    expect(screen.getAllByText('Budgets').length).toBeGreaterThanOrEqual(1);
  });

  test('renders Policies tab by default', () => {
    renderComponent();

    expect(screen.getByTestId('budgets-list')).toBeInTheDocument();
    expect(screen.queryByTestId('webhooks-settings')).not.toBeInTheDocument();
  });

  test('renders Alerts tab when selected', async () => {
    renderComponent();

    await userEvent.click(screen.getByText('Alerts'));

    expect(screen.getByTestId('webhooks-settings')).toBeInTheDocument();
    expect(screen.queryByTestId('budgets-list')).not.toBeInTheDocument();
  });

  test('renders Alerts tab when ?tab=alerts is in URL', () => {
    renderComponent('/gateway/budgets?tab=alerts');

    expect(screen.getByTestId('webhooks-settings')).toBeInTheDocument();
    expect(screen.queryByTestId('budgets-list')).not.toBeInTheDocument();
  });

  test('shows create button on Policies tab', () => {
    renderComponent();

    expect(screen.getByText('Create budget policy')).toBeInTheDocument();
  });

  test('hides create button on Alerts tab', async () => {
    renderComponent();

    await userEvent.click(screen.getByText('Alerts'));

    expect(screen.queryByText('Create budget policy')).not.toBeInTheDocument();
  });

  test('switches back to Policies tab', async () => {
    renderComponent();

    await userEvent.click(screen.getByText('Alerts'));
    expect(screen.getByTestId('webhooks-settings')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Policies'));
    expect(screen.getByTestId('budgets-list')).toBeInTheDocument();
    expect(screen.queryByTestId('webhooks-settings')).not.toBeInTheDocument();
  });
});
