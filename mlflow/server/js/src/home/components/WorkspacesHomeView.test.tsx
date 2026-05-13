/* eslint-disable @databricks/no-mock-location*/
import { describe, jest, beforeEach, test, expect, afterEach } from '@jest/globals';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { WorkspacesHomeView } from './WorkspacesHomeView';
import { useWorkspaces } from '../../workspaces/hooks/useWorkspaces';
import { getLastUsedWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { useCurrentUserAdminWorkspaces, useCurrentUserIsAdmin } from '../../account/hooks';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

jest.mock('../../workspaces/hooks/useWorkspaces');
jest.mock('../../workspaces/utils/WorkspaceUtils');
jest.mock('../../account/hooks', () => ({
  useCurrentUserAdminWorkspaces: jest.fn<() => Set<string>>(() => new Set()),
  useCurrentUserIsAdmin: jest.fn<() => boolean>(() => false),
}));

const mockedAdminWorkspaces = jest.mocked(useCurrentUserAdminWorkspaces);
const mockedIsAdmin = jest.mocked(useCurrentUserIsAdmin);

const reloadMock = jest.fn();

const mockNavigate = jest.fn();
jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/RoutingUtils')>('../../common/utils/RoutingUtils'),
  useNavigate: () => mockNavigate,
}));

describe('WorkspacesHomeView', () => {
  const mockOnCreateWorkspace = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock last used workspace for "Last used" badge
    jest.mocked(getLastUsedWorkspace).mockReturnValue('ml-research');
    // Default: regular user with no admin reach. Individual cases override
    // for the workspace-manager / platform-admin column-visibility tests.
    mockedAdminWorkspaces.mockReturnValue(new Set());
    mockedIsAdmin.mockReturnValue(false);

    Object.defineProperty(window, 'location', {
      value: { ...window.location, reload: reloadMock },
      writable: true,
    });
  });

  afterEach(() => {
    reloadMock.mockClear();
  });

  const renderComponent = () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    return renderWithDesignSystem(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <WorkspacesHomeView onCreateWorkspace={mockOnCreateWorkspace} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  };

  test('renders loading state', () => {
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [],
      isLoading: true,
      isError: false,
      refetch: jest.fn() as (options: any) => Promise<any>,
    });

    renderComponent();
    expect(screen.getByText('Loading workspaces...')).toBeInTheDocument();
  });

  test('renders empty state when no workspaces', () => {
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as (options: any) => Promise<any>,
    });

    renderComponent();
    expect(screen.getByText('Create your first workspace')).toBeInTheDocument();
    expect(
      screen.getByText('Create a workspace to organize and logically isolate your experiments and models.'),
    ).toBeInTheDocument();
  });

  test('calls onCreateWorkspace when create button clicked in empty state', async () => {
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as (options: any) => Promise<any>,
    });

    renderComponent();
    const createButton = screen.getByText('Create workspace');
    await userEvent.click(createButton);
    expect(mockOnCreateWorkspace).toHaveBeenCalledTimes(1);
  });

  test('renders workspace list with Last used badge', () => {
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        { name: 'ml-research', description: 'Research experiments for new ML models' },
        { name: 'production-models', description: 'Production-ready models' },
        { name: 'data-science-team', description: null },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as (options: any) => Promise<any>,
    });

    renderComponent();

    expect(screen.getByText('ml-research')).toBeInTheDocument();
    expect(screen.getByText('Research experiments for new ML models')).toBeInTheDocument();
    expect(screen.getByText('production-models')).toBeInTheDocument();
    expect(screen.getByText('Production-ready models')).toBeInTheDocument();
    expect(screen.getByText('data-science-team')).toBeInTheDocument();

    // Last used badge should appear for ml-research
    expect(screen.getByText('Last used')).toBeInTheDocument();
  });

  test('navigates to workspace when row clicked', async () => {
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [{ name: 'ml-research', description: 'Research experiments' }],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as (options: any) => Promise<any>,
    });

    renderComponent();

    const workspaceLink = screen.getByText('ml-research');
    await userEvent.click(workspaceLink);

    // Hard reload with workspace query param
    expect(window.location.hash).toBe('#/?workspace=ml-research');
    expect(window.location.reload).toHaveBeenCalled();
  });

  test('encodes workspace name in URL', async () => {
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [{ name: 'team-a/special', description: 'Special workspace' }],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as (options: any) => Promise<any>,
    });

    renderComponent();

    const workspaceLink = screen.getByText('team-a/special');
    await userEvent.click(workspaceLink);

    // Hard reload with encoded workspace query param
    expect(window.location.hash).toBe('#/?workspace=team-a%2Fspecial');
    expect(window.location.reload).toHaveBeenCalled();
  });

  test('shows create new workspace button when workspaces exist', () => {
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [{ name: 'ml-research', description: 'Research experiments' }],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as (options: any) => Promise<any>,
    });

    renderComponent();
    expect(screen.getByText('Create new workspace')).toBeInTheDocument();
  });

  test('renders error state', () => {
    const mockRefetch = jest.fn() as (options: any) => Promise<any>;
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [],
      isLoading: false,
      isError: true,
      refetch: mockRefetch as any,
    });

    renderComponent();
    expect(screen.getByText("We couldn't load your workspaces.")).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  test('calls refetch when retry button clicked', async () => {
    const mockRefetch = jest.fn() as (options: any) => Promise<any>;
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [],
      isLoading: false,
      isError: true,
      refetch: mockRefetch as any,
    });

    renderComponent();
    const retryButton = screen.getByText('Retry');
    await userEvent.click(retryButton);
    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });

  test('hides Manage column when the user has no admin workspaces', () => {
    // Regular user with no admin reach — the typical case.
    mockedAdminWorkspaces.mockReturnValue(new Set());
    mockedIsAdmin.mockReturnValue(false);
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        { name: 'ml-research', description: 'Research experiments' },
        { name: 'production-models', description: 'Production-ready models' },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    expect(screen.queryByText('Manage')).not.toBeInTheDocument();
    expect(screen.queryAllByLabelText(/Manage workspace/)).toHaveLength(0);
  });

  test('hides Manage column for platform admins even if their admin workspaces set is non-empty', () => {
    // Defense-in-depth: ``useCurrentUserAdminWorkspaces`` already short-
    // circuits to an empty set for admins, but the visibility predicate
    // additionally gates on ``!isAdmin`` so the gear stays hidden if that
    // short-circuit ever changes. Simulate a future hook returning the
    // admin's MANAGE roles and assert the column is still hidden.
    mockedIsAdmin.mockReturnValue(true);
    mockedAdminWorkspaces.mockReturnValue(new Set(['ml-research']));
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        { name: 'ml-research', description: 'Research experiments' },
        { name: 'production-models', description: 'Production-ready models' },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    expect(screen.queryByText('Manage')).not.toBeInTheDocument();
    expect(screen.queryAllByLabelText(/Manage workspace/)).toHaveLength(0);
  });

  test('shows Manage column with gear icon only on workspaces the user administers', () => {
    mockedAdminWorkspaces.mockReturnValue(new Set(['ml-research']));
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        { name: 'ml-research', description: 'Research experiments' },
        { name: 'production-models', description: 'Production-ready models' },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    expect(screen.getByText('Manage')).toBeInTheDocument();
    const gears = screen.getAllByLabelText(/Manage workspace/);
    expect(gears).toHaveLength(1);
    expect(gears[0]).toHaveAttribute('aria-label', 'Manage workspace ml-research');
  });

  test('Manage gear navigates to the per-workspace admin route', async () => {
    mockedAdminWorkspaces.mockReturnValue(new Set(['ml-research']));
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [{ name: 'ml-research', description: 'Research experiments' }],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    const gear = screen.getByLabelText('Manage workspace ml-research');
    await userEvent.click(gear);

    // Hard reload onto ``/admin/ws?workspace=…`` — the per-workspace mode.
    expect(window.location.hash).toBe('#/admin/ws?workspace=ml-research');
    expect(window.location.reload).toHaveBeenCalled();
  });
});
