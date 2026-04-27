/* eslint-disable @databricks/no-mock-location*/
import { describe, jest, beforeEach, test, expect, afterEach } from '@jest/globals';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { WorkspacesHomeView } from './WorkspacesHomeView';
import { useWorkspaces } from '../../workspaces/hooks/useWorkspaces';
import { getLastUsedWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

jest.mock('../../workspaces/hooks/useWorkspaces');
jest.mock('../../workspaces/utils/WorkspaceUtils');

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
    return renderWithIntl(
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
});
