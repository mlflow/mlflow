import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from '../common/utils/RoutingUtils';
import WorkspaceLandingPage from './WorkspaceLandingPage';
import { useWorkspaces } from '../workspaces/hooks/useWorkspaces';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

jest.mock('../workspaces/hooks/useWorkspaces');
jest.mock('../workspaces/utils/WorkspaceUtils', () => ({
  ...jest.requireActual<typeof import('../workspaces/utils/WorkspaceUtils')>('../workspaces/utils/WorkspaceUtils'),
  getActiveWorkspace: jest.fn().mockReturnValue('default'),
  setActiveWorkspace: jest.fn(),
}));

const mockNavigate = jest.fn();
jest.mock('../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../common/utils/RoutingUtils')>('../common/utils/RoutingUtils'),
  useNavigate: () => mockNavigate,
}));

// Mock child components to simplify testing
jest.mock('./components/features', () => ({
  FeaturesSection: () => <div data-testid="features-section">Features Section</div>,
}));

jest.mock('./components/DiscoverNews', () => ({
  __esModule: true,
  default: () => <div data-testid="discover-news">Discover News Component</div>,
}));

jest.mock('./components/LogTracesDrawerLoader', () => ({
  __esModule: true,
  default: () => <div data-testid="log-traces-drawer">Log Traces Drawer</div>,
}));

jest.mock('../telemetry/TelemetryInfoAlert', () => ({
  TelemetryInfoAlert: () => <div data-testid="telemetry-alert">Telemetry Alert</div>,
}));

describe('WorkspaceLandingPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (useWorkspaces as jest.Mock).mockReturnValue({
      workspaces: [
        { name: 'ml-research', description: 'Research experiments' },
        { name: 'production-models', description: 'Production models' },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn(),
    });
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
          <WorkspaceLandingPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  };

  test('renders Welcome to MLflow header', () => {
    renderComponent();
    expect(screen.getByText('Welcome to MLflow')).toBeInTheDocument();
  });

  test('renders all main sections', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByTestId('telemetry-alert')).toBeInTheDocument();
      expect(screen.getByTestId('features-section')).toBeInTheDocument();
      expect(screen.getByTestId('log-traces-drawer')).toBeInTheDocument();
    });
  });

  test('renders workspaces section', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Workspaces')).toBeInTheDocument();
      expect(screen.getByText('Select a workspace to start experiments')).toBeInTheDocument();
    });
  });

  test('renders workspace list', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('ml-research')).toBeInTheDocument();
      expect(screen.getByText('Research experiments')).toBeInTheDocument();
      expect(screen.getByText('production-models')).toBeInTheDocument();
      expect(screen.getByText('Production models')).toBeInTheDocument();
    });
  });

  test('shows create workspace button', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Create new workspace')).toBeInTheDocument();
    });
  });

  test('opens create workspace modal when button clicked', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Create new workspace')).toBeInTheDocument();
    });

    const createButton = screen.getByText('Create new workspace');
    await userEvent.click(createButton);

    // Modal should open with title
    await waitFor(() => {
      expect(screen.getByText('Create Workspace')).toBeInTheDocument();
    });
  });
});
