/* eslint-disable @databricks/no-mock-location*/
import { describe, jest, beforeEach, test, expect, afterEach } from '@jest/globals';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { WorkspacesHomeView } from './WorkspacesHomeView';
import { useWorkspaces } from '../../workspaces/hooks/useWorkspaces';
import { getLastUsedWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { useCurrentUserAdminWorkspaces, useCurrentUserIsAdmin, useIsAuthAvailable } from '../../account/hooks';
import { useUpdateWorkspace } from '../../workspaces/hooks/useUpdateWorkspace';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

jest.mock('../../workspaces/hooks/useWorkspaces');
jest.mock('../../account/hooks', () => ({
  useCurrentUserAdminWorkspaces: jest.fn<() => Set<string>>(() => new Set()),
  useCurrentUserIsAdmin: jest.fn<() => boolean>(() => false),
  useIsAuthAvailable: jest.fn<() => boolean>(() => true),
}));
jest.mock('../../workspaces/hooks/useUpdateWorkspace');
jest.mock('../../workspaces/utils/WorkspaceUtils', () => {
  const actualWorkspaceUtils = jest.requireActual<typeof import('../../workspaces/utils/WorkspaceUtils')>(
    '../../workspaces/utils/WorkspaceUtils',
  );
  return {
    ...actualWorkspaceUtils,
    getLastUsedWorkspace: jest.fn(),
    setLastUsedWorkspace: jest.fn(),
  };
});

const mockedAdminWorkspaces = jest.mocked(useCurrentUserAdminWorkspaces);
const mockedIsAdmin = jest.mocked(useCurrentUserIsAdmin);
const mockedIsAuthAvailable = jest.mocked(useIsAuthAvailable);

const reloadMock = jest.fn();
const mockUpdateWorkspace = jest.fn();

describe('WorkspacesHomeView', () => {
  const mockOnCreateWorkspace = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(getLastUsedWorkspace).mockReturnValue('ml-research');
    // Default: regular user with no admin reach. Individual cases override
    // for the workspace-manager / platform-admin column-visibility tests.
    mockedAdminWorkspaces.mockReturnValue(new Set());
    mockedIsAdmin.mockReturnValue(false);
    mockedIsAuthAvailable.mockReturnValue(true);
    Object.defineProperty(window, 'location', {
      value: { ...window.location, hash: '', reload: reloadMock },
      writable: true,
    });
    jest.mocked(useUpdateWorkspace).mockReturnValue({
      mutate: mockUpdateWorkspace,
      isLoading: false,
    } as any);
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

  const openTraceArchivalSection = async () => {
    await userEvent.click(screen.getByRole('button', { name: 'Trace archival settings' }));
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
    await userEvent.click(screen.getByText('Create workspace'));
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

    await userEvent.click(screen.getByText('ml-research'));

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

    await userEvent.click(screen.getByText('team-a/special'));

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

  test('opens edit modal with workspace fields', async () => {
    mockedIsAdmin.mockReturnValue(true);
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        {
          name: 'ml-research',
          description: 'Research experiments',
          default_artifact_root: 's3://artifacts/ml-research',
          trace_archival_config: { location: 's3://archive/ml-research', retention: '30d' },
        },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    expect(screen.queryByText('Artifact Root')).not.toBeInTheDocument();
    expect(screen.queryByText('s3://artifacts/ml-research')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await openTraceArchivalSection();

    expect(screen.getByText('Edit Workspace')).toBeInTheDocument();
    expect(screen.getByDisplayValue('Research experiments')).toBeInTheDocument();
    expect(screen.getByDisplayValue('s3://artifacts/ml-research')).toBeInTheDocument();
    expect(screen.getByDisplayValue('s3://archive/ml-research')).toBeInTheDocument();
    expect(screen.getByLabelText('Trace Archival Retention')).toHaveValue('30');
    expect(screen.getByText('Clear any optional field and save to remove the workspace override.')).toBeInTheDocument();
  });

  test('saves updated fields from the edit modal', async () => {
    mockedIsAdmin.mockReturnValue(true);
    mockUpdateWorkspace.mockImplementation((_variables, options: any) => {
      options?.onSuccess?.({} as any, undefined as any, undefined as any);
    });
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        {
          name: 'ml-research',
          description: 'Research experiments',
          default_artifact_root: 's3://artifacts/ml-research',
        },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await userEvent.clear(screen.getByDisplayValue('Research experiments'));
    await userEvent.type(screen.getByPlaceholderText('Enter workspace description'), 'Updated description');
    await userEvent.clear(screen.getByPlaceholderText('Enter default artifact root URI'));
    await userEvent.type(screen.getByPlaceholderText('Enter default artifact root URI'), 's3://artifacts/new-team');

    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(mockUpdateWorkspace).toHaveBeenCalledWith(
        {
          name: 'ml-research',
          description: 'Updated description',
          default_artifact_root: 's3://artifacts/new-team',
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
          onError: expect.any(Function),
        }),
      );
    });
  });

  test('saves updated archival fields from the edit modal', async () => {
    mockedIsAdmin.mockReturnValue(true);
    mockUpdateWorkspace.mockImplementation((_variables, options: any) => {
      options?.onSuccess?.({} as any, undefined as any, undefined as any);
    });
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        {
          name: 'ml-research',
          description: 'Research experiments',
          default_artifact_root: 's3://artifacts/ml-research',
          trace_archival_config: { location: 's3://archive/ml-research', retention: '30d' },
        },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await openTraceArchivalSection();
    await userEvent.clear(screen.getByDisplayValue('s3://archive/ml-research'));
    await userEvent.type(screen.getByPlaceholderText('Enter trace archival location URI'), 's3://archive/new-team');
    await userEvent.clear(screen.getByLabelText('Trace Archival Retention'));
    await userEvent.type(screen.getByLabelText('Trace Archival Retention'), '14');

    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(mockUpdateWorkspace).toHaveBeenCalledWith(
        {
          name: 'ml-research',
          trace_archival_config: { location: 's3://archive/new-team', retention: '14d' },
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
          onError: expect.any(Function),
        }),
      );
    });
  });

  test('saves only the changed trace archival location field from the edit modal', async () => {
    mockedIsAdmin.mockReturnValue(true);
    mockUpdateWorkspace.mockImplementation((_variables, options: any) => {
      options?.onSuccess?.({} as any, undefined as any, undefined as any);
    });
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        {
          name: 'ml-research',
          description: 'Research experiments',
          trace_archival_config: { location: 's3://archive/ml-research', retention: '30d' },
        },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await openTraceArchivalSection();
    await userEvent.clear(screen.getByDisplayValue('s3://archive/ml-research'));
    await userEvent.type(screen.getByPlaceholderText('Enter trace archival location URI'), 's3://archive/new-team');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(mockUpdateWorkspace).toHaveBeenCalledWith(
        {
          name: 'ml-research',
          trace_archival_config: { location: 's3://archive/new-team' },
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
          onError: expect.any(Function),
        }),
      );
    });
  });

  test('clears archival overrides from the edit modal', async () => {
    mockedIsAdmin.mockReturnValue(true);
    mockUpdateWorkspace.mockImplementation((_variables, options: any) => {
      options?.onSuccess?.({} as any, undefined as any, undefined as any);
    });
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        {
          name: 'ml-research',
          description: 'Research experiments',
          default_artifact_root: 's3://artifacts/ml-research',
          trace_archival_config: { location: 's3://archive/ml-research', retention: '30d' },
        },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await openTraceArchivalSection();
    await userEvent.clear(screen.getByDisplayValue('s3://archive/ml-research'));
    await userEvent.clear(screen.getByLabelText('Trace Archival Retention'));

    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(mockUpdateWorkspace).toHaveBeenCalledWith(
        {
          name: 'ml-research',
          trace_archival_config: { location: '', retention: '' },
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
          onError: expect.any(Function),
        }),
      );
    });
  });

  test('does not save archival overrides when only whitespace changes', async () => {
    mockedIsAdmin.mockReturnValue(true);
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        {
          name: 'ml-research',
          description: 'Research experiments',
          default_artifact_root: 's3://artifacts/ml-research',
          trace_archival_config: { location: 's3://archive/ml-research', retention: '30d' },
        },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await openTraceArchivalSection();
    await userEvent.type(screen.getByDisplayValue('s3://archive/ml-research'), ' ');
    await userEvent.type(screen.getByLabelText('Trace Archival Retention'), ' ');

    await userEvent.click(screen.getByText('Save'));

    expect(mockUpdateWorkspace).not.toHaveBeenCalled();
  });

  test('preserves an unparseable retention value when saving other workspace fields', async () => {
    mockedIsAdmin.mockReturnValue(true);
    mockUpdateWorkspace.mockImplementation((_variables, options: any) => {
      options?.onSuccess?.({} as any, undefined as any, undefined as any);
    });
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [
        {
          name: 'ml-research',
          description: 'Research experiments',
          trace_archival_config: { retention: 'future-format' },
        },
      ],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await openTraceArchivalSection();
    expect(screen.getByLabelText('Trace Archival Retention')).toHaveValue('');

    await userEvent.clear(screen.getByDisplayValue('Research experiments'));
    await userEvent.type(screen.getByPlaceholderText('Enter workspace description'), 'Updated description');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(mockUpdateWorkspace).toHaveBeenCalledWith(
        {
          name: 'ml-research',
          description: 'Updated description',
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
          onError: expect.any(Function),
        }),
      );
    });
  });

  test('shows an inline error when saving the edit modal fails', async () => {
    mockedIsAdmin.mockReturnValue(true);
    mockUpdateWorkspace.mockImplementation((_variables, options: any) => {
      options?.onError?.(new Error('Save failed'));
    });
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [{ name: 'ml-research', description: 'Research experiments' }],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await userEvent.clear(screen.getByDisplayValue('Research experiments'));
    await userEvent.type(screen.getByPlaceholderText('Enter workspace description'), 'Updated description');
    await userEvent.click(screen.getByText('Save'));

    expect(await screen.findByText('Save failed')).toBeInTheDocument();
  });

  test('shows validation error for invalid trace archival retention in edit modal', async () => {
    mockedIsAdmin.mockReturnValue(true);
    jest.mocked(useWorkspaces).mockReturnValue({
      workspaces: [{ name: 'ml-research', description: 'Research experiments' }],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    });

    renderComponent();
    await userEvent.click(screen.getByRole('button', { name: 'Edit workspace' }));
    await openTraceArchivalSection();
    await userEvent.type(screen.getByLabelText('Trace Archival Retention'), '30days');
    await userEvent.click(screen.getByText('Save'));

    expect(
      await screen.findByText(
        "Trace archival retention must use the format <int><unit>, where unit is one of 'm', 'h', or 'd'.",
      ),
    ).toBeInTheDocument();
    expect(mockUpdateWorkspace).not.toHaveBeenCalled();
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
    await userEvent.click(screen.getByText('Retry'));
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
    expect(screen.queryByRole('button', { name: 'Edit workspace' })).not.toBeInTheDocument();
    expect(screen.queryAllByLabelText(/Manage workspace/)).toHaveLength(0);
  });

  test('shows edit column when auth is unavailable', () => {
    mockedIsAuthAvailable.mockReturnValue(false);
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
    expect(screen.getAllByRole('button', { name: 'Edit workspace' })).toHaveLength(2);
    expect(screen.queryByText('Manage')).not.toBeInTheDocument();
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
    expect(screen.getAllByRole('button', { name: 'Edit workspace' })).toHaveLength(2);
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
    expect(screen.queryByRole('button', { name: 'Edit workspace' })).not.toBeInTheDocument();
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
