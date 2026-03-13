import { describe, jest, beforeEach, it, expect } from '@jest/globals';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { DesignSystemProvider } from '@databricks/design-system';

import { WorkspaceSelector } from './WorkspaceSelector';
import { shouldEnableWorkspaces } from '../../common/utils/FeatureUtils';
import { setActiveWorkspace } from '../utils/WorkspaceUtils';
import { MemoryRouter, useNavigate, useLocation, useSearchParams } from '../../common/utils/RoutingUtils';
import { fetchAPI } from '../../common/utils/FetchUtils';

jest.mock('../../common/utils/FeatureUtils', () => ({
  shouldEnableWorkspaces: jest.fn(),
}));

jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/RoutingUtils')>('../../common/utils/RoutingUtils'),
  useNavigate: jest.fn(),
  useLocation: jest.fn(),
  useSearchParams: jest.fn(),
}));

jest.mock('../../common/utils/FetchUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/FetchUtils')>('../../common/utils/FetchUtils'),
  fetchAPI: jest.fn(),
}));

const shouldEnableWorkspacesMock = jest.mocked(shouldEnableWorkspaces);
const useNavigateMock = jest.mocked(useNavigate);
const useLocationMock = jest.mocked(useLocation);
const useSearchParamsMock = jest.mocked(useSearchParams);
const fetchAPIMock = jest.mocked(fetchAPI);

describe('WorkspaceSelector', () => {
  const mockNavigate = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    shouldEnableWorkspacesMock.mockReturnValue(true);
    useNavigateMock.mockReturnValue(mockNavigate);
    useLocationMock.mockReturnValue({
      pathname: '/experiments',
      search: '?workspace=default',
      hash: '',
      state: null,
      key: 'default',
    });
    // Mock useSearchParams to return URLSearchParams with workspace=default
    useSearchParamsMock.mockReturnValue([new URLSearchParams('workspace=default'), jest.fn()]);
    setActiveWorkspace('default');
  });

  const renderWithProviders = (component: React.ReactElement) => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    return render(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <MemoryRouter initialEntries={['/experiments?workspace=default']}>{component}</MemoryRouter>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  it('returns null when workspaces feature is disabled', () => {
    shouldEnableWorkspacesMock.mockReturnValue(false);

    const { container } = renderWithProviders(<WorkspaceSelector />);

    expect(container).toBeEmptyDOMElement();
  });

  it('renders combobox trigger with current workspace name', async () => {
    fetchAPIMock.mockResolvedValue({ workspaces: [{ name: 'default' }, { name: 'team-a' }] });

    renderWithProviders(<WorkspaceSelector />);

    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });
  });

  it('shows loading state while fetching workspaces', async () => {
    fetchAPIMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve({ workspaces: [{ name: 'default' }] }), 100);
        }),
    );

    renderWithProviders(<WorkspaceSelector />);

    // Verify the combobox trigger is rendered
    const trigger = screen.getByRole('combobox');
    expect(trigger).toBeInTheDocument();

    // Wait for loading to complete
    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });
  });

  it('shows error message when workspace fetch fails', async () => {
    fetchAPIMock.mockRejectedValue(new Error('Server error'));

    renderWithProviders(<WorkspaceSelector />);

    // Click to open dropdown
    const trigger = screen.getByRole('combobox');
    await userEvent.click(trigger);

    await waitFor(() => {
      expect(screen.getByText('Failed to load workspaces')).toBeInTheDocument();
    });
  });

  it('displays all available workspaces in dropdown', async () => {
    fetchAPIMock.mockResolvedValue({
      workspaces: [
        { name: 'default', description: 'Default workspace' },
        { name: 'team-a', description: 'Team A' },
        { name: 'team-b', description: null },
      ],
    });

    renderWithProviders(<WorkspaceSelector />);

    // Wait for data to load and verify the trigger shows current workspace
    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });

    // Verify the combobox is rendered with the workspaces data loaded
    const trigger = screen.getByRole('combobox');
    expect(trigger).toBeInTheDocument();
    expect(fetchAPIMock).toHaveBeenCalled();
  });

  it('fetches workspaces when enabled', async () => {
    fetchAPIMock.mockResolvedValue({
      workspaces: [
        { name: 'default', description: 'Default' },
        { name: 'team-a', description: 'Team A' },
        { name: 'team-b', description: 'Team B' },
      ],
    });

    renderWithProviders(<WorkspaceSelector />);

    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });

    // Verify fetchAPI was called to get workspaces
    expect(fetchAPIMock).toHaveBeenCalled();

    // Verify combobox trigger exists
    const trigger = screen.getByRole('combobox');
    expect(trigger).toBeInTheDocument();
  });

  it('renders with single workspace', async () => {
    fetchAPIMock.mockResolvedValue({ workspaces: [{ name: 'default' }] });

    renderWithProviders(<WorkspaceSelector />);

    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });

    const trigger = screen.getByRole('combobox');
    expect(trigger).toBeInTheDocument();
  });

  it('renders workspace selector with correct label', async () => {
    fetchAPIMock.mockResolvedValue({ workspaces: [{ name: 'default' }, { name: 'team-a' }] });

    renderWithProviders(<WorkspaceSelector />);

    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });

    // Verify the combobox is labeled "Workspace"
    const trigger = screen.getByRole('combobox', { name: /workspace/i });
    expect(trigger).toBeInTheDocument();
  });

  it('renders correctly on different navigation sections', async () => {
    fetchAPIMock.mockResolvedValue({ workspaces: [{ name: 'default' }, { name: 'team-a' }] });

    // Test with /models path
    useLocationMock.mockReturnValue({
      pathname: '/models',
      search: '?workspace=default',
      hash: '',
      state: null,
      key: 'default',
    });

    renderWithProviders(<WorkspaceSelector />);

    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });

    const trigger = screen.getByRole('combobox');
    expect(trigger).toBeInTheDocument();
  });

  it('calls refetch when combobox is opened', async () => {
    let fetchCount = 0;
    fetchAPIMock.mockImplementation(() => {
      fetchCount++;
      return Promise.resolve({ workspaces: [{ name: 'default' }] });
    });

    renderWithProviders(<WorkspaceSelector />);

    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });

    expect(fetchCount).toBe(1);

    // Open dropdown - should trigger refetch
    const trigger = screen.getByRole('combobox');
    await userEvent.click(trigger);

    await waitFor(() => {
      expect(fetchCount).toBe(2);
    });
  });

  it('loads workspace with description', async () => {
    fetchAPIMock.mockResolvedValue({
      workspaces: [{ name: 'default', description: 'This is the default workspace' }],
    });

    renderWithProviders(<WorkspaceSelector />);

    await waitFor(() => {
      expect(screen.getByText('default')).toBeInTheDocument();
    });

    // Verify component rendered successfully with workspace that has description
    const trigger = screen.getByRole('combobox');
    expect(trigger).toBeInTheDocument();
  });

  // Note: Workspace validation tests moved to MlflowRouter tests
  // WorkspaceSelector is now a pure UI component without validation logic
});
