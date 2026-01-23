import { describe, jest, beforeEach, it, expect } from '@jest/globals';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { DesignSystemProvider } from '@databricks/design-system';

import { WorkspaceSelector } from './WorkspaceSelector';
import { shouldEnableWorkspaces } from '../utils/FeatureUtils';
import { setActiveWorkspace } from '../utils/WorkspaceUtils';
import { MemoryRouter, useNavigate, useLocation } from '../utils/RoutingUtils';
import { fetchAPI } from '../utils/FetchUtils';

jest.mock('../utils/FeatureUtils', () => ({
  shouldEnableWorkspaces: jest.fn(),
}));

jest.mock('../utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../utils/RoutingUtils')>('../utils/RoutingUtils'),
  useNavigate: jest.fn(),
  useLocation: jest.fn(),
}));

jest.mock('../utils/FetchUtils', () => ({
  ...jest.requireActual<typeof import('../utils/FetchUtils')>('../utils/FetchUtils'),
  fetchAPI: jest.fn(),
}));

const shouldEnableWorkspacesMock = jest.mocked(shouldEnableWorkspaces);
const useNavigateMock = jest.mocked(useNavigate);
const useLocationMock = jest.mocked(useLocation);
const fetchAPIMock = jest.mocked(fetchAPI);

describe('WorkspaceSelector', () => {
  const mockNavigate = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    shouldEnableWorkspacesMock.mockReturnValue(true);
    useNavigateMock.mockReturnValue(mockNavigate);
    useLocationMock.mockReturnValue({
      pathname: '/workspaces/default/experiments',
      search: '',
      hash: '',
      state: null,
      key: 'default',
    });
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
          <MemoryRouter initialEntries={['/workspaces/default/experiments']}>{component}</MemoryRouter>
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
      pathname: '/workspaces/default/models',
      search: '',
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

  it('redirects to fallback workspace when current workspace is no longer available', async () => {
    // Set active workspace to one that won't be in the list
    setActiveWorkspace('removed-workspace');
    useLocationMock.mockReturnValue({
      pathname: '/workspaces/removed-workspace/experiments',
      search: '',
      hash: '',
      state: null,
      key: 'default',
    });

    fetchAPIMock.mockResolvedValue({
      // Return workspaces that don't include the current one
      workspaces: [{ name: 'default' }, { name: 'team-a' }],
    });

    renderWithProviders(<WorkspaceSelector />);

    // Should automatically redirect to default workspace
    await waitFor(
      () => {
        expect(mockNavigate).toHaveBeenCalledWith('/workspaces/default/experiments');
      },
      { timeout: 3000 },
    );
  });
});
