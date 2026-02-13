import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { render, waitFor, screen } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  ServerFeaturesProvider,
  useWorkspacesEnabled,
  getWorkspacesEnabledSync,
  resetServerFeaturesCache,
} from './ServerFeaturesContext';

// Mock fetch globally
const mockFetch = jest.fn() as jest.MockedFunction<typeof fetch>;
global.fetch = mockFetch;

const TestComponent = () => {
  const { workspacesEnabled, loading } = useWorkspacesEnabled();
  return (
    <div>
      <span data-testid="loading">{loading ? 'loading' : 'loaded'}</span>
      <span data-testid="workspaces-enabled">{workspacesEnabled ? 'true' : 'false'}</span>
    </div>
  );
};

// Helper to create a fresh QueryClient for each test
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

// Helper to render with providers
const renderWithProviders = (ui: React.ReactElement, queryClient: QueryClient) => {
  return render(
    <QueryClientProvider client={queryClient}>
      <ServerFeaturesProvider>{ui}</ServerFeaturesProvider>
    </QueryClientProvider>,
  );
};

describe('ServerFeaturesContext', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
    mockFetch.mockReset();
  });

  afterEach(() => {
    resetServerFeaturesCache();
    queryClient.clear();
  });

  test('should show loading state initially', async () => {
    // Never resolve the fetch to test loading state
    mockFetch.mockImplementation(() => new Promise(() => {}));

    renderWithProviders(<TestComponent />, queryClient);

    expect(screen.getByTestId('loading').textContent).toBe('loading');
  });

  test('should fetch server features and enable workspaces when server returns enabled', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ workspaces_enabled: true }),
    } as Response);

    renderWithProviders(<TestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('true');
    expect(getWorkspacesEnabledSync()).toBe(true);
  });

  test('should disable workspaces when server returns disabled', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ workspaces_enabled: false }),
    } as Response);

    renderWithProviders(<TestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
    expect(getWorkspacesEnabledSync()).toBe(false);
  });

  test('should disable workspaces when server returns 404 (old backend)', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({}),
    } as Response);

    renderWithProviders(<TestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
    expect(getWorkspacesEnabledSync()).toBe(false);
  });

  test('should disable workspaces when fetch fails with network error', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'));

    renderWithProviders(<TestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
    expect(getWorkspacesEnabledSync()).toBe(false);
  });

  test('should disable workspaces when server returns error status', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    } as Response);

    renderWithProviders(<TestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
    expect(getWorkspacesEnabledSync()).toBe(false);
  });

  test('should cache server features via React Query and not fetch again', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ workspaces_enabled: true }),
    } as Response);

    const { unmount } = renderWithProviders(<TestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);

    // Unmount and remount - use same queryClient to test caching
    unmount();

    render(
      <QueryClientProvider client={queryClient}>
        <ServerFeaturesProvider>
          <TestComponent />
        </ServerFeaturesProvider>
      </QueryClientProvider>,
    );

    // Should immediately show loaded since React Query has cached the data
    expect(screen.getByTestId('loading').textContent).toBe('loaded');
    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('true');

    // Should not have called fetch again - React Query handles deduplication
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  test('getWorkspacesEnabledSync returns false before fetch completes', () => {
    // Create a fresh queryClient without any cached data
    const freshQueryClient = createTestQueryClient();
    mockFetch.mockImplementation(() => new Promise(() => {}));

    // Before render, no queryClient reference is set
    expect(getWorkspacesEnabledSync()).toBe(false);

    // Render to set up the queryClient reference
    renderWithProviders(<TestComponent />, freshQueryClient);

    // While still loading (fetch not complete), should return false
    expect(getWorkspacesEnabledSync()).toBe(false);
  });
});
