import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { render, renderHook, waitFor, screen } from '@testing-library/react';
import React from 'react';
import { rest } from 'msw';
import { setupServer } from '../../common/utils/setup-msw';
import {
  useServerInfo,
  useIsFileStore,
  useWorkspacesEnabled,
  getWorkspacesEnabledSync,
  resetServerInfoCache,
  ServerInfoProvider,
} from './useServerInfo';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

describe('useServerInfo', () => {
  describe('when backend returns FileStore', () => {
    setupServer(
      rest.get('/server-info', (_req, res, ctx) => {
        return res(ctx.json({ store_type: 'FileStore', workspaces_enabled: false }));
      }),
    );

    test('should return store_type as FileStore', async () => {
      const { result } = renderHook(() => useServerInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data?.store_type).toBe('FileStore');
    });
  });

  describe('when backend returns SqlAlchemyStore', () => {
    setupServer(
      rest.get('/server-info', (_req, res, ctx) => {
        return res(ctx.json({ store_type: 'SqlAlchemyStore', workspaces_enabled: false }));
      }),
    );

    test('should return store_type as SqlAlchemyStore', async () => {
      const { result } = renderHook(() => useServerInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data?.store_type).toBe('SqlAlchemyStore');
    });
  });

  describe('when backend returns an error', () => {
    setupServer(
      rest.get('/server-info', (_req, res, ctx) => {
        return res(ctx.status(500));
      }),
    );

    test('should return default value (store_type: empty string)', async () => {
      const { result } = renderHook(() => useServerInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data?.store_type).toBe('');
    });
  });

  describe('when endpoint does not exist (404)', () => {
    setupServer(
      rest.get('/server-info', (_req, res, ctx) => {
        return res(ctx.status(404));
      }),
    );

    test('should return default value (store_type: empty string)', async () => {
      const { result } = renderHook(() => useServerInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data?.store_type).toBe('');
    });
  });
});

describe('useIsFileStore', () => {
  describe('when backend uses FileStore', () => {
    setupServer(
      rest.get('/server-info', (_req, res, ctx) => {
        return res(ctx.json({ store_type: 'FileStore', workspaces_enabled: false }));
      }),
    );

    test('should return true', async () => {
      const { result } = renderHook(() => useIsFileStore(), { wrapper });

      await waitFor(() => {
        expect(result.current).toBe(true);
      });
    });
  });

  describe('when backend uses SqlAlchemyStore', () => {
    setupServer(
      rest.get('/server-info', (_req, res, ctx) => {
        return res(ctx.json({ store_type: 'SqlAlchemyStore', workspaces_enabled: false }));
      }),
    );

    test('should return false', async () => {
      const { result } = renderHook(() => useIsFileStore(), { wrapper });

      await waitFor(() => {
        expect(result.current).toBe(false);
      });
    });
  });
});

// Helper component to test useWorkspacesEnabled
const WorkspacesTestComponent = () => {
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
      <ServerInfoProvider>{ui}</ServerInfoProvider>
    </QueryClientProvider>,
  );
};

describe('useWorkspacesEnabled and getWorkspacesEnabledSync', () => {
  // Mock fetch globally for these tests
  const mockFetch = jest.fn() as jest.MockedFunction<typeof fetch>;
  const originalFetch = global.fetch;

  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
    global.fetch = mockFetch;
    mockFetch.mockReset();
  });

  afterEach(() => {
    resetServerInfoCache();
    queryClient.clear();
    global.fetch = originalFetch;
  });

  test('should enable workspaces when server returns enabled', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ store_type: 'SqlStore', workspaces_enabled: true }),
    } as Response);

    renderWithProviders(<WorkspacesTestComponent />, queryClient);

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
      json: async () => ({ store_type: 'SqlStore', workspaces_enabled: false }),
    } as Response);

    renderWithProviders(<WorkspacesTestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
    expect(getWorkspacesEnabledSync()).toBe(false);
  });

  test('should disable workspaces on network error', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'));

    renderWithProviders(<WorkspacesTestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
    expect(getWorkspacesEnabledSync()).toBe(false);
  });

  test('should disable workspaces on server error', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    } as Response);

    renderWithProviders(<WorkspacesTestComponent />, queryClient);

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('loaded');
    });

    expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
    expect(getWorkspacesEnabledSync()).toBe(false);
  });

  test('getWorkspacesEnabledSync returns false before fetch completes', () => {
    const freshQueryClient = createTestQueryClient();
    mockFetch.mockImplementation(() => new Promise(() => {}));

    // Before render, no queryClient reference is set
    expect(getWorkspacesEnabledSync()).toBe(false);

    // Render to set up the queryClient reference
    renderWithProviders(<WorkspacesTestComponent />, freshQueryClient);

    // While still loading (fetch not complete), should return false
    expect(getWorkspacesEnabledSync()).toBe(false);
  });
});
