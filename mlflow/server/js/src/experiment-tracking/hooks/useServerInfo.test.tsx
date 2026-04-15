import { describe, beforeEach, afterEach, test, expect } from '@jest/globals';
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
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
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
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
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
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
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
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
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
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
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
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
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
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
  });

  afterEach(() => {
    resetServerInfoCache();
    queryClient.clear();
  });

  describe('when server returns workspaces enabled', () => {
    setupServer(
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
        return res(ctx.json({ store_type: 'SqlStore', workspaces_enabled: true }));
      }),
    );

    test('should enable workspaces', async () => {
      renderWithProviders(<WorkspacesTestComponent />, queryClient);

      await waitFor(() => {
        expect(screen.getByTestId('loading').textContent).toBe('loaded');
      });

      expect(screen.getByTestId('workspaces-enabled').textContent).toBe('true');
      expect(getWorkspacesEnabledSync()).toBe(true);
    });
  });

  describe('when server returns workspaces disabled', () => {
    setupServer(
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
        return res(ctx.json({ store_type: 'SqlStore', workspaces_enabled: false }));
      }),
    );

    test('should disable workspaces', async () => {
      renderWithProviders(<WorkspacesTestComponent />, queryClient);

      await waitFor(() => {
        expect(screen.getByTestId('loading').textContent).toBe('loaded');
      });

      expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
      expect(getWorkspacesEnabledSync()).toBe(false);
    });
  });

  describe('when network error occurs', () => {
    setupServer(
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res) => {
        return res.networkError('Failed to connect');
      }),
    );

    test('should disable workspaces', async () => {
      renderWithProviders(<WorkspacesTestComponent />, queryClient);

      await waitFor(() => {
        expect(screen.getByTestId('loading').textContent).toBe('loaded');
      });

      expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
      expect(getWorkspacesEnabledSync()).toBe(false);
    });
  });

  describe('when server error occurs', () => {
    setupServer(
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
        return res(ctx.status(500));
      }),
    );

    test('should disable workspaces', async () => {
      renderWithProviders(<WorkspacesTestComponent />, queryClient);

      await waitFor(() => {
        expect(screen.getByTestId('loading').textContent).toBe('loaded');
      });

      expect(screen.getByTestId('workspaces-enabled').textContent).toBe('false');
      expect(getWorkspacesEnabledSync()).toBe(false);
    });
  });

  describe('when fetch has not completed', () => {
    setupServer(
      rest.get('/ajax-api/3.0/mlflow/server-info', (_req, res, ctx) => {
        return res(ctx.json({ store_type: 'SqlStore', workspaces_enabled: false }));
        // Never resolve to simulate pending request
        return new Promise(() => {});
      }),
    );

    test('getWorkspacesEnabledSync returns false before fetch completes', () => {
      // Before render, no queryClient reference is set
      expect(getWorkspacesEnabledSync()).toBe(false);

      const freshQueryClient = createTestQueryClient();
      renderWithProviders(<WorkspacesTestComponent />, freshQueryClient);

      // While still loading (fetch not complete), should return false
      expect(getWorkspacesEnabledSync()).toBe(false);
    });
  });
});
