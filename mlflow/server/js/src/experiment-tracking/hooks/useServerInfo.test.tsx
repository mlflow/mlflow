import { describe, test, expect } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../common/utils/setup-msw';
import { renderHook, waitFor } from '@testing-library/react';
import { useServerInfo, useIsFileStore } from './useServerInfo';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

describe('useServerInfo', () => {
  describe('when backend returns FileStore', () => {
    setupServer(
      rest.get('/server-info', (_req, res, ctx) => {
        return res(ctx.json({ store_type: 'FileStore' }));
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
        return res(ctx.json({ store_type: 'SqlAlchemyStore' }));
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
        return res(ctx.json({ store_type: 'FileStore' }));
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
        return res(ctx.json({ store_type: 'SqlAlchemyStore' }));
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
