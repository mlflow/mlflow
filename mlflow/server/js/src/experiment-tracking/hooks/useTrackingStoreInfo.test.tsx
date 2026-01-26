import { describe, test, expect } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../common/utils/setup-msw';
import { renderHook, waitFor } from '@testing-library/react';
import { useTrackingStoreInfo, useIsFileStore } from './useTrackingStoreInfo';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

describe('useTrackingStoreInfo', () => {
  describe('when backend returns FileStore', () => {
    setupServer(
      rest.get('/tracking-store-info', (_req, res, ctx) => {
        return res(ctx.json({ is_file_store: true }));
      }),
    );

    test('should return is_file_store as true', async () => {
      const { result } = renderHook(() => useTrackingStoreInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data?.is_file_store).toBe(true);
    });
  });

  describe('when backend returns SQLAlchemyStore', () => {
    setupServer(
      rest.get('/tracking-store-info', (_req, res, ctx) => {
        return res(ctx.json({ is_file_store: false }));
      }),
    );

    test('should return is_file_store as false', async () => {
      const { result } = renderHook(() => useTrackingStoreInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data?.is_file_store).toBe(false);
    });
  });

  describe('when backend returns an error', () => {
    setupServer(
      rest.get('/tracking-store-info', (_req, res, ctx) => {
        return res(ctx.status(500));
      }),
    );

    test('should return default value (is_file_store: false)', async () => {
      const { result } = renderHook(() => useTrackingStoreInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data?.is_file_store).toBe(false);
    });
  });

  describe('when endpoint does not exist (404)', () => {
    setupServer(
      rest.get('/tracking-store-info', (_req, res, ctx) => {
        return res(ctx.status(404));
      }),
    );

    test('should return default value (is_file_store: false)', async () => {
      const { result } = renderHook(() => useTrackingStoreInfo(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data?.is_file_store).toBe(false);
    });
  });
});

describe('useIsFileStore', () => {
  describe('when backend uses FileStore', () => {
    setupServer(
      rest.get('/tracking-store-info', (_req, res, ctx) => {
        return res(ctx.json({ is_file_store: true }));
      }),
    );

    test('should return true', async () => {
      const { result } = renderHook(() => useIsFileStore(), { wrapper });

      await waitFor(() => {
        expect(result.current).toBe(true);
      });
    });
  });

  describe('when backend uses SQLAlchemyStore', () => {
    setupServer(
      rest.get('/tracking-store-info', (_req, res, ctx) => {
        return res(ctx.json({ is_file_store: false }));
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
