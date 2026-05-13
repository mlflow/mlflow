import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

/**
 * Creates a QueryClientProvider wrapper for use with renderHook in tests.
 * Configures retry: false and silences console output.
 */
export function createQueryWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
    logger: {
      error: () => {},
      log: () => {},
      warn: () => {},
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

/**
 * Creates a mock successful fetch response that resolves with the given data.
 */
export function mockFetchResponse(data: unknown) {
  return {
    ok: true,
    status: 200,
    json: () => Promise.resolve(data),
  } as any;
}
