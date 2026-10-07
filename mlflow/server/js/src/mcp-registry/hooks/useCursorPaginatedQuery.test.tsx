import { describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, act, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../common/utils/setup-msw';
import { useCursorPaginatedQuery } from './useCursorPaginatedQuery';

const BASE_URL = 'ajax-api/3.0/mlflow/test-endpoint';

interface TestResponse {
  items: string[];
  next_page_token?: string;
}

describe('useCursorPaginatedQuery', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  const mockServer = setupServer(
    rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) =>
      res(ctx.json({ items: ['item-1', 'item-2'], next_page_token: 'page-2-token' })),
    ),
  );

  const createWrapper = () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return ({ children }: { children: React.ReactNode }) => (
      <IntlProvider locale="en">
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </IntlProvider>
    );
  };

  const defaultOptions = {
    queryKeyPrefix: 'test_query',
    storageKey: 'test.page_size',
    queryFn: ({
      searchFilter,
      pageToken,
      pageSize,
    }: {
      searchFilter?: string;
      pageToken?: string;
      pageSize: number;
    }) => {
      const params = new URLSearchParams();
      if (searchFilter) params.set('filter', searchFilter);
      if (pageToken) params.set('page_token', pageToken);
      params.set('max_results', String(pageSize));
      return fetch(getAjaxUrl(`${BASE_URL}?${params.toString()}`)).then((r) => r.json()) as Promise<TestResponse>;
    },
    extractData: (response: TestResponse) => response.items,
  };

  it('returns first page of data', async () => {
    const { result } = renderHook(() => useCursorPaginatedQuery(defaultOptions), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(['item-1', 'item-2']);
    expect(result.current.hasNextPage).toBe(true);
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('onNextPage advances the page token', async () => {
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        const token = req.url.searchParams.get('page_token');
        if (token === 'page-2-token') {
          return res(ctx.json({ items: ['page-2-item'], next_page_token: undefined }));
        }
        return res(ctx.json({ items: ['page-1-item'], next_page_token: 'page-2-token' }));
      }),
    );

    const { result } = renderHook(() => useCursorPaginatedQuery(defaultOptions), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.data).toEqual(['page-1-item']);

    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(result.current.data).toEqual(['page-2-item']);
    });
    expect(result.current.hasPreviousPage).toBe(true);
    expect(result.current.hasNextPage).toBe(false);
  });

  it('onPreviousPage goes back to previous token', async () => {
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        const token = req.url.searchParams.get('page_token');
        if (token === 'page-2-token') {
          return res(ctx.json({ items: ['page-2-item'], next_page_token: 'page-3-token' }));
        }
        return res(ctx.json({ items: ['page-1-item'], next_page_token: 'page-2-token' }));
      }),
    );

    const { result } = renderHook(() => useCursorPaginatedQuery(defaultOptions), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.data).toEqual(['page-1-item']);
    });

    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(result.current.data).toEqual(['page-2-item']);
    });

    act(() => {
      result.current.onPreviousPage();
    });

    await waitFor(() => {
      expect(result.current.data).toEqual(['page-1-item']);
    });
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('filter change resets pagination', async () => {
    const capturedTokens: (string | null)[] = [];
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedTokens.push(req.url.searchParams.get('page_token'));
        return res(ctx.json({ items: ['item'], next_page_token: 'next' }));
      }),
    );

    const { result, rerender } = renderHook(
      ({ filter }: { filter?: string }) => useCursorPaginatedQuery({ ...defaultOptions, searchFilter: filter }),
      { wrapper: createWrapper(), initialProps: { filter: undefined as string | undefined } },
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Navigate to page 2
    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(capturedTokens).toContain('next');
    });

    // Change filter — should reset token
    rerender({ filter: 'new-filter' });

    await waitFor(() => {
      const lastToken = capturedTokens[capturedTokens.length - 1];
      expect(lastToken).toBeNull();
    });
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('extraQueryKeys change resets pagination', async () => {
    const capturedTokens: (string | null)[] = [];
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedTokens.push(req.url.searchParams.get('page_token'));
        return res(ctx.json({ items: ['item'], next_page_token: 'next' }));
      }),
    );

    const { result, rerender } = renderHook(
      ({ extra }: { extra: Record<string, unknown> }) =>
        useCursorPaginatedQuery({ ...defaultOptions, extraQueryKeys: extra }),
      { wrapper: createWrapper(), initialProps: { extra: { availableOnly: true } } },
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(capturedTokens).toContain('next');
    });

    rerender({ extra: { availableOnly: false } });

    await waitFor(() => {
      const lastToken = capturedTokens[capturedTokens.length - 1];
      expect(lastToken).toBeNull();
    });
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('enabled=false prevents query from firing', async () => {
    let requestCount = 0;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) => {
        requestCount++;
        return res(ctx.json({ items: ['item'], next_page_token: undefined }));
      }),
    );

    const { result } = renderHook(() => useCursorPaginatedQuery({ ...defaultOptions, enabled: false }), {
      wrapper: createWrapper(),
    });

    // Wait a tick to ensure no request fires
    await new Promise((r) => setTimeout(r, 50));
    expect(result.current.data).toBeUndefined();
    expect(result.current.isLoading).toBe(true);
    expect(requestCount).toBe(0);
  });
});
