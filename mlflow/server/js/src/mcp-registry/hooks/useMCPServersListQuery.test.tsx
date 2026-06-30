import { describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, act, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../common/utils/setup-msw';
import { useMCPServersListQuery } from './useMCPServersListQuery';
import { createMockMCPServer } from '../test-utils';

const BASE_URL = 'ajax-api/3.0/mlflow/mcp-servers';

describe('useMCPServersListQuery', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  const mockServer = setupServer(
    rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) =>
      res(ctx.json({ mcp_servers: [createMockMCPServer()], next_page_token: 'page-2-token' })),
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

  it('returns data and pagination state on initial load', async () => {
    const { result } = renderHook(() => useMCPServersListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toHaveLength(1);
    expect(result.current.hasNextPage).toBe(true);
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('navigates to next page and back using token stack', async () => {
    const capturedTokens: (string | null)[] = [];
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedTokens.push(req.url.searchParams.get('page_token'));
        const token = req.url.searchParams.get('page_token');
        if (token === 'page-2-token') {
          return res(
            ctx.json({ mcp_servers: [createMockMCPServer({ name: 'page2' })], next_page_token: 'page-3-token' }),
          );
        }
        return res(
          ctx.json({ mcp_servers: [createMockMCPServer({ name: 'page1' })], next_page_token: 'page-2-token' }),
        );
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Go to page 2
    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(result.current.data?.[0]?.name).toBe('page2');
    });
    expect(result.current.hasPreviousPage).toBe(true);

    // Go back to page 1
    act(() => {
      result.current.onPreviousPage();
    });

    await waitFor(() => {
      expect(result.current.data?.[0]?.name).toBe('page1');
    });
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('resets pagination when searchFilter changes', async () => {
    const capturedTokens: (string | null)[] = [];
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedTokens.push(req.url.searchParams.get('page_token'));
        return res(ctx.json({ mcp_servers: [createMockMCPServer()], next_page_token: 'next' }));
      }),
    );

    let searchFilter = '';
    const { result, rerender } = renderHook(() => useMCPServersListQuery({ searchFilter }), {
      wrapper: createWrapper(),
    });

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

    // Change filter — should reset
    searchFilter = 'new-filter';
    rerender();

    await waitFor(() => {
      const lastToken = capturedTokens[capturedTokens.length - 1];
      expect(lastToken).toBeNull();
    });
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('provides pageSizeSelect config with correct defaults', async () => {
    const { result } = renderHook(() => useMCPServersListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.pageSizeSelect).toBeDefined();
    expect(result.current.pageSizeSelect!.options).toEqual([10, 25, 50, 100]);
    expect(result.current.pageSizeSelect!.default).toBe(25);
  });

  it('resets pagination when page size changes', async () => {
    const capturedParams: { token: string | null; maxResults: string | null }[] = [];
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedParams.push({
          token: req.url.searchParams.get('page_token'),
          maxResults: req.url.searchParams.get('max_results'),
        });
        return res(ctx.json({ mcp_servers: [createMockMCPServer()], next_page_token: 'next' }));
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Navigate to page 2
    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(capturedParams.some((p) => p.token === 'next')).toBe(true);
    });

    // Change page size — should reset token and use new max_results
    act(() => {
      result.current.pageSizeSelect!.onChange(50);
    });

    await waitFor(() => {
      const lastParam = capturedParams[capturedParams.length - 1];
      expect(lastParam.token).toBeNull();
      expect(lastParam.maxResults).toBe('50');
    });
  });

  it('sends max_results matching default page size', async () => {
    let capturedMaxResults: string | null = null;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedMaxResults = req.url.searchParams.get('max_results');
        return res(ctx.json({ mcp_servers: [createMockMCPServer()], next_page_token: undefined }));
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(capturedMaxResults).toBe('25');
  });

  it('returns error when API fails', async () => {
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) => res(ctx.status(500), ctx.json({ message: 'Server error' }))),
    );

    const { result } = renderHook(() => useMCPServersListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.data).toBeUndefined();
  });

  it('wraps plain text search in ILIKE clause', async () => {
    let capturedFilter: string | null = null;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ mcp_servers: [], next_page_token: undefined }));
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({ searchFilter: 'github' }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(capturedFilter).toBe("name ILIKE '%github%'");
  });

  it('passes SQL filter syntax through without modification', async () => {
    let capturedFilter: string | null = null;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ mcp_servers: [], next_page_token: undefined }));
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({ searchFilter: "tags.env = 'production'" }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(capturedFilter).toBe("tags.env = 'production'");
  });

  it('escapes ILIKE wildcards in plain text search', async () => {
    let capturedFilter: string | null = null;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ mcp_servers: [], next_page_token: undefined }));
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({ searchFilter: 'my_server%' }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(capturedFilter).toBe("name ILIKE '%my\\_server\\%%'");
  });

  it('ignores onNextPage while query is fetching', async () => {
    let callCount = 0;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        callCount++;
        const token = req.url.searchParams.get('page_token');
        if (token) {
          // Delay page-2 response so isFetching stays true during the second click
          return res(
            ctx.delay(500),
            ctx.json({ mcp_servers: [createMockMCPServer({ name: 'page2' })], next_page_token: 'page-3' }),
          );
        }
        return res(ctx.json({ mcp_servers: [createMockMCPServer()], next_page_token: 'page-2' }));
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    const callsAfterInitialLoad = callCount;

    // First click triggers fetch for page 2 (slow response)
    act(() => {
      result.current.onNextPage();
    });
    // Second click should be ignored because isFetching is true
    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(result.current.data?.[0]?.name).toBe('page2');
    });

    // Only one additional API call should have been made (the second click was ignored)
    expect(callCount - callsAfterInitialLoad).toBe(1);
  });

  it('escapes single quotes in plain text search', async () => {
    let capturedFilter: string | null = null;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ mcp_servers: [], next_page_token: undefined }));
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({ searchFilter: "O'Brien" }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(capturedFilter).toBe("name ILIKE '%O''Brien%'");
  });

  it('ignores onPreviousPage while query is fetching', async () => {
    let requestCount = 0;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        requestCount++;
        const token = req.url.searchParams.get('page_token');
        if (token === 'page-2') {
          // Delay page 2 response so isFetching stays true
          return res(
            ctx.delay(500),
            ctx.json({ mcp_servers: [createMockMCPServer({ name: 'page2' })], next_page_token: undefined }),
          );
        }
        return res(ctx.json({ mcp_servers: [createMockMCPServer()], next_page_token: 'page-2' }));
      }),
    );

    const { result } = renderHook(() => useMCPServersListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Navigate to page 2 (slow response)
    act(() => {
      result.current.onNextPage();
    });

    // Immediately try going back while page 2 is still fetching
    act(() => {
      result.current.onPreviousPage();
    });

    // Wait for page 2 to arrive
    await waitFor(() => {
      expect(result.current.data?.[0]?.name).toBe('page2');
    });

    // Should still be on page 2 (onPreviousPage was ignored)
    expect(result.current.hasPreviousPage).toBe(true);
  });
});
