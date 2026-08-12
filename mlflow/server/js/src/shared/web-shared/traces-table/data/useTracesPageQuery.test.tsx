import { describe, expect, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { rest } from 'msw';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';
import { setupServer } from '../../test-utils/setup-msw';
import { useTracesPageQuery, type TracesQueryIdentity } from './useTracesPageQuery';
import { useTraceTokenCache } from './useTraceTokenCache';
import { makeTraces } from '../test-utils/mockTraces';

// OSS serves the synchronous 3.0 endpoint; its response lists rows under `traces` (remapped to
// `trace_infos` by the adapter). The progressive 4.0 endpoint is Databricks-only (dead in OSS).
const SEARCH_URL = '*/ajax-api/3.0/mlflow/traces/search';
const PROGRESSIVE_URL = '*/ajax-api/4.0/mlflow/traces/search-progressive';

const IDENTITY: TracesQueryIdentity = {
  locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } }],
  pageSize: 2,
};

describe('useTracesPageQuery', () => {
  const { server } = setupServer();

  const makeWrapper = () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };

  test('fetches the first page and reports last-page when it is short', async () => {
    server.use(
      rest.post(SEARCH_URL, (_req, res, ctx) =>
        // One row for a page size of 2 → a short page → treated as the last page.
        res(ctx.json({ traces: makeTraces(1), next_page_token: 'tok-2' })),
      ),
    );

    let page = 1;
    const { result } = renderHook(
      () => {
        const tokenCache = useTraceTokenCache();
        return useTracesPageQuery({
          identity: IDENTITY,
          pageIndex: page,
          tokenCache,
          enabled: true,
          onPageIndexChange: (next) => {
            page = next;
          },
        });
      },
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.traces).toHaveLength(1));
    expect(result.current.hasPrev).toBe(false);
    // Short page → last page even though the server sent a token.
    expect(result.current.hasNext).toBe(false);
  });

  test('a full page enables Next; paging forward sends the recorded cursor', async () => {
    const seenTokens: (string | undefined)[] = [];
    server.use(
      rest.post(SEARCH_URL, async (req, res, ctx) => {
        const body = (await req.json()) as { page_token?: string };
        seenTokens.push(body.page_token);
        // Page 1 (no token) → full page + a cursor. Page 2 (tok-2) → short page (last).
        if (!body.page_token) {
          return res(ctx.json({ traces: makeTraces(2), next_page_token: 'tok-2' }));
        }
        return res(ctx.json({ traces: makeTraces(1, 'p2'), next_page_token: '' }));
      }),
    );

    let page = 1;
    const { result, rerender } = renderHook(
      ({ pageIndex }: { pageIndex: number }) => {
        const tokenCache = useTraceTokenCache();
        return useTracesPageQuery({
          identity: IDENTITY,
          pageIndex,
          tokenCache,
          enabled: true,
          onPageIndexChange: (next) => {
            page = next;
          },
        });
      },
      { wrapper: makeWrapper(), initialProps: { pageIndex: 1 } },
    );

    await waitFor(() => expect(result.current.traces).toHaveLength(2));
    expect(result.current.hasNext).toBe(true);

    // Advance to page 2 through goToPage (writes `page`), then re-render at that index.
    act(() => result.current.goToPage(2));
    expect(page).toBe(2);
    rerender({ pageIndex: 2 });

    await waitFor(() => expect(result.current.traces).toHaveLength(1));
    // Page 2's request used the cursor recorded from page 1's response.
    expect(seenTokens).toContain('tok-2');
    expect(result.current.hasPrev).toBe(true);
    expect(result.current.hasNext).toBe(false);
  });

  test('uses the progressive transport when useProgressiveSearch is set; token cache still drives nav', async () => {
    const progressiveTokens: (string | undefined)[] = [];
    let searchHits = 0;
    server.use(
      rest.post(SEARCH_URL, (_req, res, ctx) => {
        searchHits += 1;
        return res(ctx.json({ traces: makeTraces(2), next_page_token: 'tok-2' }));
      }),
      rest.post(PROGRESSIVE_URL, async (req, res, ctx) => {
        const body = (await req.json()) as { page_token?: string };
        progressiveTokens.push(body.page_token);
        // Page 1 (no token) → full page + a cursor. Page 2 (tok-2) → short page (last).
        if (!body.page_token) {
          return res(
            ctx.json({ name: 'op', done: true, response: { trace_infos: makeTraces(2), next_page_token: 'tok-2' } }),
          );
        }
        return res(
          ctx.json({ name: 'op', done: true, response: { trace_infos: makeTraces(1, 'p2'), next_page_token: '' } }),
        );
      }),
    );

    let page = 1;
    const { result, rerender } = renderHook(
      ({ pageIndex }: { pageIndex: number }) => {
        const tokenCache = useTraceTokenCache();
        return useTracesPageQuery({
          identity: IDENTITY,
          pageIndex,
          tokenCache,
          enabled: true,
          onPageIndexChange: (next) => {
            page = next;
          },
          useProgressiveSearch: true,
        });
      },
      { wrapper: makeWrapper(), initialProps: { pageIndex: 1 } },
    );

    await waitFor(() => expect(result.current.traces).toHaveLength(2));
    // The synchronous transport was never used — progressive fully replaced it.
    expect(searchHits).toBe(0);
    expect(result.current.hasNext).toBe(true);

    act(() => result.current.goToPage(2));
    expect(page).toBe(2);
    rerender({ pageIndex: 2 });

    await waitFor(() => expect(result.current.traces).toHaveLength(1));
    // Page 2's request used the cursor recorded from page 1's response.
    expect(progressiveTokens).toContain('tok-2');
    expect(result.current.hasPrev).toBe(true);
    expect(result.current.hasNext).toBe(false);
  });

  test('is disabled (no fetch) when enabled=false', async () => {
    let requested = false;
    server.use(
      rest.post(SEARCH_URL, (_req, res, ctx) => {
        requested = true;
        return res(ctx.json({ traces: [], next_page_token: '' }));
      }),
    );

    const { result } = renderHook(
      () => {
        const tokenCache = useTraceTokenCache();
        return useTracesPageQuery({
          identity: IDENTITY,
          pageIndex: 1,
          tokenCache,
          enabled: false,
          onPageIndexChange: () => {},
        });
      },
      { wrapper: makeWrapper() },
    );

    expect(result.current.isLoading).toBe(false);
    expect(result.current.traces).toEqual([]);
    expect(requested).toBe(false);
  });

  test('surfaces a server error', async () => {
    server.use(rest.post(SEARCH_URL, (_req, res, ctx) => res(ctx.status(500), ctx.json({ message: 'boom' }))));

    const { result } = renderHook(
      () => {
        const tokenCache = useTraceTokenCache();
        return useTracesPageQuery({
          identity: IDENTITY,
          pageIndex: 1,
          tokenCache,
          enabled: true,
          onPageIndexChange: () => {},
        });
      },
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.error).toBeTruthy());
  });
});
