import { describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, act, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../common/utils/setup-msw';
import { useSkillsListQuery } from './useSkillsListQuery';
import { createMockSkill } from '../test-utils';

const BASE_URL = 'ajax-api/3.0/mlflow/skills';

describe('useSkillsListQuery', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  const mockServer = setupServer(
    rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) =>
      res(ctx.json({ skills: [createMockSkill()], next_page_token: 'page-2-token' })),
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
    const { result } = renderHook(() => useSkillsListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toHaveLength(1);
    expect(result.current.hasNextPage).toBe(true);
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('treats a null next_page_token as the last page', async () => {
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) =>
        res(ctx.json({ skills: [createMockSkill()], next_page_token: null })),
      ),
    );

    const { result } = renderHook(() => useSkillsListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.hasNextPage).toBe(false);
  });

  it('navigates to the next page and back using the token stack', async () => {
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        const token = req.url.searchParams.get('page_token');
        if (token === 'page-2-token') {
          return res(ctx.json({ skills: [createMockSkill({ name: 'page2' })], next_page_token: 'page-3-token' }));
        }
        return res(ctx.json({ skills: [createMockSkill({ name: 'page1' })], next_page_token: 'page-2-token' }));
      }),
    );

    const { result } = renderHook(() => useSkillsListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(result.current.data?.[0]?.name).toBe('page2');
    });
    expect(result.current.hasPreviousPage).toBe(true);

    act(() => {
      result.current.onPreviousPage();
    });

    await waitFor(() => {
      expect(result.current.data?.[0]?.name).toBe('page1');
    });
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('sends catalog filters as a single filter_string', async () => {
    let capturedFilter: string | null = null;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ skills: [], next_page_token: null }));
      }),
    );

    const { result } = renderHook(
      () =>
        useSkillsListQuery({
          searchText: 'review',
          filterActive: true,
          organization: 'acme',
          tagKey: 'team',
          tagValue: 'platform',
          sourceType: 'git',
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(capturedFilter).toBe(
      "search_text ILIKE '%review%' AND status = 'active' AND organization = 'acme' AND tags.team = 'platform' AND source_type = 'git'",
    );
  });

  it('omits filter_string when no catalog filters are set', async () => {
    let capturedFilter: string | null = null;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ skills: [], next_page_token: null }));
      }),
    );

    const { result } = renderHook(() => useSkillsListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(capturedFilter).toBeNull();
  });

  it('resets pagination when catalog filters change', async () => {
    const capturedTokens: (string | null)[] = [];
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedTokens.push(req.url.searchParams.get('page_token'));
        return res(ctx.json({ skills: [createMockSkill()], next_page_token: 'next' }));
      }),
    );

    let filterActive = false;
    const { result, rerender } = renderHook(() => useSkillsListQuery({ filterActive }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    act(() => {
      result.current.onNextPage();
    });

    await waitFor(() => {
      expect(capturedTokens).toContain('next');
    });

    filterActive = true;
    rerender();

    await waitFor(() => {
      expect(capturedTokens[capturedTokens.length - 1]).toBeNull();
    });
    expect(result.current.hasPreviousPage).toBe(false);
  });

  it('sends max_results matching the default page size', async () => {
    let capturedMaxResults: string | null = null;
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedMaxResults = req.url.searchParams.get('max_results');
        return res(ctx.json({ skills: [createMockSkill()], next_page_token: null }));
      }),
    );

    const { result } = renderHook(() => useSkillsListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(capturedMaxResults).toBe('25');
  });

  it('returns an error when the API fails', async () => {
    mockServer.use(
      rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) => res(ctx.status(500), ctx.json({ message: 'Server error' }))),
    );

    const { result } = renderHook(() => useSkillsListQuery({}), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.data).toBeUndefined();
  });
});
