import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import React from 'react';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { workspaceFetch } from '@databricks/web-shared/spog/workspace-console';
import { useDatasetsPageQuery } from './useDatasetsPageQuery';

jest.mock('@databricks/web-shared/spog/workspace-console', () => ({
  workspaceFetch: jest.fn(),
}));

const createWrapper = () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
    logger: { error: () => {}, log: () => {}, warn: () => {} },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
};

const mockOk = (datasets: unknown[] = []) => {
  jest.mocked(workspaceFetch).mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve({ datasets }),
  } as any);
};

const lastFilterParam = () => {
  const [url] = jest.mocked(workspaceFetch).mock.calls.at(-1) ?? [];
  if (!url) throw new Error('workspaceFetch was not called');
  // The hook builds `${ajaxUrl}?${params}`; URLSearchParams round-trips the filter literal.
  const search = String(url).split('?')[1] ?? '';
  return new URLSearchParams(search).get('filter');
};

describe('useDatasetsPageQuery filter building', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('builds a plain experiment_id clause when nameFilter is empty', async () => {
    mockOk();
    const { result } = renderHook(
      () => useDatasetsPageQuery({ experimentId: 'exp-1', nameFilter: '', pageSize: 25, pageToken: undefined }),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(lastFilterParam()).toBe("experiment_id='exp-1'");
  });

  test('escapes single quotes inside the nameFilter to prevent breaking the filter literal', async () => {
    mockOk();
    const { result } = renderHook(
      () =>
        useDatasetsPageQuery({
          experimentId: 'exp-1',
          nameFilter: "O'Brien",
          pageSize: 25,
          pageToken: undefined,
        }),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(lastFilterParam()).toBe("experiment_id='exp-1' AND name ILIKE '%O''Brien%'");
  });

  test('strips literal % and _ from the user search so they do not become unintended LIKE wildcards', async () => {
    mockOk();
    const { result } = renderHook(
      () =>
        useDatasetsPageQuery({
          experimentId: 'exp-1',
          nameFilter: '100%_off',
          pageSize: 25,
          pageToken: undefined,
        }),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // Backend filter parser (managed-evals/DatasetSearchFilter.scala) doesn't support ESCAPE,
    // so we can't escape these — we strip them and accept that searching for literal % or _
    // is not supported. The outer % characters are part of the ILIKE substring match, but
    // nothing inside the user-supplied value should contain a wildcard character.
    expect(lastFilterParam()).toBe("experiment_id='exp-1' AND name ILIKE '%100off%'");
  });

  test('drops the ILIKE clause entirely if the search is purely wildcards (would otherwise match everything)', async () => {
    mockOk();
    const { result } = renderHook(
      () =>
        useDatasetsPageQuery({
          experimentId: 'exp-1',
          nameFilter: '%_%',
          pageSize: 25,
          pageToken: undefined,
        }),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // After stripping, the sanitized value is empty — emitting `name ILIKE '%%'` would match
    // every row, making the search bar appear broken. Skipping the clause is the right call.
    expect(lastFilterParam()).toBe("experiment_id='exp-1'");
  });
});
