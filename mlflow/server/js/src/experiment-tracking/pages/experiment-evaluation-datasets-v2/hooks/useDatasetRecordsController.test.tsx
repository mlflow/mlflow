// @ts-nocheck — punting test typing; see PR2 plan in branch import { type ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from '@mlflow/mlflow/src/common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  setupTestRouter,
  testRoute,
  TestRouter,
  waitForRoutesToBeRendered,
} from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import type { Dataset, DatasetRecord } from '../hooks/useDatasetsQueries';
import { useDatasetRecordsController } from './useDatasetRecordsController';

const mockDataset: Dataset = {
  dataset_id: 'd1',
  create_time: '2024-01-01T00:00:00Z',
  name: 'Test dataset',
};

const makeRecords = (count: number): DatasetRecord[] =>
  Array.from(
    { length: count },
    (_, i): DatasetRecord => ({
      dataset_record_id: `rec-${i + 1}`,
      create_time: '2024-01-01T00:00:00Z',
      inputs: { question: `q${i + 1}` },
    }),
  );

interface MockState {
  dataset: Dataset;
  records: DatasetRecord[];
}

// Shared in-memory state — handler closures read this on every request so mutations between
// assertions (e.g. shrinking the record set) are visible on the next refetch.
const state: MockState = { dataset: mockDataset, records: [] };

// useListDatasetRecordsQuery (in @databricks/web-shared) consumes the REST shape:
// inputs as `{ key, value }[]`, expectations as `{ [key]: { value } }`.
const toRestRecord = (record: DatasetRecord) => ({
  ...record,
  inputs: Object.entries(record.inputs ?? {}).map(([key, value]) => ({ key, value })),
  expectations: record.expectations
    ? Object.entries(record.expectations).reduce<Record<string, { value: unknown }>>((acc, [key, value]) => {
        acc[key] = { value };
        return acc;
      }, {})
    : undefined,
});

const server = setupServer(
  rest.get('/ajax-api/2.0/managed-evals/datasets/:datasetId', (_req, res, ctx) => res(ctx.json(state.dataset))),
  rest.get('/ajax-api/2.0/managed-evals/datasets/:datasetId/records', (_req, res, ctx) =>
    res(ctx.json({ dataset_records: state.records.map(toRestRecord) })),
  ),
);

interface ControllerHookOptions {
  initialUrl: string;
  history: ReturnType<typeof setupTestRouter>['history'];
}

async function renderController({ initialUrl, history }: ControllerHookOptions) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <TestRouter routes={[testRoute(<div>{children}</div>)]} history={history} initialEntries={[initialUrl]} />
    </QueryClientProvider>
  );

  const result = renderHook(() => useDatasetRecordsController({ experimentId: 'e1', datasetId: 'd1' }), { wrapper });
  await waitForRoutesToBeRendered();
  return { ...result, history, queryClient };
}

describe('useDatasetRecordsController', () => {
  // setupTestRouter registers beforeAll/afterAll hooks; must live at describe scope.
  const { history } = setupTestRouter();

  beforeEach(() => {
    window.localStorage.clear();
    state.dataset = mockDataset;
    state.records = [];
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('clamps URL page when totalRecords shrinks below the requested page', async () => {
    state.records = makeRecords(100);
    const { result, queryClient } = await renderController({
      initialUrl: '/p?page=3',
      history,
    });

    await waitFor(() => expect(result.current.records.isLoading).toBe(false));
    expect(result.current.url.pageIndex).toBe(3);

    // Shrink the data behind the controller; force a refetch.
    state.records = makeRecords(30);
    await act(async () => {
      await queryClient.invalidateQueries(['listDatasetRecords', 'd1']);
    });

    // 30 records / 25 per page = 2 valid pages; page 3 should snap back to 2.
    await waitFor(() => expect(result.current.url.pageIndex).toBe(2));
  });

  test('selectedRecord is undefined and URL recordId is cleared when no match exists', async () => {
    state.records = makeRecords(5);
    const { result } = await renderController({
      initialUrl: '/p?recordId=missing',
      history,
    });

    await waitFor(() => expect(result.current.records.isLoading).toBe(false));
    await waitFor(() => expect(result.current.url.recordId).toBeUndefined());
    expect(result.current.selectedRecord).toBeUndefined();
  });

  test('selectedRecord resolves to the matching record when the URL points at a real id', async () => {
    state.records = makeRecords(5);
    const { result } = await renderController({
      initialUrl: '/p?recordId=rec-3',
      history,
    });

    await waitFor(() => expect(result.current.records.isLoading).toBe(false));
    expect(result.current.selectedRecord?.dataset_record_id).toBe('rec-3');
    expect(result.current.url.recordId).toBe('rec-3');
  });

  test('bulk.selected clears when the search query changes', async () => {
    state.records = makeRecords(3);
    const { result } = await renderController({
      initialUrl: '/p',
      history,
    });

    await waitFor(() => expect(result.current.records.isLoading).toBe(false));

    act(() => {
      result.current.bulk.toggle('rec-1');
      result.current.bulk.toggle('rec-2');
    });
    expect(result.current.bulk.selected.size).toBe(2);

    act(() => {
      result.current.url.setSearch('hello');
    });
    await waitFor(() => expect(result.current.bulk.selected.size).toBe(0));
  });

  test('flags.hasNoRecordsAtAll is true on an empty dataset with no active search', async () => {
    state.records = [];
    const { result } = await renderController({
      initialUrl: '/p',
      history,
    });

    await waitFor(() => expect(result.current.records.isLoading).toBe(false));
    expect(result.current.flags.hasNoRecordsAtAll).toBe(true);
    expect(result.current.flags.hasNoSearchResults).toBe(false);
    expect(result.current.flags.hasActiveSearch).toBe(false);
  });

  test('flags.hasNoSearchResults is true when filter excludes every record', async () => {
    state.records = makeRecords(5);
    const { result } = await renderController({
      initialUrl: '/p?q=this-substring-will-never-match-any-record',
      history,
    });

    await waitFor(() => expect(result.current.records.isLoading).toBe(false));
    expect(result.current.flags.hasActiveSearch).toBe(true);
    expect(result.current.flags.hasNoSearchResults).toBe(true);
    expect(result.current.flags.hasNoRecordsAtAll).toBe(false);
  });

  test('flags collapse to false when matching records exist', async () => {
    state.records = makeRecords(5);
    const { result } = await renderController({
      initialUrl: '/p?q=q1',
      history,
    });

    await waitFor(() => expect(result.current.records.isLoading).toBe(false));
    expect(result.current.flags.hasActiveSearch).toBe(true);
    expect(result.current.flags.hasNoSearchResults).toBe(false);
    expect(result.current.flags.hasNoRecordsAtAll).toBe(false);
  });

  test('setPageIndex flushes the pending debounced search write before navigating', async () => {
    // Reproduces the race: schedule a debounced search via setSearchInput, then immediately
    // page-navigate. Without the flush, setSearch lands after setPageIndex and removes the
    // page param, snapping the user back to page 1.
    // Use a substring that matches every record so the clamp-on-shrink effect won't snap
    // page 2 back to 1 on filtered results.
    state.records = makeRecords(60);
    const { result, history: routerHistory } = await renderController({
      initialUrl: '/p',
      history,
    });
    await waitFor(() => expect(result.current.records.isLoading).toBe(false));

    act(() => {
      result.current.searchInput.setInput('q');
    });
    act(() => {
      result.current.setPageIndex(2);
    });

    await waitFor(() => {
      const params = new URLSearchParams(routerHistory.location.search);
      expect(params.get('page')).toBe('2');
      expect(params.get('q')).toBe('q');
    });
  });
});