import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { DatasetRecord } from './useDatasetsQueries';
import {
  listDatasetRecordsQueryKey,
  useCreateDatasetRecordMutation,
  useUpdateDatasetRecordMutation,
} from './useDatasetsQueries';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  getAjaxUrl: (url: string) => `/${url}`,
  fetchAPI: jest.fn(),
}));

const mockFetchAPI = jest.mocked(fetchAPI);

const DATASET_ID = 'ds-1';
const RECORDS_KEY = listDatasetRecordsQueryKey(DATASET_ID);

const makeClient = () =>
  new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });

const wrapperFor = (queryClient: QueryClient) =>
  function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client: queryClient }, children);
  };

const seedRecords = (queryClient: QueryClient, records: DatasetRecord[]) =>
  queryClient.setQueryData(RECORDS_KEY, records);

const cachedRecords = (queryClient: QueryClient) => queryClient.getQueryData<DatasetRecord[]>(RECORDS_KEY) ?? [];

afterEach(() => {
  jest.clearAllMocks();
});

describe('useCreateDatasetRecordMutation', () => {
  test('resolves the new id from the response record_ids and inserts it into the list cache', async () => {
    const queryClient = makeClient();
    seedRecords(queryClient, []);
    mockFetchAPI.mockResolvedValueOnce({ inserted_count: 1, updated_count: 0, record_ids: ['dr-new'] });

    const { result } = renderHook(() => useCreateDatasetRecordMutation(DATASET_ID), {
      wrapper: wrapperFor(queryClient),
    });

    let created: DatasetRecord | undefined;
    await act(async () => {
      created = await result.current.mutateAsync({ inputs: { q: 'hi' } });
    });

    // The id used by the single-step add flow to select the new record comes from record_ids.
    expect(created?.dataset_record_id).toBe('dr-new');
    // Optimistic insert: the record is in the cache before any refetch, so selecting it by id
    // resolves immediately (avoids the "record not found -> close panel" race).
    expect(cachedRecords(queryClient).map((r) => r.dataset_record_id)).toContain('dr-new');
  });

  test('falls back gracefully when the response omits record_ids', async () => {
    const queryClient = makeClient();
    seedRecords(queryClient, []);
    mockFetchAPI.mockResolvedValueOnce({ inserted_count: 1, updated_count: 0 });

    const { result } = renderHook(() => useCreateDatasetRecordMutation(DATASET_ID), {
      wrapper: wrapperFor(queryClient),
    });

    let created: DatasetRecord | undefined;
    await act(async () => {
      created = await result.current.mutateAsync({ inputs: { q: 'hi' } });
    });
    // No id available -> empty string (the page guards on this before selecting).
    expect(created?.dataset_record_id).toBe('');
  });
});

describe('useUpdateDatasetRecordMutation', () => {
  test('PATCHes the records endpoint and optimistically patches the cached record', async () => {
    const queryClient = makeClient();
    seedRecords(queryClient, [{ dataset_record_id: 'dr-1', inputs: { q: 'old' } } as unknown as DatasetRecord]);
    mockFetchAPI.mockResolvedValueOnce({ updated_count: 1 });

    const { result } = renderHook(() => useUpdateDatasetRecordMutation(DATASET_ID), {
      wrapper: wrapperFor(queryClient),
    });

    await act(async () => {
      await result.current.mutateAsync([{ recordId: 'dr-1', updates: { inputs: { q: 'new' } } }]);
    });

    expect(mockFetchAPI).toHaveBeenCalledWith(
      '/ajax-api/3.0/mlflow/datasets/ds-1/records',
      expect.objectContaining({ method: 'PATCH' }),
    );
    expect(cachedRecords(queryClient)[0].inputs).toEqual({ q: 'new' });
  });

  test('rolls the optimistic patch back to the original on error', async () => {
    const queryClient = makeClient();
    seedRecords(queryClient, [
      { dataset_record_id: 'dr-1', inputs: { q: 'old' }, expectations: {} } as unknown as DatasetRecord,
    ]);
    mockFetchAPI.mockRejectedValueOnce(new Error('boom'));

    const { result } = renderHook(() => useUpdateDatasetRecordMutation(DATASET_ID), {
      wrapper: wrapperFor(queryClient),
    });

    await act(async () => {
      await expect(
        result.current.mutateAsync([{ recordId: 'dr-1', updates: { inputs: { q: 'new' } } }]),
      ).rejects.toThrow('boom');
    });

    // onError restores the pre-mutation snapshot — the failed edit must not linger in the cache.
    await waitFor(() => expect(cachedRecords(queryClient)[0].inputs).toEqual({ q: 'old' }));
  });
});
