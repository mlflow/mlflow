import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import type { DatasetRecord } from './useDatasetsQueries';
import { useUpdateDatasetRecordMutation } from './useDatasetsQueries';
import { useRecordSaveState } from './useRecordSaveState';

jest.mock('../hooks/useDatasetsQueries', () => ({
  useUpdateDatasetRecordMutation: jest.fn(),
  listDatasetRecordsQueryKey: (datasetId: string) => ['listDatasetRecords', datasetId],
}));

type UpdateArg = Array<{ recordId: string; updates: Partial<DatasetRecord> }>;

interface FakeMutation {
  isLoading: boolean;
  calls: Array<{ variables: UpdateArg; onSuccess: () => void; onError: (err: unknown) => void }>;
  mutate: jest.Mock;
}

const installFakeMutation = (): FakeMutation => {
  const fake: FakeMutation = { isLoading: false, calls: [], mutate: jest.fn() };
  fake.mutate.mockImplementation((variables: unknown, options: unknown) => {
    fake.calls.push({
      variables: variables as UpdateArg,
      onSuccess: (options as { onSuccess: () => void }).onSuccess,
      onError: (options as { onError: (err: unknown) => void }).onError,
    });
  });
  jest.mocked(useUpdateDatasetRecordMutation).mockReturnValue({
    mutate: fake.mutate,
    get isLoading() {
      return fake.isLoading;
    },
  } as unknown as ReturnType<typeof useUpdateDatasetRecordMutation>);
  return fake;
};

const makeRecord = (overrides: Partial<DatasetRecord> = {}): DatasetRecord =>
  ({
    dataset_record_id: 'rec-1',
    inputs: { q: 'original' },
    expectations: { a: 'original' },
    ...overrides,
  }) as DatasetRecord;

const renderSaveState = (
  props: Partial<{
    record: DatasetRecord;
    existingRecords: DatasetRecord[];
    onSaveError: (e: Error) => void;
  }> = {},
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
  const initialProps = {
    record: props.record ?? makeRecord(),
    existingRecords: props.existingRecords ?? [],
    onSaveError: props.onSaveError,
  };
  return renderHook(
    ({ record, existingRecords, onSaveError }) =>
      useRecordSaveState({
        datasetId: 'ds-1',
        record,
        fallbackErrorMessage: 'Generic save failure',
        existingRecords,
        onSaveError,
        // Tiny debounce so autosave fires within the test's waitFor window.
        debounceMs: 10,
      }),
    { wrapper, initialProps },
  );
};

describe('useRecordSaveState (autosave by id)', () => {
  beforeEach(() => {
    installFakeMutation();
  });
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('initial state is clean and not dirty', () => {
    const { result } = renderSaveState();
    expect(result.current.status).toBe('clean');
    expect(result.current.isDirty).toBe(false);
  });

  test('editing inputs autosaves the dirty field by id after the debounce', async () => {
    const fake = installFakeMutation();
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"q":"edited"}'));
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    expect(fake.calls[0].variables).toEqual([{ recordId: 'rec-1', updates: { inputs: { q: 'edited' } } }]);
  });

  test('editing only expectations autosaves only that field', async () => {
    const fake = installFakeMutation();
    const { result } = renderSaveState();
    act(() => result.current.expectations.setText('{"a":"edited"}'));
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    expect(fake.calls[0].variables).toEqual([{ recordId: 'rec-1', updates: { expectations: { a: 'edited' } } }]);
  });

  test('defers autosave while JSON is invalid (status=invalid, no write)', async () => {
    const fake = installFakeMutation();
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{not valid'));
    expect(result.current.status).toBe('invalid');
    await new Promise((r) => setTimeout(r, 40));
    expect(fake.calls).toHaveLength(0);
  });

  test('defers autosave when inputs are emptied to {} (status=empty-inputs, no write)', async () => {
    const fake = installFakeMutation();
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{}'));
    expect(result.current.status).toBe('empty-inputs');
    await new Promise((r) => setTimeout(r, 40));
    expect(fake.calls).toHaveLength(0);
  });

  test('a dirty valid edit reads as "saving" (there is no manual save step)', () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"q":"edited"}'));
    expect(result.current.status).toBe('saving');
  });

  test('onSuccess clears dirty and resolves to saved', async () => {
    const fake = installFakeMutation();
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"q":"edited"}'));
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    act(() => fake.calls[0].onSuccess());
    await waitFor(() => expect(result.current.status).toBe('saved'));
    expect(result.current.isDirty).toBe(false);
  });

  test('onError surfaces errorMessage and status=error', async () => {
    const fake = installFakeMutation();
    const onSaveError = jest.fn();
    const { result } = renderSaveState({ onSaveError });
    act(() => result.current.inputs.setText('{"q":"edited"}'));
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    act(() => fake.calls[0].onError(new Error('Backend rejected')));
    await waitFor(() => expect(result.current.errorMessage).toBe('Backend rejected'));
    expect(result.current.status).toBe('error');
    expect(onSaveError).toHaveBeenCalled();
  });

  test('does not re-fire a failed save until the value changes (no error-toast spam)', async () => {
    const fake = installFakeMutation();
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"q":"dupe"}'));
    await waitFor(() => expect(fake.calls).toHaveLength(1));

    act(() => fake.calls[0].onError(new Error('A record with identical inputs already exists')));
    await waitFor(() => expect(result.current.status).toBe('error'));

    // Same value stays put — the debounce must NOT keep retrying the doomed payload.
    await new Promise((r) => setTimeout(r, 60));
    expect(fake.calls).toHaveLength(1);

    // Editing to a different value clears the block and retries.
    act(() => result.current.inputs.setText('{"q":"unique"}'));
    await waitFor(() => expect(fake.calls).toHaveLength(2));
  });

  test('switching to a different record flushes the outgoing record’s pending edit', async () => {
    const fake = installFakeMutation();
    const { result, rerender } = renderSaveState({ record: makeRecord({ dataset_record_id: 'rec-A' }) });
    act(() => result.current.inputs.setText('{"q":"edited-A"}'));
    // Switch before the debounce fires — the cleanup flush must commit rec-A's edit, not drop it.
    rerender({ record: makeRecord({ dataset_record_id: 'rec-B' }), existingRecords: [], onSaveError: undefined });
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    expect(fake.calls[0].variables).toEqual([{ recordId: 'rec-A', updates: { inputs: { q: 'edited-A' } } }]);
  });

  test('no autosave fires when nothing is dirty', async () => {
    const fake = installFakeMutation();
    renderSaveState();
    await new Promise((r) => setTimeout(r, 40));
    expect(fake.calls).toHaveLength(0);
  });

  test('Cmd+S commits immediately and preventDefaults', async () => {
    const fake = installFakeMutation();
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"q":"edited"}'));
    const preventDefault = jest.fn();
    act(() =>
      result.current.onContainerKeyDown({
        metaKey: true,
        ctrlKey: false,
        key: 's',
        preventDefault,
      } as unknown as React.KeyboardEvent),
    );
    expect(preventDefault).toHaveBeenCalled();
    await waitFor(() => expect(fake.calls).toHaveLength(1));
  });

  test('discard reverts editors to the record values and cancels pending autosave', async () => {
    const fake = installFakeMutation();
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"q":"edited"}'));
    act(() => result.current.discard());
    expect(result.current.inputs.text).toContain('original');
    await new Promise((r) => setTimeout(r, 40));
    expect(fake.calls).toHaveLength(0);
  });
});
