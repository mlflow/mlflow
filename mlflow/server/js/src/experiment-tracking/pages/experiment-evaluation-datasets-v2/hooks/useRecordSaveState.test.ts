// @ts-nocheck — punting test typing; see PR2 plan in branch import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import type { DatasetRecord } from '../hooks/useDatasetsQueries';
import { useUpsertDatasetRecordsMutation } from '../hooks/useDatasetsQueries';
import { useRecordSaveState } from './useRecordSaveState';

// Mock just the mutation hook from the legacy queries module — the rest of the file (types,
// fetch helpers) is left intact. The mock lets each test drive `mutate`'s success/error
// callbacks deterministically without an MSW round-trip; the FSM is the unit under test.
jest.mock('../hooks/useDatasetsQueries', () => ({
  useUpsertDatasetRecordsMutation: jest.fn(),
  // `useRecordSaveState` calls this for the pre-save refetch step. The mock returns a stable
  // query-key array; the actual cache value is irrelevant because `queryClient.getQueryData`
  // returns undefined in tests and the hook falls back to the `existingRecords` prop.
  listDatasetRecordsQueryKey: (datasetId: string) => ['listDatasetRecords', datasetId],
}));

type MutateCall = {
  variables: Array<{
    recordId: string;
    updates: Partial<DatasetRecord>;
    updateMask: Partial<DatasetRecord>;
  }>;
  onSuccess: () => void;
  onError: (err: unknown) => void;
};

interface FakeMutation {
  isLoading: boolean;
  calls: MutateCall[];
  mutate: jest.Mock;
}

const installFakeMutation = (): FakeMutation => {
  const fake: FakeMutation = {
    isLoading: false,
    calls: [],
    mutate: jest.fn(),
  };
  fake.mutate.mockImplementation((variables: unknown, options: unknown) => {
    fake.calls.push({
      variables: variables as MutateCall['variables'],
      onSuccess: (options as { onSuccess: () => void }).onSuccess,
      onError: (options as { onError: (err: unknown) => void }).onError,
    });
  });
  // The hook returns a much wider UseMutationResult; we only stub the two fields the FSM
  // reads. `as unknown as ReturnType<...>` makes the partial-stub intent explicit.
  jest.mocked(useUpsertDatasetRecordsMutation).mockReturnValue({
    mutate: fake.mutate,
    get isLoading() {
      return fake.isLoading;
    },
  } as unknown as ReturnType<typeof useUpsertDatasetRecordsMutation>);
  return fake;
};

const baseRecord: DatasetRecord = {
  dataset_record_id: 'rec-1',
  inputs: { question: 'q1' },
  expectations: { answer: 'a1' },
  tags: {},
  source: { human: { user_name: 'alice@databricks.com' } },
  create_time: '2026-01-01T00:00:00Z',
  last_update_time: '2026-01-01T00:00:00Z',
  created_by: 'alice@databricks.com',
  last_updated_by: 'alice@databricks.com',
};

const renderSaveState = (
  record: DatasetRecord | undefined = baseRecord,
  overrides: Partial<{
    onSaveSuccess: () => void;
    onSaveError: (err: Error) => void;
    fallbackErrorMessage: string;
    existingRecords: DatasetRecord[];
    queryClient: QueryClient;
  }> = {},
) => {
  // Each test gets a fresh QueryClient so cache state never leaks across cases. The
  // refetch-before-validate step inside `save()` will call refetchQueries on this client.
  const queryClient =
    overrides.queryClient ??
    new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  const utils = renderHook(
    ({ rec }: { rec: DatasetRecord | undefined }) =>
      useRecordSaveState({
        datasetId: 'ds-1',
        record: rec,
        fallbackErrorMessage: overrides.fallbackErrorMessage ?? 'Save failed',
        // Default: the record under test plus nothing else — singleturn schema, so the
        // validator is a no-op for every test that doesn't explicitly probe its behavior.
        existingRecords: overrides.existingRecords ?? (rec ? [rec] : []),
        onSaveSuccess: overrides.onSaveSuccess,
        onSaveError: overrides.onSaveError,
      }),
    { initialProps: { rec: record }, wrapper },
  );
  return { ...utils, queryClient };
};

// Helper that mirrors the production save() chain: the user clicks Save, then we wait for
// the refetch+validate microtasks to settle before asserting on the mutation.
const saveAndAwait = async (trigger: () => void, expectation: () => void) => {
  await act(async () => {
    trigger();
  });
  await waitFor(expectation);
};

describe('useRecordSaveState', () => {
  let fake: FakeMutation;

  beforeEach(() => {
    fake = installFakeMutation();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('initial state is clean with no dirty or error', () => {
    const { result } = renderSaveState();
    expect(result.current.status).toBe('clean');
    expect(result.current.isDirty).toBe(false);
    expect(result.current.errorMessage).toBeUndefined();
  });

  test('editing inputs transitions to dirty', () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"question": "edited"}'));
    expect(result.current.isDirty).toBe(true);
    expect(result.current.status).toBe('dirty');
  });

  test('invalid JSON in either editor surfaces as invalid and blocks save', async () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{not json'));
    expect(result.current.status).toBe('invalid');

    await act(async () => {
      result.current.save();
    });
    expect(fake.mutate).not.toHaveBeenCalled();
  });

  test('empty inputs blocks save with empty-inputs status (would otherwise wipe field server-side)', async () => {
    const { result } = renderSaveState();
    // User Ctrl-A-deletes the inputs editor: it's valid JSON ('' → {}), but treating that
    // as a save would round-trip to `inputs: []` with `update_mask=inputs` and clear the
    // field on the server. The hook must block this rather than silently destroy data.
    act(() => result.current.inputs.setText(''));
    expect(result.current.status).toBe('empty-inputs');

    await act(async () => {
      result.current.save();
    });
    expect(fake.mutate).not.toHaveBeenCalled();
  });

  test('save called when not dirty is a no-op', async () => {
    const { result } = renderSaveState();
    await act(async () => {
      result.current.save();
    });
    expect(fake.mutate).not.toHaveBeenCalled();
  });

  test('save sends only the dirty fields in updates/updateMask (inputs only)', async () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );

    const { variables } = fake.calls[0];
    expect(variables).toHaveLength(1);
    expect(variables[0].recordId).toBe('rec-1');
    expect(variables[0].updates).toEqual({ inputs: { question: 'new' } });
    expect(variables[0].updateMask).toEqual({ inputs: { question: 'new' } });
    expect(variables[0].updates).not.toHaveProperty('expectations');
    expect(variables[0].updateMask).not.toHaveProperty('expectations');
  });

  test('save sends only expectations when only expectations are dirty', async () => {
    const { result } = renderSaveState();
    act(() => result.current.expectations.setText('{"answer":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );

    expect(fake.calls[0].variables[0].updates).toEqual({ expectations: { answer: 'new' } });
    expect(fake.calls[0].variables[0].updates).not.toHaveProperty('inputs');
  });

  test('save sends both inputs and expectations when both are dirty', async () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"new"}'));
    act(() => result.current.expectations.setText('{"answer":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );

    expect(fake.calls[0].variables[0].updates).toEqual({
      inputs: { question: 'new' },
      expectations: { answer: 'new' },
    });
  });

  test('status flips to saving while the mutation is in-flight', async () => {
    const { result, rerender } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );

    // Simulate React Query's `isLoading=true` while the request is in-flight; re-render so
    // the hook reads the new mutation state.
    fake.isLoading = true;
    rerender({ rec: baseRecord });
    expect(result.current.status).toBe('saving');
  });

  test('refetch-before-validate uses the fresh cache for schema validation (rejects mid-flight peer-tab mix)', async () => {
    // Setup: cache holds an all-singleturn snapshot at mount, but a peer tab adds a
    // multiturn record between mount and Save. We seed the cache with the peer-tab state
    // post-Save so the refetched validation sees the mixed shape.
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const singleturnRecord: DatasetRecord = {
      ...baseRecord,
      dataset_record_id: 'rec-1',
      inputs: { messages: [] },
    };
    const onSaveError = jest.fn();
    const { result } = renderSaveState(singleturnRecord, {
      existingRecords: [singleturnRecord],
      queryClient,
      onSaveError,
    });

    // Peer tab introduced a multiturn record — the parent prop is stale, but the cache
    // updated. Our save() must observe the cache state, not the stale prop.
    queryClient.setQueryData(
      ['listDatasetRecords', 'ds-1'],
      [singleturnRecord, { ...baseRecord, dataset_record_id: 'rec-2', inputs: { goal: 'g2' } }],
    );

    act(() => result.current.inputs.setText('{"messages":[{"role":"user","content":"hi"}]}'));
    await act(async () => {
      result.current.save();
    });
    await waitFor(() => expect(result.current.status).toBe('error'));

    expect(fake.mutate).not.toHaveBeenCalled();
    expect(onSaveError).toHaveBeenCalledTimes(1);
    expect(result.current.errorMessage).toMatch(/Mixed schemas/);
  });

  test('onSuccess resets editors, transitions to saved, and calls onSaveSuccess', async () => {
    const onSaveSuccess = jest.fn();
    const { result } = renderSaveState(baseRecord, { onSaveSuccess });
    act(() => result.current.inputs.setText('{"question":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );

    act(() => fake.calls[0].onSuccess());
    expect(result.current.isDirty).toBe(false);
    expect(result.current.status).toBe('saved');
    expect(onSaveSuccess).toHaveBeenCalledTimes(1);
  });

  test('onError sets errorMessage from the Error and calls onSaveError', async () => {
    const onSaveError = jest.fn();
    const { result } = renderSaveState(baseRecord, { onSaveError });
    act(() => result.current.inputs.setText('{"question":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );

    const err = new Error('Backend exploded');
    act(() => fake.calls[0].onError(err));

    expect(result.current.status).toBe('error');
    expect(result.current.errorMessage).toBe('Backend exploded');
    expect(onSaveError).toHaveBeenCalledWith(err);
  });

  test('fallback error message is used when rejection is not an Error instance', async () => {
    const onSaveError = jest.fn();
    const { result } = renderSaveState(baseRecord, {
      onSaveError,
      fallbackErrorMessage: 'Something went wrong',
    });
    act(() => result.current.inputs.setText('{"question":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );

    act(() => fake.calls[0].onError('not-an-error'));
    expect(result.current.errorMessage).toBe('Something went wrong');
    expect(onSaveError).toHaveBeenCalledTimes(1);
    expect(onSaveError.mock.calls[0][0]).toBeInstanceOf(Error);
  });

  test('discard resets both editors and clears errorMessage', async () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );
    act(() => fake.calls[0].onError(new Error('boom')));
    expect(result.current.status).toBe('error');

    act(() => result.current.discard());
    expect(result.current.errorMessage).toBeUndefined();
    expect(result.current.isDirty).toBe(false);
    expect(result.current.status).toBe('clean');
  });

  test('switching to a different record clears errorMessage and justSaved', async () => {
    const { result, rerender } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"new"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );
    act(() => fake.calls[0].onError(new Error('boom')));
    expect(result.current.status).toBe('error');

    const otherRecord: DatasetRecord = { ...baseRecord, dataset_record_id: 'rec-2' };
    rerender({ rec: otherRecord });
    expect(result.current.errorMessage).toBeUndefined();
    expect(result.current.status).toBe('clean');
  });

  test('Cmd+S triggers save and preventDefaults the event', async () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"new"}'));

    const preventDefault = jest.fn();
    await act(async () => {
      result.current.onContainerKeyDown({
        metaKey: true,
        ctrlKey: false,
        key: 's',
        preventDefault,
      } as unknown as React.KeyboardEvent);
    });
    expect(preventDefault).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(fake.calls).toHaveLength(1));
  });

  test('Ctrl+S also triggers save (Linux/Windows)', async () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"new"}'));

    const preventDefault = jest.fn();
    await act(async () => {
      result.current.onContainerKeyDown({
        metaKey: false,
        ctrlKey: true,
        key: 's',
        preventDefault,
      } as unknown as React.KeyboardEvent);
    });
    expect(preventDefault).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(fake.calls).toHaveLength(1));
  });

  test('Cmd+S is a no-op when clean (no mutation call)', async () => {
    const { result } = renderSaveState();
    const preventDefault = jest.fn();
    await act(async () => {
      result.current.onContainerKeyDown({
        metaKey: true,
        ctrlKey: false,
        key: 's',
        preventDefault,
      } as unknown as React.KeyboardEvent);
    });
    expect(fake.mutate).not.toHaveBeenCalled();
  });

  test('Cmd+S is a no-op when JSON is invalid', async () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{not json'));

    const preventDefault = jest.fn();
    await act(async () => {
      result.current.onContainerKeyDown({
        metaKey: true,
        ctrlKey: false,
        key: 's',
        preventDefault,
      } as unknown as React.KeyboardEvent);
    });
    expect(fake.mutate).not.toHaveBeenCalled();
  });

  test('non-S keys do not trigger save', () => {
    const { result } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"new"}'));

    const preventDefault = jest.fn();
    act(() =>
      result.current.onContainerKeyDown({
        metaKey: true,
        ctrlKey: false,
        key: 'a',
        preventDefault,
      } as unknown as React.KeyboardEvent),
    );
    expect(preventDefault).not.toHaveBeenCalled();
    expect(fake.mutate).not.toHaveBeenCalled();
  });

  test('save with no record is a no-op (defensive guard)', async () => {
    const { result } = renderSaveState(undefined);
    await act(async () => {
      result.current.save();
    });
    expect(fake.mutate).not.toHaveBeenCalled();
  });

  test('onSuccess advances the baseline to the just-saved text even if the record prop swaps before the callback fires', async () => {
    // Concretizes the stale-closure concern: if the parent swaps `record` to a different
    // baseline before the upsert resolves, the post-save reset must still use what the
    // user actually submitted — not the new record's baseline.
    const { result, rerender } = renderSaveState();
    act(() => result.current.inputs.setText('{"question":"submitted-value"}'));
    await saveAndAwait(
      () => result.current.save(),
      () => expect(fake.calls).toHaveLength(1),
    );

    // Parent rerenders with a different record before the network resolves.
    const otherRecord: DatasetRecord = {
      ...baseRecord,
      dataset_record_id: 'rec-1',
      inputs: { question: 'server-stomped-this-in' },
    };
    rerender({ rec: otherRecord });

    act(() => fake.calls[0].onSuccess());
    expect(result.current.isDirty).toBe(false);
    expect(result.current.inputs.text).toBe('{"question":"submitted-value"}');
  });
});