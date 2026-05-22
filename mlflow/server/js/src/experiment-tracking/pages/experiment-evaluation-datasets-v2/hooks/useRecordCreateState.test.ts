// @ts-nocheck — punting test typing; see PR2 plan in branch import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import type { DatasetRecord } from '../hooks/useDatasetsQueries';
import { useCreateDatasetRecordMutation } from '../hooks/useDatasetsQueries';
import { useRecordCreateState, type PendingNewRecord } from './useRecordCreateState';

jest.mock('../hooks/useDatasetsQueries', () => ({
  useCreateDatasetRecordMutation: jest.fn(),
  // Returns a stable key array — the hook only uses this to invalidate / re-read the cache.
  listDatasetRecordsQueryKey: (datasetId: string) => ['listDatasetRecords', datasetId],
}));

type MutateCall = {
  variables: Partial<DatasetRecord>;
  onSuccess: () => void;
  onError: (err: unknown) => void;
};

interface FakeMutation {
  isLoading: boolean;
  calls: MutateCall[];
  mutate: jest.Mock;
}

const installFakeMutation = (): FakeMutation => {
  const fake: FakeMutation = { isLoading: false, calls: [], mutate: jest.fn() };
  fake.mutate.mockImplementation((variables: unknown, options: unknown) => {
    fake.calls.push({
      variables: variables as Partial<DatasetRecord>,
      onSuccess: (options as { onSuccess: () => void }).onSuccess,
      onError: (options as { onError: (err: unknown) => void }).onError,
    });
  });
  jest.mocked(useCreateDatasetRecordMutation).mockReturnValue({
    mutate: fake.mutate,
    get isLoading() {
      return fake.isLoading;
    },
  } as unknown as ReturnType<typeof useCreateDatasetRecordMutation>);
  return fake;
};

type OnPendingChangePayload = PendingNewRecord;

const renderCreateState = (
  overrides: Partial<{
    existingRecords: DatasetRecord[];
    onSaveSuccess: () => void;
    onSaveError: (e: Error) => void;
    onPendingChange: (next: OnPendingChangePayload) => void;
  }> = {},
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
  return renderHook(
    () =>
      useRecordCreateState({
        datasetId: 'ds-1',
        fallbackErrorMessage: 'Generic create failure',
        existingRecords: overrides.existingRecords ?? [],
        onSaveSuccess: overrides.onSaveSuccess,
        onSaveError: overrides.onSaveError,
        onPendingChange: overrides.onPendingChange,
      }),
    { wrapper },
  );
};

describe('useRecordCreateState', () => {
  beforeEach(() => {
    installFakeMutation();
  });
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('seeds editors with singleturn sample JSON for an empty dataset', () => {
    const { result } = renderCreateState();
    // Singleturn defaults — match against the keys (not the full JSON) so a future tweak to
    // the sample copy doesn't churn this test.
    expect(result.current.inputs.text).toContain('"messages"');
    expect(result.current.expectations.text).toContain('"guidelines"');
    // baseline == seeded, so the form is not "dirty" in the diff sense …
    expect(result.current.isDirty).toBe(false);
    // … but it has content, which the FSM uses to mark the seed as savable from mount.
    expect(result.current.hasContent).toBe(true);
    expect(result.current.status).toBe('dirty');
  });

  test('seeds editors with multiturn sample JSON when existing records use a goal-shaped schema', () => {
    const existingRecords = [
      {
        dataset_record_id: 'rec-multi',
        inputs: { goal: 'Help the user' },
      } as unknown as DatasetRecord,
    ];
    const { result } = renderCreateState({ existingRecords });
    expect(result.current.inputs.text).toContain('"goal"');
    expect(result.current.inputs.text).toContain('"persona"');
    expect(result.current.expectations.text).toContain('"guidelines"');
  });

  test('save submits the unedited seeded payload (user can click Add record without typing)', async () => {
    const fake = installFakeMutation();
    const { result } = renderCreateState();
    // No user edits — call save() directly, simulating an immediate "Add record" click.
    act(() => result.current.save());
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    // Singleturn default payload.
    expect(fake.calls[0].variables.inputs).toEqual({ messages: [{ role: 'user', content: 'Hello' }] });
    expect(fake.calls[0].variables.expectations).toEqual({ guidelines: ['The response must be professional'] });
  });

  test('typing into inputs flips isDirty and hasContent', () => {
    const { result } = renderCreateState();
    act(() => result.current.inputs.setText('{"q":"hi"}'));
    expect(result.current.isDirty).toBe(true);
    expect(result.current.hasContent).toBe(true);
    expect(result.current.status).toBe('dirty');
  });

  test('onPendingChange fires with parsed inputs/expectations as the user types', async () => {
    const onPendingChange = jest.fn();
    const { result } = renderCreateState({ onPendingChange });

    // First effect fires on mount with the seeded defaults; clear so the assertion below
    // only sees the post-setText payload.
    await waitFor(() => expect(onPendingChange).toHaveBeenCalled());
    onPendingChange.mockClear();

    act(() => result.current.inputs.setText('{"q":"hi"}'));
    await waitFor(() => expect(onPendingChange).toHaveBeenCalled());
    const lastCall = onPendingChange.mock.calls[onPendingChange.mock.calls.length - 1]?.[0] as OnPendingChangePayload;
    expect(lastCall.inputs).toEqual({ q: 'hi' });
    expect(lastCall.inputsText).toBe('{"q":"hi"}');
  });

  test('onPendingChange emits raw text alongside undefined parsed value for an invalid field', async () => {
    const onPendingChange = jest.fn();
    const { result } = renderCreateState({ onPendingChange });
    onPendingChange.mockClear();
    act(() => result.current.inputs.setText('{not valid'));
    await waitFor(() => expect(onPendingChange).toHaveBeenCalled());
    const lastCall = onPendingChange.mock.calls[onPendingChange.mock.calls.length - 1]?.[0] as OnPendingChangePayload;
    expect(lastCall.inputs).toBeUndefined();
    // Raw text is forwarded so the phantom row can echo partial typing live.
    expect(lastCall.inputsText).toBe('{not valid');
  });

  test('save passes parsed inputs and expectations to the create mutation', async () => {
    const fake = installFakeMutation();
    const { result } = renderCreateState();
    act(() => result.current.inputs.setText('{"q":"hi"}'));
    act(() => result.current.expectations.setText('{"a":"world"}'));

    act(() => result.current.save());
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    expect(fake.calls[0].variables).toEqual({ inputs: { q: 'hi' }, expectations: { a: 'world' } });
  });

  test('adding a tag flips isDirty and hasContent', () => {
    const { result } = renderCreateState();
    act(() => result.current.setTags({ env: 'prod' }));
    expect(result.current.isDirty).toBe(true);
    expect(result.current.hasContent).toBe(true);
    expect(result.current.tags).toEqual({ env: 'prod' });
  });

  test('save includes tags in the create-mutation payload when non-empty', async () => {
    const fake = installFakeMutation();
    const { result } = renderCreateState();
    act(() => result.current.inputs.setText('{"q":"hi"}'));
    act(() => result.current.expectations.setText('{"a":"world"}'));
    act(() => result.current.setTags({ env: 'prod', region: 'us-west-2' }));

    act(() => result.current.save());
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    expect(fake.calls[0].variables).toEqual({
      inputs: { q: 'hi' },
      expectations: { a: 'world' },
      tags: { env: 'prod', region: 'us-west-2' },
    });
  });

  test('save omits the tags field when the tag map is empty', async () => {
    const fake = installFakeMutation();
    const { result } = renderCreateState();
    act(() => result.current.inputs.setText('{"q":"hi"}'));

    act(() => result.current.save());
    await waitFor(() => expect(fake.calls).toHaveLength(1));
    expect(fake.calls[0].variables).not.toHaveProperty('tags');
  });

  test('onPendingChange emits the current tag map alongside inputs/expectations', async () => {
    const onPendingChange = jest.fn();
    const { result } = renderCreateState({ onPendingChange });
    onPendingChange.mockClear();
    act(() => result.current.setTags({ env: 'prod' }));
    await waitFor(() => expect(onPendingChange).toHaveBeenCalled());
    const lastCall = onPendingChange.mock.calls[onPendingChange.mock.calls.length - 1]?.[0] as OnPendingChangePayload;
    expect(lastCall.tags).toEqual({ env: 'prod' });
  });

  test('save onSuccess clears the draft tag map so post-save status lands on saved', async () => {
    const fake = installFakeMutation();
    const { result } = renderCreateState();
    act(() => result.current.inputs.setText('{"q":"hi"}'));
    act(() => result.current.setTags({ env: 'prod' }));
    act(() => result.current.save());
    await waitFor(() => expect(fake.calls).toHaveLength(1));

    act(() => fake.calls[0].onSuccess());
    await waitFor(() => expect(result.current.status).toBe('saved'));
    expect(result.current.tags).toEqual({});
    expect(result.current.isDirty).toBe(false);
  });

  test('save is a no-op when inputs are an empty object (status surfaces empty-inputs)', () => {
    const fake = installFakeMutation();
    const { result } = renderCreateState();
    // Type a non-empty string that parses to {} (i.e., an empty object literal).
    act(() => result.current.inputs.setText('{}'));
    expect(result.current.status).toBe('empty-inputs');
    act(() => result.current.save());
    expect(fake.calls).toHaveLength(0);
  });

  test('discard reverts editors to the seeded baseline and clears draft tags', () => {
    const { result } = renderCreateState();
    const seededInputs = result.current.inputs.text;
    const seededExpectations = result.current.expectations.text;
    act(() => result.current.inputs.setText('{"q":"hi"}'));
    act(() => result.current.expectations.setText('{"a":"hi"}'));
    act(() => result.current.setTags({ env: 'prod' }));
    act(() => result.current.discard());
    // Discard reverts to baseline (the seeded defaults), not to empty strings.
    expect(result.current.inputs.text).toBe(seededInputs);
    expect(result.current.expectations.text).toBe(seededExpectations);
    expect(result.current.tags).toEqual({});
    expect(result.current.isDirty).toBe(false);
    // The seeded content remains, so the form stays savable post-discard.
    expect(result.current.hasContent).toBe(true);
  });

  test('onSuccess transitions to saved and fires onSaveSuccess', async () => {
    const fake = installFakeMutation();
    const onSaveSuccess = jest.fn();
    const { result } = renderCreateState({ onSaveSuccess });
    act(() => result.current.inputs.setText('{"q":"hi"}'));
    act(() => result.current.save());
    await waitFor(() => expect(fake.calls).toHaveLength(1));

    act(() => fake.calls[0].onSuccess());
    await waitFor(() => expect(onSaveSuccess).toHaveBeenCalled());
    expect(result.current.status).toBe('saved');
  });

  test('onError surfaces errorMessage and calls onSaveError', async () => {
    const fake = installFakeMutation();
    const onSaveError = jest.fn();
    const { result } = renderCreateState({ onSaveError });
    act(() => result.current.inputs.setText('{"q":"hi"}'));
    act(() => result.current.save());
    await waitFor(() => expect(fake.calls).toHaveLength(1));

    act(() => fake.calls[0].onError(new Error('Backend rejected')));
    await waitFor(() => expect(onSaveError).toHaveBeenCalled());
    expect(result.current.errorMessage).toBe('Backend rejected');
    expect(result.current.status).toBe('error');
  });

  test('Cmd+S triggers save and preventDefaults the event', async () => {
    const fake = installFakeMutation();
    const { result } = renderCreateState();
    act(() => result.current.inputs.setText('{"q":"hi"}'));
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

  test('save status reflects invalid JSON before mutation fires', () => {
    const fake = installFakeMutation();
    const { result } = renderCreateState();
    act(() => result.current.inputs.setText('{not valid'));
    expect(result.current.status).toBe('invalid');
    act(() => result.current.save());
    expect(fake.calls).toHaveLength(0);
  });
});