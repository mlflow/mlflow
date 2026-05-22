/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck — punting test typing; see PR2 plan in branch import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import type { ReactNode } from 'react';
import type { Dataset } from './useDatasetsQueries';
import { useDeleteDatasetMutation } from './useDatasetsQueries';
import type { DatasetNotifyApi } from './useDatasetNotifications';
import { useDatasetDelete } from './useDatasetDelete';
import { jest } from '@jest/globals';
import { describe } from '@jest/globals';
import { beforeEach } from '@jest/globals';
import { afterEach } from '@jest/globals';
import { test } from '@jest/globals';
import { expect } from '@jest/globals';

jest.mock('../hooks/useDatasetsQueries', () => ({
  useDeleteDatasetMutation: jest.fn(),
}));

interface FakeMutation {
  mutateAsync: jest.Mock;
  error: Error | null;
  isLoading: boolean;
  /** Resolves the most recent mutateAsync invocation. */
  resolve: (value?: unknown) => void;
  /** Rejects the most recent mutateAsync invocation. */
  reject: (err: unknown) => void;
}

const installFakeMutation = (): FakeMutation => {
  const fake: FakeMutation = {
    mutateAsync: jest.fn(),
    error: null,
    isLoading: false,
    resolve: () => {},
    reject: () => {},
  };
  fake.mutateAsync.mockImplementation(
    () =>
      new Promise((resolve, reject) => {
        fake.resolve = resolve;
        fake.reject = reject;
      }),
  );
  // The hook returns a much wider UseMutationResult; only the three fields the controller
  // reads are stubbed here. `as unknown as ReturnType<...>` makes the partial-stub intent
  // explicit instead of leaking `as any`.
  jest.mocked(useDeleteDatasetMutation).mockReturnValue({
    mutateAsync: fake.mutateAsync,
    get error() {
      return fake.error;
    },
    get isLoading() {
      return fake.isLoading;
    },
  } as unknown as ReturnType<typeof useDeleteDatasetMutation>);
  return fake;
};

const datasetFixture: Dataset = {
  dataset_id: 'ds-1',
  name: 'Customer Eval',
  create_time: '2026-01-01T00:00:00Z',
};

const makeNotify = (): DatasetNotifyApi => ({
  success: jest.fn(),
  error: jest.fn(),
});

interface RenderOpts {
  notify?: DatasetNotifyApi;
  onMutated?: jest.Mock<(dataset: Dataset) => void>;
  onAfterDelete?: jest.Mock<(dataset: Dataset) => void>;
  pollForPropagation?: jest.Mock<(target: Dataset, signal: AbortSignal) => Promise<boolean>>;
}

const renderDelete = ({ notify, onMutated, onAfterDelete, pollForPropagation }: RenderOpts = {}) => {
  const usedNotify = notify ?? makeNotify();
  const wrapper = ({ children }: { children: ReactNode }) => <IntlProvider locale="en">{children}</IntlProvider>;
  return {
    notify: usedNotify,
    ...renderHook(
      () =>
        useDatasetDelete({
          experimentId: 'exp-1',
          notify: usedNotify,
          onMutated,
          onAfterDelete,
          pollForPropagation,
        }),
      { wrapper },
    ),
  };
};

describe('useDatasetDelete', () => {
  let fake: FakeMutation;

  beforeEach(() => {
    fake = installFakeMutation();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('requestDelete sets pendingDataset', () => {
    const { result } = renderDelete();
    act(() => result.current.requestDelete(datasetFixture));
    expect(result.current.pendingDataset).toEqual(datasetFixture);
  });

  test('cancelDelete clears pendingDataset', () => {
    const { result } = renderDelete();
    act(() => result.current.requestDelete(datasetFixture));
    act(() => result.current.cancelDelete());
    expect(result.current.pendingDataset).toBeNull();
  });

  test('confirmDelete with no pending is a no-op (no mutation, no toast)', async () => {
    const { result, notify } = renderDelete();
    await act(async () => {
      await result.current.confirmDelete();
    });
    expect(fake.mutateAsync).not.toHaveBeenCalled();
    expect(notify.success).not.toHaveBeenCalled();
    expect(notify.error).not.toHaveBeenCalled();
  });

  test('happy path without polling: mutateAsync → onMutated → success toast → onAfterDelete', async () => {
    const onMutated = jest.fn();
    const onAfterDelete = jest.fn();
    const { result, notify } = renderDelete({ onMutated, onAfterDelete });

    act(() => result.current.requestDelete(datasetFixture));
    let confirmPromise!: Promise<void>;
    act(() => {
      confirmPromise = result.current.confirmDelete();
    });
    // Mutation in-flight; resolve it.
    expect(fake.mutateAsync).toHaveBeenCalledWith('ds-1');
    await act(async () => {
      fake.resolve();
      await confirmPromise;
    });

    expect(onMutated).toHaveBeenCalledWith(datasetFixture);
    expect(notify.success).toHaveBeenCalledTimes(1);
    expect(onAfterDelete).toHaveBeenCalledWith(datasetFixture);
    expect(result.current.pendingDataset).toBeNull();
    expect(result.current.isPolling).toBe(false);
  });

  test('happy path with pollForPropagation observing propagation: bare success toast', async () => {
    let resolvePoll!: (propagated: boolean) => void;
    const pollForPropagation = jest.fn<(target: Dataset, signal: AbortSignal) => Promise<boolean>>(
      () =>
        new Promise<boolean>((resolve) => {
          resolvePoll = resolve;
        }),
    );
    const onAfterDelete = jest.fn();
    const { result, notify } = renderDelete({ pollForPropagation, onAfterDelete });

    act(() => result.current.requestDelete(datasetFixture));
    let confirmPromise!: Promise<void>;
    act(() => {
      confirmPromise = result.current.confirmDelete();
    });
    await act(async () => {
      fake.resolve();
      // Yield once so confirmDelete picks up after mutateAsync resolves.
      await Promise.resolve();
    });
    await waitFor(() => expect(result.current.isPolling).toBe(true));
    expect(pollForPropagation).toHaveBeenCalledWith(datasetFixture, expect.any(AbortSignal));
    // Modal must stay open through propagation — clearing it on mutation resolve produces
    // a visible flicker where the just-deleted row briefly reappears.
    expect(result.current.pendingDataset).toEqual(datasetFixture);

    await act(async () => {
      resolvePoll(true);
      await confirmPromise;
    });
    expect(result.current.isPolling).toBe(false);
    expect(notify.success).toHaveBeenCalledTimes(1);
    expect(notify.success).toHaveBeenCalledWith('Deleted dataset "Customer Eval"');
    expect(onAfterDelete).toHaveBeenCalledTimes(1);
    expect(result.current.pendingDataset).toBeNull();
  });

  test('pollForPropagation times out (resolves false): success toast carries the refresh suffix', async () => {
    let resolvePoll!: (propagated: boolean) => void;
    const pollForPropagation = jest.fn<(target: Dataset, signal: AbortSignal) => Promise<boolean>>(
      () =>
        new Promise<boolean>((resolve) => {
          resolvePoll = resolve;
        }),
    );
    const { result, notify } = renderDelete({ pollForPropagation });

    act(() => result.current.requestDelete(datasetFixture));
    let confirmPromise!: Promise<void>;
    act(() => {
      confirmPromise = result.current.confirmDelete();
    });
    await act(async () => {
      fake.resolve();
      await Promise.resolve();
    });
    await waitFor(() => expect(result.current.isPolling).toBe(true));

    // Resolve false to simulate the poll exhausting its attempt cap without observing the
    // delete reflected in the list. The mutation succeeded — we just couldn't confirm
    // propagation — so the user still gets a success toast, but with a "may take a moment
    // to refresh" suffix so the brief re-appearance of the row doesn't look like a bug.
    await act(async () => {
      resolvePoll(false);
      await confirmPromise;
    });

    expect(notify.success).toHaveBeenCalledTimes(1);
    expect(notify.success).toHaveBeenCalledWith('Deleted dataset "Customer Eval" (may take a moment to refresh)');
    expect(notify.error).not.toHaveBeenCalled();
    expect(result.current.pendingDataset).toBeNull();
  });

  test('no pollForPropagation provided: defaults to bare success toast', async () => {
    const { result, notify } = renderDelete();

    act(() => result.current.requestDelete(datasetFixture));
    let confirmPromise!: Promise<void>;
    act(() => {
      confirmPromise = result.current.confirmDelete();
    });
    await act(async () => {
      fake.resolve();
      await confirmPromise;
    });

    expect(notify.success).toHaveBeenCalledWith('Deleted dataset "Customer Eval"');
  });

  test('poll rejection: modal stays open, error toast fires, isPolling clears', async () => {
    let rejectPoll!: (err: unknown) => void;
    const pollForPropagation = jest.fn<(target: Dataset, signal: AbortSignal) => Promise<boolean>>(
      () =>
        new Promise<boolean>((_, reject) => {
          rejectPoll = reject;
        }),
    );
    const onAfterDelete = jest.fn();
    const { result, notify } = renderDelete({ pollForPropagation, onAfterDelete });

    act(() => result.current.requestDelete(datasetFixture));
    let confirmPromise!: Promise<void>;
    act(() => {
      confirmPromise = result.current.confirmDelete();
    });
    await act(async () => {
      fake.resolve();
      await Promise.resolve();
    });
    await waitFor(() => expect(result.current.isPolling).toBe(true));

    await act(async () => {
      rejectPoll(new Error('poll failed'));
      await confirmPromise;
    });

    expect(result.current.isPolling).toBe(false);
    // Modal must remain open so the user can retry or cancel.
    expect(result.current.pendingDataset).toEqual(datasetFixture);
    expect(notify.error).toHaveBeenCalledTimes(1);
    expect(notify.success).not.toHaveBeenCalled();
    expect(onAfterDelete).not.toHaveBeenCalled();
  });

  test('mutation rejection: notify.error, no success toast, no onAfterDelete', async () => {
    const onAfterDelete = jest.fn();
    const { result, notify } = renderDelete({ onAfterDelete });

    act(() => result.current.requestDelete(datasetFixture));
    let confirmPromise!: Promise<void>;
    act(() => {
      confirmPromise = result.current.confirmDelete();
    });
    await act(async () => {
      fake.reject(new Error('boom'));
      await confirmPromise;
    });

    expect(notify.error).toHaveBeenCalledTimes(1);
    expect(notify.success).not.toHaveBeenCalled();
    expect(onAfterDelete).not.toHaveBeenCalled();
    expect(result.current.isPolling).toBe(false);
    // Modal must remain open after a mutation failure so the user can retry or cancel.
    expect(result.current.pendingDataset).toEqual(datasetFixture);
  });

  test('aborted during poll (via a second requestDelete): no success, no onAfterDelete', async () => {
    let resolvePoll!: (propagated: boolean) => void;
    const pollForPropagation = jest.fn<(target: Dataset, signal: AbortSignal) => Promise<boolean>>(
      () =>
        new Promise<boolean>((resolve) => {
          resolvePoll = resolve;
        }),
    );
    const onAfterDelete = jest.fn();
    const { result, notify } = renderDelete({ pollForPropagation, onAfterDelete });

    act(() => result.current.requestDelete(datasetFixture));
    let firstConfirm!: Promise<void>;
    act(() => {
      firstConfirm = result.current.confirmDelete();
    });
    await act(async () => {
      fake.resolve();
      await Promise.resolve();
    });
    await waitFor(() => expect(result.current.isPolling).toBe(true));

    // Trigger a second confirmDelete with a fresh request — this aborts the first poll.
    const secondDataset: Dataset = { ...datasetFixture, dataset_id: 'ds-2', name: 'Other' };
    act(() => result.current.requestDelete(secondDataset));
    act(() => {
      // Fire-and-forget; we don't await it.
      result.current.confirmDelete();
    });

    // Unblock the first poll AFTER the abort fires; the original branch should bail out.
    await act(async () => {
      resolvePoll(true);
      await firstConfirm;
    });

    // First confirm produced neither a toast nor the after-delete hook because its controller
    // was aborted before the poll resolved.
    expect(notify.success).not.toHaveBeenCalledWith(expect.stringContaining('Customer Eval'));
    expect(onAfterDelete).not.toHaveBeenCalledWith(datasetFixture);
  });

  test('unmount during poll prevents success toast firing on the dead component', async () => {
    let resolvePoll!: (propagated: boolean) => void;
    const pollForPropagation = jest.fn<(target: Dataset, signal: AbortSignal) => Promise<boolean>>(
      () =>
        new Promise<boolean>((resolve) => {
          resolvePoll = resolve;
        }),
    );
    const onAfterDelete = jest.fn();
    const { result, notify, unmount } = renderDelete({ pollForPropagation, onAfterDelete });

    act(() => result.current.requestDelete(datasetFixture));
    let confirmPromise!: Promise<void>;
    act(() => {
      confirmPromise = result.current.confirmDelete();
    });
    await act(async () => {
      fake.resolve();
      await Promise.resolve();
    });
    await waitFor(() => expect(result.current.isPolling).toBe(true));

    unmount();

    await act(async () => {
      resolvePoll(true);
      await confirmPromise;
    });

    expect(notify.success).not.toHaveBeenCalled();
    expect(onAfterDelete).not.toHaveBeenCalled();
  });

  test('error field reflects the underlying mutation error', () => {
    fake.error = new Error('network down');
    const { result } = renderDelete();
    expect(result.current.error).toEqual(new Error('network down'));
  });

  test('error field is null when mutation.error is non-Error (defensive)', () => {
    fake.error = 'not-an-error' as unknown as Error;
    const { result } = renderDelete();
    expect(result.current.error).toBeNull();
  });
});
