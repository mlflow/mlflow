import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import React from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { invalidateMlflowSearchTracesCache } from '@databricks/web-shared/genai-traces-table';

import { describeTraceDeletionOutcome, useDeleteTracesMutation } from './useDeleteTraces';
import { MlflowService } from '../../../sdk/MlflowService';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  ),
  invalidateMlflowSearchTracesCache: jest.fn(),
}));

const rejection = (message: string): PromiseRejectedResult => ({
  status: 'rejected',
  reason: new ErrorWrapper(JSON.stringify({ error_code: 'PERMISSION_DENIED', message }), 403),
});

describe('describeTraceDeletionOutcome', () => {
  // Traces delete 100 per request, so a larger selection is several requests and the
  // deletion is not atomic. A condition can refuse one chunk and permit another, and the
  // permitted chunk really deletes. `Promise.all` reported the whole deletion as failed,
  // leaving the user with an error beside a list that had genuinely shrunk.

  it('reports how many were deleted alongside the failure', () => {
    const message = describeTraceDeletionOutcome([rejection('Permission denied.')], 100);
    expect(message).toContain('Could not delete some traces');
    expect(message).toContain('Permission denied');
    expect(message).toContain('100 traces were deleted');
    expect(message).not.toContain('..');
  });

  it('says plainly when nothing was deleted', () => {
    // The hook does not use this branch -- when nothing was deleted it rethrows the
    // original `ErrorWrapper` so downstream consumers can still inspect its status. The
    // wording is kept because the function is exported and the branch is reachable.
    const message = describeTraceDeletionOutcome([rejection('Permission denied.')], 0);
    expect(message).toContain('No traces were deleted');
  });

  it('agrees in number for a single trace', () => {
    expect(describeTraceDeletionOutcome([rejection('Nope.')], 1)).toContain('1 trace was deleted');
  });

  it('states one shared reason once', () => {
    const message = describeTraceDeletionOutcome(
      [rejection('Permission denied.'), rejection('Permission denied.')],
      50,
    );
    expect(message.match(/Permission denied/g)).toHaveLength(1);
  });
});

describe('useDeleteTracesMutation cache invalidation', () => {
  // A partial deletion throws, so it lands on the mutation's error path. The invalidation
  // used to hang off `onSuccess`, which that path never reaches -- the permitted chunk was
  // really gone from the server while the table went on listing it, and it reappeared to
  // the user as a delete that had silently failed.

  let queryClient: QueryClient;

  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  const denied = () =>
    new ErrorWrapper(JSON.stringify({ error_code: 'PERMISSION_DENIED', message: 'Permission denied' }), 403);

  // 150 ids is two requests, because the hook chunks at 100. One chunk is permitted and
  // the other refused: the shape a condition produces.
  const ids = Array.from({ length: 150 }, (_, i) => `tr-${i}`);

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  });

  const runMutation = async () => {
    const { result } = renderHook(() => useDeleteTracesMutation(), { wrapper });
    await act(async () => {
      result.current.mutate({ experimentId: '1', traceRequestIds: ids });
    });
    return result;
  };

  it('refreshes the trace list when only some chunks were deleted', async () => {
    jest
      .spyOn(MlflowService, 'deleteTracesV3')
      .mockResolvedValueOnce({ traces_deleted: 100 } as any)
      .mockRejectedValueOnce(denied());

    const result = await runMutation();

    await waitFor(() => expect(result.current.isError).toBe(true));
    // The deletion is reported as failed -- and the list must still refresh, because 100
    // traces really were deleted.
    expect(result.current.error?.message).toContain('100 traces were deleted');
    expect(invalidateMlflowSearchTracesCache).toHaveBeenCalled();
  });

  it('still refreshes on a complete success', async () => {
    jest.spyOn(MlflowService, 'deleteTracesV3').mockResolvedValue({ traces_deleted: 75 } as any);

    const result = await runMutation();

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(invalidateMlflowSearchTracesCache).toHaveBeenCalled();
  });
});
