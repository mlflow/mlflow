import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { IntlProvider } from 'react-intl';

import { NotFoundError, ServiceUnavailableError } from '@databricks/web-shared/errors';
import { SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import {
  getAssessmentValue,
  type Assessment,
  type ModelTrace,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { createAssessment, fetchTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer/api';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useCurrentUserQuery } from '@mlflow/mlflow/src/account/hooks';

import {
  usePairwiseReviewAssessmentMutation,
  type SavePairwiseReviewParams,
} from './usePairwiseReviewAssessmentMutation';

jest.mock('@databricks/web-shared/model-trace-explorer/api', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/model-trace-explorer/api')>(
    '@databricks/web-shared/model-trace-explorer/api',
  ),
  createAssessment: jest.fn(),
  fetchTraceInfoV3: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/account/hooks', () => ({ useCurrentUserQuery: jest.fn() }));

const makeTraceInfo = (traceId: string, assessments: Assessment[] = []) =>
  ({
    trace_id: traceId,
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: '1' } },
    assessments,
  }) as unknown as ModelTraceInfoV3;

// A fake tracking server: creating an assessment that overrides another marks the other invalid.
let storedAssessments: Record<string, Assessment[]> = {};
let nextAssessmentId = 0;
const storeAssessment = (assessment: Assessment) => {
  const traceAssessments = storedAssessments[assessment.trace_id] ?? [];
  const stored = {
    ...assessment,
    assessment_id: `a-${(nextAssessmentId += 1)}`,
    create_time: new Date(Date.UTC(2026, 0, 1, 0, 0, nextAssessmentId)).toISOString(),
    valid: true,
  } as Assessment;
  storedAssessments[assessment.trace_id] = [
    ...traceAssessments.map((existing) =>
      existing.assessment_id === assessment.overrides ? { ...existing, valid: false } : existing,
    ),
    stored,
  ];
  return stored;
};
// Valid stored preferences per trace; more than one means a save left a duplicate.
const validAssessments = (traceId: string) =>
  (storedAssessments[traceId] ?? []).filter((assessment) => assessment.valid);
// The fake server answers after a moment, so saves started together really overlap.
const serverDelay = () => new Promise((resolve) => setTimeout(resolve, 5));

const save = (score: 0 | 1, rationale = ''): SavePairwiseReviewParams => ({
  baseline: { traceInfo: makeTraceInfo('tr-a'), opponentTraceId: 'tr-b', score },
  candidate: { traceInfo: makeTraceInfo('tr-b'), opponentTraceId: 'tr-a', score: score === 1 ? 0 : 1 },
  rationale,
});

const renderSaveHook = (queryClient = new QueryClient()) =>
  renderHook(() => usePairwiseReviewAssessmentMutation(), {
    wrapper: ({ children }) => (
      <IntlProvider locale="en">
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </IntlProvider>
    ),
  });

const mockCurrentUser = (result: Record<string, unknown>) =>
  jest.mocked(useCurrentUserQuery).mockReturnValue({ refetch: jest.fn(), errorUpdatedAt: 0, ...result } as any);

beforeEach(() => {
  storedAssessments = {};
  nextAssessmentId = 0;
  mockCurrentUser({ data: { user: { username: 'alice' } } as any });
  jest.mocked(fetchTraceInfoV3).mockImplementation(async ({ traceId }) => ({
    trace: { trace_info: makeTraceInfo(traceId, storedAssessments[traceId]) },
  }));
  jest.mocked(createAssessment).mockImplementation(async ({ payload }) => {
    await serverDelay();
    return { assessment: storeAssessment(payload.assessment as Assessment) };
  });
});

describe('usePairwiseReviewAssessmentMutation', () => {
  test('refetches the traces table, whose pages are cached as { trace_infos } responses', async () => {
    const queryClient = new QueryClient();
    const pageKey = [SEARCH_MLFLOW_TRACES_QUERY_KEY, 'paged', 'experiment-1', 0];
    queryClient.setQueryData(pageKey, { trace_infos: [makeTraceInfo('tr-a'), makeTraceInfo('tr-b')] });
    const { result } = renderSaveHook(queryClient);

    await act(() => result.current.savePairwiseReview(save(1)));

    expect(queryClient.getQueryState(pageKey)?.isInvalidated).toBe(true);
  });

  test('runs overlapping saves one after the other, leaving one preference per trace', async () => {
    const { result } = renderSaveHook();

    // A comment saved on blur and a new winner chosen by the same click, both from the same stale trace data.
    await act(() =>
      Promise.all([
        result.current.savePairwiseReview(save(1, 'comment')),
        result.current.savePairwiseReview(save(0, 'comment')),
      ]),
    );

    expect(validAssessments('tr-a').map(getAssessmentValue)).toEqual([0]);
    expect(validAssessments('tr-b').map(getAssessmentValue)).toEqual([1]);
  });

  test('does not leave a duplicate preference when a saved write loses its response and is retried', async () => {
    const { result } = renderSaveHook();
    jest.mocked(createAssessment).mockImplementationOnce(async ({ payload }) => {
      await serverDelay();
      storeAssessment(payload.assessment as Assessment);
      throw new TypeError('Failed to fetch');
    });

    await act(() => result.current.savePairwiseReview(save(1)));

    expect(validAssessments('tr-a')).toHaveLength(1);
    expect(validAssessments('tr-b')).toHaveLength(1);
  });

  test('keeps the saved side in the cache when the other side fails', async () => {
    const queryClient = new QueryClient();
    const traceKey = ['getTrace', 'tr-a', undefined];
    queryClient.setQueryData<ModelTrace>(traceKey, { info: makeTraceInfo('tr-a'), data: { spans: [] } });
    jest.mocked(createAssessment).mockImplementation(async ({ payload }) => {
      if (payload.assessment.trace_id === 'tr-b') {
        throw new NotFoundError({});
      }
      return { assessment: storeAssessment(payload.assessment as Assessment) };
    });
    const { result } = renderSaveHook(queryClient);

    await act(() => expect(result.current.savePairwiseReview(save(1))).rejects.toBeInstanceOf(NotFoundError));

    const cachedTrace = queryClient.getQueryData<ModelTrace>(traceKey);
    expect((cachedTrace?.info as ModelTraceInfoV3).assessments).toHaveLength(1);
  });

  test('keeps voting locked while /users/current fails, and uses the placeholder only without auth', async () => {
    const refetch = jest.fn();
    mockCurrentUser({ data: undefined, error: new ServiceUnavailableError({ status: 503 }), refetch });
    const { result, rerender } = renderSaveHook();
    expect(result.current.isReviewerResolved).toBe(false);
    await waitFor(() => expect(refetch).toHaveBeenCalled(), { timeout: 4000 });

    mockCurrentUser({ data: undefined, error: new NotFoundError({ status: 404 }) });
    rerender();
    expect(result.current.isReviewerResolved).toBe(true);
    expect(result.current.reviewerSourceId).toBe('User');
  });
});
