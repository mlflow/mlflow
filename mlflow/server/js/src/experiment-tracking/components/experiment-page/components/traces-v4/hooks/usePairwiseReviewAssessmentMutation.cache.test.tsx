import { describe, expect, jest, test } from '@jest/globals';
import { act, renderHook } from '@testing-library/react';
import { IntlProvider } from 'react-intl';

import { SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import type { Assessment, ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { createAssessment } from '@databricks/web-shared/model-trace-explorer/api';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { usePairwiseReviewAssessmentMutation } from './usePairwiseReviewAssessmentMutation';

jest.mock('@databricks/web-shared/model-trace-explorer/api', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/model-trace-explorer/api')>(
    '@databricks/web-shared/model-trace-explorer/api',
  ),
  createAssessment: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/account/hooks', () => ({
  useCurrentUserQuery: () => ({ data: undefined, isLoading: false }),
}));

const makeTraceInfo = (traceId: string) =>
  ({
    trace_id: traceId,
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: '1' } },
    assessments: [],
  }) as unknown as ModelTraceInfoV3;

describe('usePairwiseReviewAssessmentMutation', () => {
  test('refetches the traces table, whose pages are cached as { trace_infos } responses', async () => {
    jest.mocked(createAssessment).mockImplementation(async ({ payload }) => ({
      assessment: {
        ...payload.assessment,
        assessment_id: `a-${payload.assessment.trace_id}`,
        create_time: '2026-01-01T00:00:00.000Z',
        last_update_time: '2026-01-01T00:00:00.000Z',
      } as Assessment,
    }));
    const queryClient = new QueryClient();
    const pageKey = [SEARCH_MLFLOW_TRACES_QUERY_KEY, 'paged', 'experiment-1', 0];
    queryClient.setQueryData(pageKey, { trace_infos: [makeTraceInfo('tr-a'), makeTraceInfo('tr-b')] });

    const { result } = renderHook(() => usePairwiseReviewAssessmentMutation(), {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
        </IntlProvider>
      ),
    });
    await act(async () => {
      await result.current.savePairwiseReview({
        baseline: { traceInfo: makeTraceInfo('tr-a'), opponentTraceId: 'tr-b', score: 1 },
        candidate: { traceInfo: makeTraceInfo('tr-b'), opponentTraceId: 'tr-a', score: 0 },
        rationale: '',
      });
    });

    expect(queryClient.getQueryState(pageKey)?.isInvalidated).toBe(true);
  });
});
