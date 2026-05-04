import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useSampledMetricHistory } from './useSampledMetricHistory';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
import { getSampledMetricHistoryBulkAction } from '../../../sdk/SampledMetricHistoryService';
import React from 'react';
import { shouldEnableGraphQLSampledMetrics } from '../../../../common/utils/FeatureUtils';
import { IntlProvider } from 'react-intl';
import { TestApolloProvider } from '../../../../common/utils/TestApolloProvider';
import { QueryClient, QueryClientProvider } from '../../../../common/utils/reactQueryHooks';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL } from '../../../utils/MetricsUtils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldEnableGraphQLSampledMetrics: jest.fn(),
}));

jest.mock('../../../sdk/SampledMetricHistoryService', () => ({
  getSampledMetricHistoryBulkAction: jest.fn(),
}));

jest.useFakeTimers();

const hookWrapper: React.FC<React.PropsWithChildren<unknown>> = ({ children }) => {
  const queryClient = new QueryClient();
  return (
    <IntlProvider locale="en">
      <TestApolloProvider>
        <QueryClientProvider client={queryClient}>
          <MockedReduxStoreProvider
            state={{
              entities: {
                sampledMetricsByRunUuid: {},
              },
            }}
          >
            {children}
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>
    </IntlProvider>
  );
};

describe('useSampledMetricHistory (REST)', () => {
  const server = setupServer();

  let callCount = 0;

  server.use(
    rest.get('/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval', (req, res, ctx) => {
      const runId = req.url.searchParams.get('run_ids');
      const metricKey = req.url.searchParams.get('metric_key');

      const metrics = [
        {
          key: metricKey,
          run_id: runId,
          step: 0,
          timestamp: 1712345000000,
          value: 100,
        },
      ];
      if (callCount > 0) {
        metrics.push({
          key: metricKey,
          run_id: runId,
          step: 1,
          timestamp: 1712345000001,
          value: 200,
        });
      }
      callCount++;
      return res(ctx.json({ metrics }));
    }),
  );

  beforeEach(() => {
    jest.mocked(shouldEnableGraphQLSampledMetrics).mockImplementation(() => false);
    jest.mocked(getSampledMetricHistoryBulkAction).mockClear();
    jest.mocked(getSampledMetricHistoryBulkAction).mockImplementation(
      () =>
        ({
          payload: Promise.resolve({}),
          type: 'GET_SAMPLED_METRIC_HISTORY_API_BULK',
        }) as any,
    );
    callCount = 0;
  });

  test('should return the data and refresh it automatically', async () => {
    const { result } = renderHook(
      () =>
        useSampledMetricHistory({
          runUuids: ['run-uuid-1'],
          metricKeys: ['metric-a'],
          enabled: true,
          autoRefreshEnabled: true,
        }),
      {
        wrapper: hookWrapper,
      },
    );

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.resultsByRunUuid['run-uuid-1']?.['metric-a'].metricsHistory).toEqual([
        {
          key: 'metric-a',
          run_id: 'run-uuid-1',
          step: 0,
          timestamp: 1712345000000,
          value: 100,
        },
      ]);
    });

    await act(() => {
      // advanceTimersByTimeAsync might not be available in OSS jest runtime
      return (jest.advanceTimersByTimeAsync ?? jest.advanceTimersByTime)(
        EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL,
      );
    });

    await waitFor(() => {
      expect(result.current.resultsByRunUuid['run-uuid-1']?.['metric-a'].metricsHistory).toEqual([
        {
          key: 'metric-a',
          run_id: 'run-uuid-1',
          step: 0,
          timestamp: 1712345000000,
          value: 100,
        },
        {
          key: 'metric-a',
          run_id: 'run-uuid-1',
          step: 1,
          timestamp: 1712345000001,
          value: 200,
        },
      ]);
    });
  });
});
