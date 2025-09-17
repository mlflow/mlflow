import { fulfilled, pending, rejected } from '../../common/utils/ActionUtils';
import { createChartAxisRangeKey } from '../components/runs-charts/components/RunsCharts.common';
import { sampledMetricsByRunUuid } from './SampledMetricsReducer';

const testRunIds = ['run_1'];
const testMetricKey = 'test_metric_key';
const testDefaultRangeKey = createChartAxisRangeKey();
const testCustomRangeKey = createChartAxisRangeKey([-100, 200]);

describe('SampledMetricsReducer', () => {
  it('should be able to initialize empty state', () => {
    expect(sampledMetricsByRunUuid(undefined, {} as any)).toEqual({});
  });

  it('should be able to start loading sampled metrics for two ranges', () => {
    const pendingFirstRangeState = sampledMetricsByRunUuid(
      {},
      {
        type: pending('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
        meta: {
          runUuids: testRunIds,
          key: testMetricKey,
          rangeKey: testDefaultRangeKey,
          maxResults: 100,
        },
      },
    );
    const pendingSecondRangeState = sampledMetricsByRunUuid(pendingFirstRangeState, {
      type: pending('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
      meta: {
        runUuids: testRunIds,
        key: testMetricKey,
        rangeKey: testCustomRangeKey,
        maxResults: 100,
      },
    });
    expect(pendingSecondRangeState).toEqual({
      run_1: {
        test_metric_key: {
          DEFAULT: {
            loading: true,
            refreshing: false,
            metricsHistory: undefined,
          },
          '-100,200': {
            loading: true,
            refreshing: false,
            metricsHistory: undefined,
          },
        },
      },
    });
  });

  it('should be able to populate sampled metrics for one of ranges', () => {
    const pendingFirstRangeState = sampledMetricsByRunUuid(
      {},
      {
        type: pending('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
        meta: {
          runUuids: testRunIds,
          key: testMetricKey,
          rangeKey: testDefaultRangeKey,
          maxResults: 100,
        },
      },
    );
    const pendingSecondRangeState = sampledMetricsByRunUuid(pendingFirstRangeState, {
      type: pending('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
      meta: {
        runUuids: testRunIds,
        key: testMetricKey,
        rangeKey: testCustomRangeKey,
        maxResults: 100,
      },
    });
    const loadedFirstRangeState = sampledMetricsByRunUuid(pendingSecondRangeState, {
      type: 'GET_SAMPLED_METRIC_HISTORY_API_BULK_FULFILLED',
      payload: {
        metrics: [
          { run_id: testRunIds[0], key: testMetricKey, value: 123, step: 0, timestamp: 1 } as any,
          { run_id: 'other_run', key: testMetricKey, value: 123, step: 0, timestamp: 1 } as any,
        ],
      },
      meta: {
        runUuids: testRunIds,
        key: testMetricKey,
        rangeKey: testDefaultRangeKey,
        maxResults: 100,
      },
    });
    expect(loadedFirstRangeState).toEqual({
      run_1: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: false,
            lastUpdatedTime: expect.anything(),
            metricsHistory: [{ run_id: testRunIds[0], key: testMetricKey, value: 123, step: 0, timestamp: 1 }],
          },
          '-100,200': {
            loading: true,
            refreshing: false,
            metricsHistory: undefined,
            lastUpdatedTime: undefined,
          },
        },
      },
    });
  });

  it('should be able to save error state', () => {
    const thrownError = new Error('This is an exception');
    const pendingFirstRangeState = sampledMetricsByRunUuid(
      {},
      {
        type: pending('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
        meta: {
          runUuids: testRunIds,
          key: testMetricKey,
          rangeKey: testDefaultRangeKey,
          maxResults: 100,
        },
      },
    );
    const errorInTheRangeState = sampledMetricsByRunUuid(pendingFirstRangeState, {
      type: rejected('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
      payload: thrownError,
      meta: {
        runUuids: testRunIds,
        key: testMetricKey,
        rangeKey: testDefaultRangeKey,
        maxResults: 100,
      },
    });
    expect(errorInTheRangeState).toEqual({
      run_1: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: false,
            error: thrownError,
          },
        },
      },
    });
  });

  it('indicates "refreshing" state of sampled metric data', () => {
    const testMetricsHistory = [{ run_id: 'run_1', key: testMetricKey, value: 123, step: 0, timestamp: 1 }] as any;
    const initialState = {
      run_1: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: false,
            metricsHistory: testMetricsHistory,
          },
        },
      },
    };
    const pendingState = sampledMetricsByRunUuid(initialState, {
      type: pending('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
      meta: {
        runUuids: testRunIds,
        key: testMetricKey,
        rangeKey: testDefaultRangeKey,
        maxResults: 100,
        isRefreshing: true,
      },
    });

    expect(pendingState).toEqual({
      run_1: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: true,
            metricsHistory: testMetricsHistory,
          },
        },
      },
    });

    const refreshedState = sampledMetricsByRunUuid(pendingState, {
      type: fulfilled('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
      payload: { metrics: testMetricsHistory },
      meta: {
        runUuids: testRunIds,
        key: testMetricKey,
        rangeKey: testDefaultRangeKey,
        maxResults: 100,
        isRefreshing: true,
      },
    });

    expect(refreshedState).toEqual({
      run_1: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: false,
            metricsHistory: testMetricsHistory,
            lastUpdatedTime: expect.anything(),
          },
        },
      },
    });
  });

  it('handles refreshing state correctly with multiple run IDs', () => {
    const multipleRunIds = ['run_1', 'run_2', 'run_3'];
    const testMetricsHistory = [{ run_id: 'run_1', key: testMetricKey, value: 123, step: 0, timestamp: 1 }] as any;

    const initialState = {
      run_1: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: false,
            metricsHistory: testMetricsHistory,
          },
        },
      },
      run_2: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: false,
            metricsHistory: testMetricsHistory,
          },
        },
      },
      run_3: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: false,
            metricsHistory: testMetricsHistory,
          },
        },
      },
    };

    const pendingState = sampledMetricsByRunUuid(initialState, {
      type: pending('GET_SAMPLED_METRIC_HISTORY_API_BULK'),
      meta: {
        runUuids: multipleRunIds,
        key: testMetricKey,
        rangeKey: testDefaultRangeKey,
        maxResults: 100,
        isRefreshing: true,
      },
    });

    expect(pendingState).toEqual({
      run_1: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: true,
            metricsHistory: testMetricsHistory,
          },
        },
      },
      run_2: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: true,
            metricsHistory: testMetricsHistory,
          },
        },
      },
      run_3: {
        test_metric_key: {
          DEFAULT: {
            loading: false,
            refreshing: true,
            metricsHistory: testMetricsHistory,
          },
        },
      },
    });
  });
});
