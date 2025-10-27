import { fetchEndpoint } from '../../common/utils/FetchUtils';
import { createChartAxisRangeKey } from '../components/runs-charts/components/RunsCharts.common';
import type { SampledMetricsByRunUuidState } from '../types';
import { getSampledMetricHistoryBulkAction } from './SampledMetricHistoryService';

jest.mock('../../common/utils/FetchUtils', () => ({
  fetchEndpoint: jest.fn(),
}));

const testRange: [number, number] = [300, 500];
const testRangeKey = createChartAxisRangeKey(testRange);

describe('getSampledMetricHistoryBulkAction service function', () => {
  const testDispatch = jest.fn();
  beforeEach(() => {
    testDispatch.mockClear();
    jest.mocked(fetchEndpoint).mockClear();
  });
  const runAction = (state: SampledMetricsByRunUuidState) => {
    const actionCreator = getSampledMetricHistoryBulkAction(['run_1', 'run_2'], 'metric_key', undefined, testRange);
    actionCreator(testDispatch, () => ({ entities: { sampledMetricsByRunUuid: state } } as any));
  };
  it('should be able to retrieve sampled metric history for all runs', () => {
    runAction({});
    expect(fetchEndpoint).toHaveBeenCalledWith({
      relativeUrl: expect.stringMatching(
        /get-history-bulk-interval\?run_ids=run_1&run_ids=run_2&metric_key=metric_key/,
      ),
    });
  });

  it('should skip retrieval of sampled metric history for runs with the data already loaded', () => {
    runAction({
      run_1: { metric_key: { [testRangeKey]: { metricsHistory: [], loading: false } } },
    });
    expect(fetchEndpoint).toHaveBeenCalledWith({
      relativeUrl: expect.stringMatching(/get-history-bulk-interval\?run_ids=run_2&metric_key=metric_key/),
    });
  });
  it('should skip retrieval of sampled metric history for runs with error reported', () => {
    runAction({
      run_2: { metric_key: { [testRangeKey]: { error: true, loading: false } } },
    });
    expect(fetchEndpoint).toHaveBeenCalledWith({
      relativeUrl: expect.stringMatching(/get-history-bulk-interval\?run_ids=run_1&metric_key=metric_key/),
    });
  });
});
