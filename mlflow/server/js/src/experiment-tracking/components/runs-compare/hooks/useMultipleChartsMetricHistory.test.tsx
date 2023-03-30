import { mount } from 'enzyme';
import { useEffect } from 'react';
import { MetricEntity } from '../../../types';
import { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import { RunsCompareChartType, RunsCompareLineCardConfig } from '../runs-compare.types';
import { useFetchCompareRunsMetricHistory } from './useFetchCompareRunsMetricHistory';
import { useMultipleChartsMetricHistory } from './useMultipleChartsMetricHistory';

jest.mock('./useFetchCompareRunsMetricHistory', () => ({
  useFetchCompareRunsMetricHistory: jest.fn().mockReturnValue({ isLoading: false }),
}));

describe('useMultipleChartsMetricHistory', () => {
  const mountWrappingComponent = (...params: Parameters<typeof useMultipleChartsMetricHistory>) => {
    const resultingData: CompareChartRunData[] = [];
    const Component = () => {
      const { chartRunDataWithHistory, isLoading } = useMultipleChartsMetricHistory(...params);

      useEffect(() => {
        resultingData.length = 0;
        resultingData.push(...chartRunDataWithHistory);
      }, [chartRunDataWithHistory]);

      return <div>{isLoading && <div data-testid='loading' />}</div>;
    };

    return {
      wrapper: mount(<Component />),
      resultingData,
    };
  };

  const mockRun = (id: string) => ({ runInfo: { run_uuid: id } } as any);

  beforeEach(() => {
    (useFetchCompareRunsMetricHistory as jest.Mock).mockClear();
  });

  it('does not fetch metric history when not necessary', async () => {
    // Prepare mocked data
    const mockRuns = [mockRun('100'), mockRun('101')];

    // Mock the component with charts that don't require the history
    const { wrapper, resultingData } = mountWrappingComponent(
      [{ type: RunsCompareChartType.SCATTER }, { type: RunsCompareChartType.CONTOUR }],
      mockRuns,
      {},
    );

    // Fetch hook is called with empty metric list
    expect(useFetchCompareRunsMetricHistory).toBeCalledTimes(1);
    expect(useFetchCompareRunsMetricHistory).toBeCalledWith([], [], {});

    // Output data is the same as input data since there is nothing to enrich
    expect(resultingData).toEqual(mockRuns);

    // Cleanup
    wrapper.unmount();
  });

  it('invokes fetch of a single metric for a single run', async () => {
    // Prepare mocked data
    const mockRuns = [mockRun('100')];

    // Mock the component with a single chart that requires a metric history
    const { wrapper } = mountWrappingComponent(
      [
        {
          type: RunsCompareChartType.LINE,
          metricKey: 'metric_name',
        } as RunsCompareLineCardConfig,
      ],
      mockRuns,
      {},
    );

    // We expect fetch hook to be called for a single metric and a single run
    expect(useFetchCompareRunsMetricHistory).toBeCalledWith(['metric_name'], [mockRuns[0]], {});

    // Cleanup
    wrapper.unmount();
  });

  it('determines a maximum number of runs to fetch for a single metric', async () => {
    // Prepare mocked data of 20 runs with ids 101, 102, 103 etc.
    const mockRuns = new Array(20).fill(null).map((_, index) => mockRun(`${101 + index}`));

    // Mock the component with two charts that require a metric history
    const { wrapper } = mountWrappingComponent(
      [
        {
          type: RunsCompareChartType.LINE,
          metricKey: 'metric_name',
          runsCountToCompare: 6,
        } as RunsCompareLineCardConfig,
        {
          type: RunsCompareChartType.LINE,
          metricKey: 'metric_name',
          runsCountToCompare: 3,
        } as RunsCompareLineCardConfig,
      ],
      mockRuns,
      {},
    );

    // We expect fetch hook to be called for a 6 runs (as the max runsCountToCompare value)
    expect(useFetchCompareRunsMetricHistory).toBeCalledWith(
      ['metric_name'],
      mockRuns.slice(0, 6),
      {},
    );

    // Cleanup
    wrapper.unmount();
  });

  it('fetches multiple metrics at once', async () => {
    // Prepare mocked data of 20 runs with ids 101, 102, 103 etc.
    const mockRuns = new Array(20).fill(null).map((_, index) => mockRun(`${101 + index}`));

    // Mock the component with two charts that require a metric history
    const { wrapper } = mountWrappingComponent(
      [
        {
          type: RunsCompareChartType.LINE,
          metricKey: 'metric_1',
          runsCountToCompare: 10,
        } as RunsCompareLineCardConfig,
        {
          type: RunsCompareChartType.LINE,
          metricKey: 'metric_2',
          runsCountToCompare: 10,
        } as RunsCompareLineCardConfig,
      ],
      mockRuns,
      {},
    );

    // We expect fetch hook to be called for two distinct metrics for 10 runs
    expect(useFetchCompareRunsMetricHistory).toBeCalledWith(
      ['metric_1', 'metric_2'],
      mockRuns.slice(0, 10),
      {},
    );

    // Cleanup
    wrapper.unmount();
  });

  it('properly enriches the runs collection with metric history', async () => {
    // Prepare mocked data of 5 runs with ids 101, 102, 103 etc.
    const mockRuns = new Array(5).fill(null).map((_, index) => mockRun(`${101 + index}`));

    // Mock the component with two charts that requires a metric history
    // and provide a fake fragment from the redux store's "metricsByRunUuid"
    const { resultingData, wrapper } = mountWrappingComponent(
      [
        {
          type: RunsCompareChartType.LINE,
          metricKey: 'metric_1',
          runsCountToCompare: 10,
        } as RunsCompareLineCardConfig,
        {
          type: RunsCompareChartType.LINE,
          metricKey: 'metric_2',
          runsCountToCompare: 10,
        } as RunsCompareLineCardConfig,
      ],
      mockRuns,
      {
        '101': {
          metric_1: [{ key: 'metric_1', value: 100 } as MetricEntity],
        },
        '104': {
          metric_1: [{ key: 'metric_1', value: 400 } as MetricEntity],
          metric_2: [{ key: 'metric_2', value: 400 } as MetricEntity],
        },
      },
    );

    // We expect the resulting data to be enriched with metric history for runs with index 0 and 3
    expect(resultingData).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          ...mockRuns[0],
          metricsHistory: { metric_1: [{ key: 'metric_1', value: 100 }] },
        }),
        expect.objectContaining({
          ...mockRuns[3],
          metricsHistory: {
            metric_1: [{ key: 'metric_1', value: 400 }],
            metric_2: [{ key: 'metric_2', value: 400 }],
          },
        }),
        // Remaining runs are passed without enriching
        mockRuns[1],
        mockRuns[2],
        mockRuns[4],
      ]),
    );

    // Cleanup
    wrapper.unmount();
  });
});
