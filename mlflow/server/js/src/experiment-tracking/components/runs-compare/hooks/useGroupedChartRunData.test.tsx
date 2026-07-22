import { describe, it, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { type UseGroupedChartRunDataParams, useGroupedChartRunData } from './useGroupedChartRunData';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunGroupingAggregateFunction } from '../../experiment-page/utils/experimentPage.row-types';
import { last } from 'lodash';
import type { SampledMetricsByRun } from '../../runs-charts/hooks/useSampledMetricHistory';

const ungroupedDataTrace: RunsChartsRunData = {
  displayName: 'Ungrouped run 1',
  belongsToGroup: false,
  runInfo: {},
} as any;

const testChartDataTraces: RunsChartsRunData[] = [
  {
    groupParentInfo: {
      runUuids: ['run_1', 'run_2'],
    },
    displayName: 'Group 1',
    uuid: 'group-1',
  } as any,
  {
    groupParentInfo: {
      runUuids: ['run_3', 'run_4'],
    },
    displayName: 'Group 2',
    uuid: 'group-2',
  },
];

const testSampledRunMetrics: Record<string, SampledMetricsByRun> = {
  run_1: {
    runUuid: 'run_1',
    metric_1: {
      metricsHistory: [
        {
          key: 'metric_1',
          value: 1,
          step: 1,
          timestamp: 1,
        },
        {
          key: 'metric_1',
          value: 10,
          step: 2,
          timestamp: 10,
        },
      ],
    },
  },
  run_2: {
    runUuid: 'run_2',
    metric_1: {
      metricsHistory: [
        {
          key: 'metric_1',
          value: 5,
          step: 1,
          timestamp: 1,
        },
        {
          key: 'metric_1',
          value: 20,
          step: 2,
          timestamp: 30,
        },
      ],
    },
  },
  run_3: {
    runUuid: 'run_3',
    metric_1: {
      metricsHistory: [
        {
          key: 'metric_1',
          value: 20,
          step: 1,
          timestamp: 1,
        },
      ],
    },
  },
  run_4: {
    runUuid: 'run_3',
    metric_1: {
      metricsHistory: [
        {
          key: 'metric_1',
          value: 40,
          step: 1,
          timestamp: 1,
        },
        {
          key: 'metric_1',
          value: 500,
          step: 2,
          timestamp: 2,
        },
      ],
    },
  },
} as any;

describe('useGroupedChartRunData', () => {
  const renderConfiguredHook = (params: Partial<UseGroupedChartRunDataParams>) =>
    renderHook<RunsChartsRunData[], UseGroupedChartRunDataParams>((props) => useGroupedChartRunData(props), {
      initialProps: {
        enabled: params.enabled ?? true,
        ungroupedRunsData: params.ungroupedRunsData ?? [],
        metricKeys: params.metricKeys ?? [],
        sampledDataResultsByRunUuid: params.sampledDataResultsByRunUuid ?? {},
        aggregateFunction: params.aggregateFunction ?? RunGroupingAggregateFunction.Average,
        ignoreOutliers: params.ignoreOutliers ?? false,
      },
    });
  it('should return the fallback data if grouping is disabled', () => {
    const ungroupedRunsData = [{ uuid: '1' }, { uuid: '2' }] as any;
    const result = renderConfiguredHook({
      ungroupedRunsData,
      enabled: false,
    });
    expect(result.result.current).toEqual(ungroupedRunsData);
  });

  it('should aggregate the data and return proper group boundaries', () => {
    const result = renderConfiguredHook({
      ungroupedRunsData: testChartDataTraces,
      sampledDataResultsByRunUuid: testSampledRunMetrics,
      enabled: true,
      aggregateFunction: RunGroupingAggregateFunction.Average,
      metricKeys: ['metric_1'],
    });

    const groupedData = result.result.current;
    expect(groupedData).toHaveLength(2);

    const [firstGroup, secondGroup] = groupedData;

    expect(firstGroup.aggregatedMetricsHistory?.['metric_1']).toEqual({
      average: [
        {
          // Expect value to be average of two source entries
          key: 'metric_1',
          step: 1,
          timestamp: 1,
          value: 3,
        },
        {
          // Expect value to be average of two source entries
          key: 'metric_1',
          step: 2,
          // For the second step in the first group, we expect to be average timestamp of two source entries
          timestamp: 20,
          value: 15,
        },
      ],
      max: [
        {
          key: 'metric_1',
          step: 1,
          timestamp: 1,
          value: 5,
        },
        {
          key: 'metric_1',
          step: 2,
          timestamp: 20,
          value: 20,
        },
      ],
      min: [
        {
          key: 'metric_1',
          step: 1,
          timestamp: 1,
          value: 1,
        },
        {
          key: 'metric_1',
          step: 2,
          timestamp: 20,
          value: 10,
        },
      ],
    });
    // We use average aggregation so we expect the metricsHistory to be the same as the calculated average dataset
    expect(firstGroup.metricsHistory?.['metric_1']).toEqual(firstGroup.aggregatedMetricsHistory?.['metric_1'].average);
    expect(secondGroup.aggregatedMetricsHistory?.['metric_1']).toEqual({
      average: [
        {
          key: 'metric_1',
          step: 1,
          timestamp: 1,
          value: 30,
        },
        {
          key: 'metric_1',
          step: 2,
          timestamp: 2,
          // In the second group, we have only one source entry for the second step,
          // so we expect the average to be the same as the value
          value: 500,
        },
      ],
      max: [
        {
          key: 'metric_1',
          step: 1,
          timestamp: 1,
          value: 40,
        },
        {
          key: 'metric_1',
          step: 2,
          timestamp: 2,
          value: 500,
        },
      ],
      min: [
        {
          key: 'metric_1',
          step: 1,
          timestamp: 1,
          value: 20,
        },
        {
          key: 'metric_1',
          step: 2,
          timestamp: 2,
          value: 500,
        },
      ],
    });
    expect(secondGroup.metricsHistory?.['metric_1']).toEqual(
      secondGroup.aggregatedMetricsHistory?.['metric_1'].average,
    );
  });

  it('should aggregate the data and append ungrouped runs at the end', () => {
    const result = renderConfiguredHook({
      ungroupedRunsData: [...testChartDataTraces, ungroupedDataTrace],
      sampledDataResultsByRunUuid: testSampledRunMetrics,
      enabled: true,
      aggregateFunction: RunGroupingAggregateFunction.Average,
      metricKeys: ['metric_1'],
    });

    const groupedData = result.result.current;
    expect(groupedData).toHaveLength(3);

    expect(last(groupedData)).toEqual(ungroupedDataTrace);
  });

  // The backend samples each run independently, so runs with differing history lengths can come
  // back with misaligned steps. Aggregation keys off the step value (not the index), so each step
  // is aggregated from whichever runs actually reported it.
  it('should aggregate runs with misaligned steps without producing NaN or Infinity', () => {
    const misalignedGroup: RunsChartsRunData[] = [
      {
        groupParentInfo: { runUuids: ['run_a', 'run_b'] },
        displayName: 'Misaligned group',
        uuid: 'group-misaligned',
      } as any,
    ];

    const misalignedMetrics: Record<string, SampledMetricsByRun> = {
      run_a: {
        runUuid: 'run_a',
        metric_1: {
          metricsHistory: [
            { key: 'metric_1', value: 10, step: 0, timestamp: 1 },
            { key: 'metric_1', value: 20, step: 2, timestamp: 2 },
            { key: 'metric_1', value: 30, step: 4, timestamp: 3 },
          ],
        },
      } as any,
      run_b: {
        runUuid: 'run_b',
        metric_1: {
          metricsHistory: [
            { key: 'metric_1', value: 100, step: 0, timestamp: 1 },
            { key: 'metric_1', value: 200, step: 3, timestamp: 2 },
            { key: 'metric_1', value: 300, step: 4, timestamp: 3 },
          ],
        },
      } as any,
    };

    const result = renderConfiguredHook({
      ungroupedRunsData: misalignedGroup,
      sampledDataResultsByRunUuid: misalignedMetrics,
      enabled: true,
      aggregateFunction: RunGroupingAggregateFunction.Average,
      metricKeys: ['metric_1'],
    });

    const aggregated = result.result.current[0].aggregatedMetricsHistory?.['metric_1'];

    // The union of both runs' steps is covered, in ascending order
    expect(aggregated?.average.map(({ step }) => step)).toEqual([0, 2, 3, 4]);

    // Steps reported by both runs average across them; steps reported by a single run
    // take that run's value rather than degrading to NaN/Infinity
    expect(aggregated?.average.map(({ value }) => value)).toEqual([55, 20, 200, 165]);
    expect(aggregated?.min.map(({ value }) => value)).toEqual([10, 20, 200, 30]);
    expect(aggregated?.max.map(({ value }) => value)).toEqual([100, 20, 200, 300]);

    for (const entry of [...aggregated!.average, ...aggregated!.min, ...aggregated!.max]) {
      expect(Number.isFinite(entry.value)).toBe(true);
    }
  });
});
