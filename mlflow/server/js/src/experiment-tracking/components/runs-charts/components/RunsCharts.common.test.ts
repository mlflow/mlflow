import { describe, it, expect, test } from '@jest/globals';
import type { MetricEntity } from '../../../types';
import type {
  RunsChartsBarCardConfig,
  RunsChartsContourCardConfig,
  RunsChartsDifferenceCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsParallelCardConfig,
  RunsChartsScatterCardConfig,
} from '../runs-charts.types';
import { RunsChartType } from '../runs-charts.types';
import type { RunsChartsRunData } from './RunsCharts.common';
import { createEmptyChartCardPredicate, isEmptyChartCard, removeOutliersFromMetricHistory } from './RunsCharts.common';

describe('removeOutliersFromMetricHistory', () => {
  it.each([
    {
      label: 'all same and two outliers',
      makeData: (): MetricEntity[] => {
        const history = Array(20).fill({ step: 0, value: 100, key: 'metric', timestamp: 0 });
        history.push({ step: 0, value: 1000, key: 'metric', timestamp: 0 });
        history.unshift({ step: 0, value: 0, key: 'metric', timestamp: 0 });
        return history;
      },
      predicate: ({ value }: MetricEntity) => value === 100,
    },
    {
      label: 'linear progression',
      makeData: (): MetricEntity[] => Array(100).map((_, i) => ({ step: 0, value: i, key: 'metric', timestamp: 0 })),
      predicate: ({ value }: MetricEntity) => value >= 5 && value <= 95,
    },
  ])('should remove outliers correctly ($label)', ({ label, makeData, predicate }) => {
    const result = removeOutliersFromMetricHistory(makeData());
    expect(result.every(predicate)).toBeTruthy();
  });
});

describe('isEmptyChartCard', () => {
  test('should consider parallel chart with no params/metrics set an empty one', () => {
    const chartConfig: RunsChartsParallelCardConfig = {
      deleted: false,
      isGenerated: false,
      selectedMetrics: [],
      selectedParams: [],
      type: RunsChartType.PARALLEL,
    };
    const isEmpty = isEmptyChartCard(
      [
        {
          uuid: 'some-run-uuid',
        } as RunsChartsRunData,
      ],
      chartConfig,
    );

    expect(isEmpty).toEqual(true);
  });
});

describe('createEmptyChartCardPredicate', () => {
  const mockRunData: RunsChartsRunData[] = [
    {
      uuid: 'run-1',
      displayName: 'Run 1',
      metrics: { metric_1: { key: 'metric_1', value: 10 }, metric_2: { key: 'metric_2', value: 20 } },
      params: {},
      tags: {},
      images: {},
      hidden: false,
    } as unknown as RunsChartsRunData,
    {
      uuid: 'run-2',
      displayName: 'Run 2',
      metrics: { metric_3: { key: 'metric_3', value: 30 } },
      params: {},
      tags: {},
      images: {},
      hidden: false,
    } as unknown as RunsChartsRunData,
  ];

  test('should create a predicate that filters hidden runs', () => {
    const runDataWithHidden: RunsChartsRunData[] = [
      ...mockRunData,
      {
        uuid: 'run-3',
        displayName: 'Run 3',
        metrics: { metric_4: { key: 'metric_4', value: 40 } },
        params: {},
        tags: {},
        images: {},
        hidden: true,
      } as unknown as RunsChartsRunData,
    ];

    const predicate = createEmptyChartCardPredicate(runDataWithHidden);
    const barChart: RunsChartsBarCardConfig = {
      type: RunsChartType.BAR,
      metricKey: 'metric_4',
      deleted: false,
      isGenerated: false,
    };

    expect(predicate(barChart)).toBe(true);
  });

  test('should identify empty bar chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const emptyBarChart: RunsChartsBarCardConfig = {
      type: RunsChartType.BAR,
      metricKey: 'nonexistent_metric',
      deleted: false,
      isGenerated: false,
    };

    expect(predicate(emptyBarChart)).toBe(true);
  });

  test('should identify non-empty bar chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const nonEmptyBarChart: RunsChartsBarCardConfig = {
      type: RunsChartType.BAR,
      metricKey: 'metric_1',
      deleted: false,
      isGenerated: false,
    };

    expect(predicate(nonEmptyBarChart)).toBe(false);
  });

  test('should identify empty contour chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const emptyContourChart: RunsChartsContourCardConfig = {
      type: RunsChartType.CONTOUR,
      xaxis: { key: 'metric_x', type: 'METRIC' },
      yaxis: { key: 'metric_y', type: 'METRIC' },
      zaxis: { key: 'metric_z', type: 'METRIC' },
      deleted: false,
      isGenerated: false,
    };

    expect(predicate(emptyContourChart)).toBe(true);
  });

  test('should identify non-empty contour chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const nonEmptyContourChart: RunsChartsContourCardConfig = {
      type: RunsChartType.CONTOUR,
      xaxis: { key: 'metric_1', type: 'METRIC' },
      yaxis: { key: 'metric_2', type: 'METRIC' },
      zaxis: { key: 'metric_3', type: 'METRIC' },
      deleted: false,
      isGenerated: false,
    };

    expect(predicate(nonEmptyContourChart)).toBe(false);
  });

  test('should identify empty scatter chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const emptyScatterChart: RunsChartsScatterCardConfig = {
      type: RunsChartType.SCATTER,
      xaxis: { key: 'metric_x', type: 'METRIC' },
      yaxis: { key: 'metric_y', type: 'METRIC' },
      deleted: false,
      isGenerated: false,
    } as unknown as RunsChartsScatterCardConfig;

    expect(predicate(emptyScatterChart)).toBe(true);
  });

  test('should identify non-empty scatter chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const nonEmptyScatterChart: RunsChartsScatterCardConfig = {
      type: RunsChartType.SCATTER,
      xaxis: { key: 'metric_1', type: 'METRIC' },
      yaxis: { key: 'metric_2', type: 'METRIC' },
      deleted: false,
      isGenerated: false,
    } as unknown as RunsChartsScatterCardConfig;

    expect(predicate(nonEmptyScatterChart)).toBe(false);
  });

  test('should identify empty line chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const emptyLineChart: RunsChartsLineCardConfig = {
      type: RunsChartType.LINE,
      metricKey: 'nonexistent_metric',
      selectedMetricKeys: ['nonexistent_metric'],
      deleted: false,
      isGenerated: false,
    } as unknown as RunsChartsLineCardConfig;

    expect(predicate(emptyLineChart)).toBe(true);
  });

  test('should identify non-empty line chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const nonEmptyLineChart: RunsChartsLineCardConfig = {
      type: RunsChartType.LINE,
      metricKey: 'metric_1',
      selectedMetricKeys: ['metric_1', 'metric_2'],
      deleted: false,
      isGenerated: false,
      runsCountToCompare: 0,
    } as unknown as RunsChartsLineCardConfig;

    expect(predicate(nonEmptyLineChart)).toBe(false);
  });

  test('should identify empty difference chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const emptyDifferenceChart: RunsChartsDifferenceCardConfig = {
      type: RunsChartType.DIFFERENCE,
      compareGroups: [],
      deleted: false,
      isGenerated: false,
    } as unknown as RunsChartsDifferenceCardConfig;

    expect(predicate(emptyDifferenceChart)).toBe(true);
  });

  test('should identify non-empty difference chart', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);
    const nonEmptyDifferenceChart: RunsChartsDifferenceCardConfig = {
      type: RunsChartType.DIFFERENCE,
      compareGroups: [{ groupId: 'group1' }],
      deleted: false,
      isGenerated: false,
    } as unknown as RunsChartsDifferenceCardConfig;

    expect(predicate(nonEmptyDifferenceChart)).toBe(false);
  });

  test('should reuse predicate for multiple chart configs efficiently', () => {
    const predicate = createEmptyChartCardPredicate(mockRunData);

    const chart1: RunsChartsBarCardConfig = {
      type: RunsChartType.BAR,
      metricKey: 'metric_1',
      deleted: false,
      isGenerated: false,
    };

    const chart2: RunsChartsBarCardConfig = {
      type: RunsChartType.BAR,
      metricKey: 'metric_2',
      deleted: false,
      isGenerated: false,
    };

    const chart3: RunsChartsBarCardConfig = {
      type: RunsChartType.BAR,
      metricKey: 'nonexistent',
      deleted: false,
      isGenerated: false,
    };

    expect(predicate(chart1)).toBe(false);
    expect(predicate(chart2)).toBe(false);
    expect(predicate(chart3)).toBe(true);
  });
});
