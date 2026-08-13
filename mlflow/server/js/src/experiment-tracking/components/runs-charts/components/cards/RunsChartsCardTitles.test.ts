import { describe, expect, test } from '@jest/globals';
import { RunsChartType, type RunsChartsBarCardConfig, type RunsChartsLineCardConfig } from '../../runs-charts.types';
import { RunsChartsLineChartXAxisType } from '../RunsCharts.common';
import { getBarChartTitle, getBarChartTitleTooltip } from './RunsChartsBarChartCard';
import { getV2ChartTitle, getV2ChartTitleTooltip } from './RunsChartsLineChartCard';

const fullMetricKey = 'train/losses/grouped_by_x/after_y/mae';

type ChartConfig = RunsChartsBarCardConfig | RunsChartsLineCardConfig;

const testCases: {
  chartType: string;
  config: ChartConfig;
  getTitle: (config: ChartConfig, useMetricDisplayName?: boolean) => string;
  getTooltip: (config: ChartConfig) => string;
}[] = [
  {
    chartType: 'bar',
    config: {
      type: RunsChartType.BAR,
      metricKey: fullMetricKey,
      displayName: 'mae',
      isGenerated: true,
      deleted: false,
    } as RunsChartsBarCardConfig,
    getTitle: (config, useMetricDisplayName) =>
      getBarChartTitle(config as RunsChartsBarCardConfig, useMetricDisplayName),
    getTooltip: (config) => getBarChartTitleTooltip(config as RunsChartsBarCardConfig),
  },
  {
    chartType: 'line',
    config: {
      type: RunsChartType.LINE,
      metricKey: fullMetricKey,
      displayName: 'mae',
      isGenerated: true,
      deleted: false,
      lineSmoothness: 0,
      scaleType: 'linear',
      xAxisScaleType: 'linear',
      xAxisKey: RunsChartsLineChartXAxisType.STEP,
      selectedXAxisMetricKey: '',
    } as RunsChartsLineCardConfig,
    getTitle: (config, useMetricDisplayName) =>
      getV2ChartTitle(config as RunsChartsLineCardConfig, useMetricDisplayName),
    getTooltip: (config) => getV2ChartTitleTooltip(config as RunsChartsLineCardConfig),
  },
];

describe.each(testCases)('$chartType chart titles', ({ config, getTitle, getTooltip }) => {
  test('uses the section-relative display name when enabled', () => {
    expect(getTitle(config)).toBe('mae');
  });

  test('uses the full metric key when display names are disabled', () => {
    expect(getTitle(config, false)).toBe(fullMetricKey);
  });

  test('preserves the full metric key in the title tooltip', () => {
    expect(getTooltip(config)).toBe(fullMetricKey);
  });
});
