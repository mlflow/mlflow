import { describe, test, expect } from '@jest/globals';
import { RunsChartsCardConfig, RunsChartsLineCardConfig, RunsChartType } from './runs-charts.types';
import type { RunsChartsRunData } from './components/RunsCharts.common';

describe('RunsChartsCardConfig.getBaseChartAndSectionConfigs', () => {
  const createMockRunData = (metrics: Record<string, any>): RunsChartsRunData => ({
    uuid: 'test-run-uuid',
    displayName: 'Test Run',
    metrics,
    params: {},
    tags: {},
    images: {},
  });

  describe('nodeLevelMetricsConfig integration', () => {
    test('creates Node system metrics section and charts when common node metrics are provided', () => {
      const runsData = [
        createMockRunData({
          'system/node_0/cpu_utilization_percentage': { key: 'system/node_0/cpu_utilization_percentage', value: 50 },
          'system/node_1/cpu_utilization_percentage': { key: 'system/node_1/cpu_utilization_percentage', value: 60 },
        }),
      ];

      const nodeLevelMetricsConfig = {
        nodeIndexes: ['0', '1'],
        commonMetrics: ['cpu_utilization_percentage'],
        gpuIndexes: [],
        commonGpuMetrics: [],
        enabled: true as const,
      };

      const { resultChartSet, resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
        runsData,
        nodeLevelMetricsConfig,
      });

      // Verify Node system metrics section was created
      const nodeSectionExists = resultSectionSet.some((section) => section.name === 'Node system metrics');
      expect(nodeSectionExists).toBe(true);

      // Verify chart was created for the common metric
      const cpuChart = resultChartSet.find(
        (chart) =>
          chart instanceof RunsChartsLineCardConfig &&
          chart.nodeLevelSystemMetricConfiguration?.metric === 'cpu_utilization_percentage' &&
          chart.nodeLevelSystemMetricConfiguration?.type === 'node',
      ) as RunsChartsLineCardConfig;

      expect(cpuChart).toBeDefined();
      expect(cpuChart.displayName).toBe('cpu_utilization_percentage');
      expect(cpuChart.selectedMetricKeys).toEqual([
        'system/node_0/cpu_utilization_percentage',
        'system/node_1/cpu_utilization_percentage',
      ]);
    });

    test('creates GPU system metrics section and charts when common GPU metrics are provided', () => {
      const runsData = [
        createMockRunData({
          'system/node_0/gpu_0_utilization_percentage': {
            key: 'system/node_0/gpu_0_utilization_percentage',
            value: 70,
          },
          'system/node_1/gpu_0_utilization_percentage': {
            key: 'system/node_1/gpu_0_utilization_percentage',
            value: 80,
          },
        }),
      ];

      const nodeLevelMetricsConfig = {
        nodeIndexes: ['0', '1'],
        commonMetrics: [],
        gpuIndexes: [0],
        commonGpuMetrics: ['utilization_percentage'],
        enabled: true as const,
      };

      const { resultChartSet, resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
        runsData,
        nodeLevelMetricsConfig,
      });

      // Verify GPU system metrics section was created
      const gpuSectionExists = resultSectionSet.some((section) => section.name === 'GPU system metrics');
      expect(gpuSectionExists).toBe(true);

      // Verify chart was created for the common GPU metric
      const gpuChart = resultChartSet.find(
        (chart) =>
          chart instanceof RunsChartsLineCardConfig &&
          chart.nodeLevelSystemMetricConfiguration?.metric === 'utilization_percentage' &&
          chart.nodeLevelSystemMetricConfiguration?.type === 'gpu',
      ) as RunsChartsLineCardConfig;

      expect(gpuChart).toBeDefined();
      expect(gpuChart.displayName).toBe('utilization_percentage');
      expect(gpuChart.selectedMetricKeys).toEqual([
        'system/node_0/gpu_0_utilization_percentage',
        'system/node_1/gpu_0_utilization_percentage',
      ]);
    });

    test('creates charts for multiple GPU indexes across multiple nodes', () => {
      const runsData = [
        createMockRunData({
          'system/node_0/gpu_0_power_usage_watts': { key: 'system/node_0/gpu_0_power_usage_watts', value: 100 },
          'system/node_0/gpu_1_power_usage_watts': { key: 'system/node_0/gpu_1_power_usage_watts', value: 110 },
          'system/node_1/gpu_0_power_usage_watts': { key: 'system/node_1/gpu_0_power_usage_watts', value: 120 },
          'system/node_1/gpu_1_power_usage_watts': { key: 'system/node_1/gpu_1_power_usage_watts', value: 130 },
        }),
      ];

      const nodeLevelMetricsConfig = {
        nodeIndexes: ['0', '1'],
        commonMetrics: [],
        gpuIndexes: [0, 1],
        commonGpuMetrics: ['power_usage_watts'],
        enabled: true as const,
      };

      const { resultChartSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
        runsData,
        nodeLevelMetricsConfig,
      });

      const powerChart = resultChartSet.find(
        (chart) =>
          chart instanceof RunsChartsLineCardConfig &&
          chart.nodeLevelSystemMetricConfiguration?.metric === 'power_usage_watts',
      ) as RunsChartsLineCardConfig;

      expect(powerChart).toBeDefined();
      expect(powerChart.selectedMetricKeys).toEqual([
        'system/node_0/gpu_0_power_usage_watts',
        'system/node_0/gpu_1_power_usage_watts',
        'system/node_1/gpu_0_power_usage_watts',
        'system/node_1/gpu_1_power_usage_watts',
      ]);
    });

    test('does not create node-level sections when nodeLevelMetricsConfig is disabled', () => {
      const runsData = [
        createMockRunData({
          'system/node_0/cpu_utilization_percentage': { key: 'system/node_0/cpu_utilization_percentage', value: 50 },
        }),
      ];

      const nodeLevelMetricsConfig = {
        nodeIndexes: [],
        commonMetrics: [],
        gpuIndexes: [],
        commonGpuMetrics: [],
        enabled: false,
      };

      const { resultChartSet, resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
        runsData,
        nodeLevelMetricsConfig,
      });

      // Verify Node/GPU system metrics sections were not created
      const nodeSectionExists = resultSectionSet.some((section) => section.name === 'Node system metrics');
      const gpuSectionExists = resultSectionSet.some((section) => section.name === 'GPU system metrics');

      expect(nodeSectionExists).toBe(false);
      expect(gpuSectionExists).toBe(false);

      // Verify no node-level charts were created
      const nodeLevelCharts = resultChartSet.filter(
        (chart) => chart instanceof RunsChartsLineCardConfig && chart.nodeLevelSystemMetricConfiguration,
      );
      expect(nodeLevelCharts).toHaveLength(0);
    });

    test('does not create sections when common metrics arrays are empty', () => {
      const runsData = [createMockRunData({})];

      const nodeLevelMetricsConfig = {
        nodeIndexes: ['0'],
        commonMetrics: [],
        gpuIndexes: [],
        commonGpuMetrics: [],
        enabled: true as const,
      };

      const { resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
        runsData,
        nodeLevelMetricsConfig,
      });

      const nodeSectionExists = resultSectionSet.some((section) => section.name === 'Node system metrics');
      const gpuSectionExists = resultSectionSet.some((section) => section.name === 'GPU system metrics');

      expect(nodeSectionExists).toBe(false);
      expect(gpuSectionExists).toBe(false);
    });
  });
});
