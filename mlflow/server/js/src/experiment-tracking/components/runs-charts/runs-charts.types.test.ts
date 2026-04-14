import { RunsChartsCardConfig, RunsChartType } from './runs-charts.types';
import type { RunsChartsRunData } from './components/RunsCharts.common';

/**
 * Stress tests for chart config generation.
 *
 * Verifies that ALL logged metrics produce chart configs (no artificial cap)
 * and that config generation stays fast even at 3000 metrics.
 */

function createMockRunData(metricCount: number, runId = 'run-1'): RunsChartsRunData {
  const metrics: Record<string, { key: string; value: number; step: number }> = {};
  for (let i = 0; i < metricCount; i++) {
    const section = ['train', 'eval', 'reward', 'loss', 'env', 'policy'][i % 6];
    const key = `${section}/layer_${Math.floor(i / 6)}/metric_${i}`;
    metrics[key] = { key, value: Math.random(), step: i > 50 ? 10 : 0 };
  }
  return {
    uuid: runId,
    displayName: runId,
    runInfo: { runUuid: runId, experimentId: '1', runName: runId } as any,
    metrics,
    params: {},
    tags: {},
    images: {},
    datasets: [],
  };
}

describe('RunsChartsCardConfig.getBaseChartAndSectionConfigs', () => {
  test('generates chart configs for ALL metrics — no 100-metric cap', () => {
    // ARRANGE
    const runsData = [createMockRunData(250)];

    // ACT
    const { resultChartSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
      runsData,
    });

    // ASSERT — every metric gets a chart, not just the first 100
    const metricCharts = resultChartSet.filter(
      (c) => c.type === RunsChartType.LINE || c.type === RunsChartType.BAR,
    );
    expect(metricCharts.length).toBe(250);
  });

  test('generates chart configs for 3000 metrics without cap', () => {
    // ARRANGE
    const runsData = [createMockRunData(3000)];

    // ACT
    const { resultChartSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
      runsData,
    });

    // ASSERT
    const metricCharts = resultChartSet.filter(
      (c) => c.type === RunsChartType.LINE || c.type === RunsChartType.BAR,
    );
    expect(metricCharts.length).toBe(3000);
  });

  test('3000 metrics: config generation completes in under 2 seconds', () => {
    // ARRANGE
    const runsData = [createMockRunData(3000)];

    // ACT
    const start = performance.now();
    RunsChartsCardConfig.getBaseChartAndSectionConfigs({ runsData });
    const elapsed = performance.now() - start;

    // ASSERT — config generation (lightweight JS objects) should be fast
    expect(elapsed).toBeLessThan(2000);
  });

  test('auto-collapses sections when chart count exceeds 100', () => {
    // ARRANGE
    const runsData = [createMockRunData(200)];

    // ACT
    const { resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
      runsData,
    });

    // ASSERT — all sections should be collapsed by default
    resultSectionSet.forEach((section) => {
      expect(section.display).toBe(false);
    });
  });

  test('sections are expanded when chart count is small', () => {
    // ARRANGE
    const runsData = [createMockRunData(50)];

    // ACT
    const { resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
      runsData,
    });

    // ASSERT — sections should be open
    resultSectionSet.forEach((section) => {
      expect(section.display).toBe(true);
    });
  });

  test('creates section groupings for 3000 metrics — all collapsed', () => {
    // ARRANGE
    const runsData = [createMockRunData(3000)];

    // ACT
    const { resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
      runsData,
    });

    // ASSERT — sections are created and all collapsed (>100 charts triggers auto-collapse)
    expect(resultSectionSet.length).toBeGreaterThan(0);
    resultSectionSet.forEach((section) => {
      expect(section.display).toBe(false);
    });
  });

  test('multiple runs with overlapping metrics — no duplicates', () => {
    // ARRANGE — 2 runs with same 500 metric keys
    const runsData = [createMockRunData(500, 'run-1'), createMockRunData(500, 'run-2')];

    // ACT
    const { resultChartSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
      runsData,
    });

    // ASSERT — should still be 500 charts, not 1000
    const metricCharts = resultChartSet.filter(
      (c) => c.type === RunsChartType.LINE || c.type === RunsChartType.BAR,
    );
    expect(metricCharts.length).toBe(500);
  });
});
