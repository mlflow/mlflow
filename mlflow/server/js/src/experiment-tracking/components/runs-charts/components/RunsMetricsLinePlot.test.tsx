import { jest, describe, test, expect } from '@jest/globals';
import type { RunsMetricsLinePlotProps } from './RunsMetricsLinePlot';
import { RunsMetricsLinePlot } from './RunsMetricsLinePlot';
import { renderWithIntl, cleanup } from '../../../../common/utils/TestUtils.react18';
import { LazyPlot } from '../../LazyPlot';
import type { PlotParams } from 'react-plotly.js';
import { RunsChartsLineChartXAxisType } from './RunsCharts.common';
import { RunGroupingAggregateFunction } from '../../experiment-page/utils/experimentPage.row-types';
import type { RunGroupParentInfo } from '../../experiment-page/utils/experimentPage.row-types';

jest.mock('../../LazyPlot', () => ({
  LazyPlot: jest.fn(() => null),
}));

describe('RunsMetricsLinePlot', () => {
  const defaultProps: RunsMetricsLinePlotProps = {
    metricKey: 'testMetric',
    xAxisKey: RunsChartsLineChartXAxisType.STEP,
    lineSmoothness: 0,
    runsData: [
      {
        displayName: 'Run 1',
        uuid: 'run-1',
        metricsHistory: {
          testMetric: [
            { value: 0, step: 0, key: 'testMetric', timestamp: 1100 },
            { value: 10, step: 1, key: 'testMetric', timestamp: 1200 },
            { value: 20, step: 2, key: 'testMetric', timestamp: 1300 },
            { value: 30, step: 3, key: 'testMetric', timestamp: 1400 },
          ],
        },
      },
    ],
    selectedXAxisMetricKey: '',
  };

  // Helper function to get the last rendered props of the LazyPlot component
  const getLastRenderedPlotProps = (): PlotParams => {
    const [props] = jest.mocked(LazyPlot).mock.lastCall ?? [];
    return props;
  };

  test('it should properly filter negative values when using log scale on X axis', () => {
    // Render with linear scale and expect all values to be present
    renderWithIntl(<RunsMetricsLinePlot {...defaultProps} xAxisScaleType="linear" />);
    expect(getLastRenderedPlotProps().data[0]).toEqual(expect.objectContaining({ x: [0, 1, 2, 3] }));
    expect(getLastRenderedPlotProps().data[0]).toEqual(expect.objectContaining({ y: [0, 10, 20, 30] }));
    cleanup();

    // Render with log scale and expect non-positive values to be filtered out
    renderWithIntl(<RunsMetricsLinePlot {...defaultProps} xAxisScaleType="log" />);
    expect(getLastRenderedPlotProps().data[0]).toEqual(expect.objectContaining({ x: [1, 2, 3] }));
    expect(getLastRenderedPlotProps().data[0]).toEqual(expect.objectContaining({ y: [10, 20, 30] }));
    cleanup();
  });

  test('it should fill the grouped run band without anchoring the Y axis to zero', () => {
    const createMetricHistory = (values: number[]) =>
      values.map((value, step) => ({ value, step, key: 'testMetric', timestamp: 1100 + step }));

    const groupedProps: RunsMetricsLinePlotProps = {
      ...defaultProps,
      runsData: [
        {
          displayName: 'Group 1',
          uuid: 'group-1',
          groupParentInfo: { groupId: 'group-1', runUuids: ['run-1', 'run-2'] } as RunGroupParentInfo,
          metricsHistory: {
            testMetric: createMetricHistory([1.05, 1.06, 1.07]),
          },
          aggregatedMetricsHistory: {
            testMetric: {
              [RunGroupingAggregateFunction.Min]: createMetricHistory([1.0, 1.01, 1.02]),
              [RunGroupingAggregateFunction.Max]: createMetricHistory([1.08, 1.09, 1.1]),
              [RunGroupingAggregateFunction.Average]: createMetricHistory([1.04, 1.05, 1.06]),
            },
          },
        },
      ],
    };

    renderWithIntl(<RunsMetricsLinePlot {...groupedProps} />);

    // The band trace is prepended to the chart data, see `plotDataWithBands`
    const bandTrace = getLastRenderedPlotProps().data[0];

    // `tozeroy` fills down to y=0, which forces Plotly's autorange to include zero and
    // squashes the actual data range. `toself` closes the min/max ring instead. The ring
    // must not be split by a null separator, otherwise `toself` fills each half on its own.
    expect(bandTrace).toEqual(
      expect.objectContaining({
        fill: 'toself',
        x: [2, 1, 0, 0, 1, 2],
        y: [1.02, 1.01, 1.0, 1.08, 1.09, 1.1],
      }),
    );

    cleanup();
  });
});
