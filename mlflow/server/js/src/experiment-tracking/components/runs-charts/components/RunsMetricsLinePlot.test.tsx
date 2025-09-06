import type { RunsMetricsLinePlotProps } from './RunsMetricsLinePlot';
import { RunsMetricsLinePlot } from './RunsMetricsLinePlot';
import { renderWithIntl, cleanup } from '../../../../common/utils/TestUtils.react18';
import { LazyPlot } from '../../LazyPlot';
import type { PlotParams } from 'react-plotly.js';
import { RunsChartsLineChartXAxisType } from './RunsCharts.common';

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
});
