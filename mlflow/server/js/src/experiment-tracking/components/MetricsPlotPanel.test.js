import React from 'react';
import { shallow } from 'enzyme';
import { MetricsPlotPanel, CHART_TYPE_BAR, CHART_TYPE_LINE } from './MetricsPlotPanel';
import { X_AXIS_RELATIVE, X_AXIS_STEP, X_AXIS_WALL } from './MetricsPlotControls';
import Utils from '../../common/utils/Utils';

describe('unit tests', () => {
  let wrapper;
  let instance;
  let minimalPropsForLineChart;
  let minimalPropsForBarChart;

  beforeEach(() => {
    const location = {
      search:
        '?runs=["runUuid1","runUuid2"]&experiment=0' +
        '&plot_metric_keys=["metric_1","metric_2"]&plot_layout={}',
    };
    const history = {
      replace: (url) => {
        location.search = '?' + url.split('?')[1];
      },
    };
    minimalPropsForLineChart = {
      experimentId: '1',
      runUuids: ['runUuid1', 'runUuid2'],
      metricKey: 'metric_1',
      latestMetricsByRunUuid: {
        runUuid1: { metric_1: 100, metric_2: 200 },
        runUuid2: { metric_1: 111, metric_2: 222 },
      },
      distinctMetricKeys: ['metric_1', 'metric_2'],
      // An array of { metricKey, history, runUuid, runDisplayName }
      metricsWithRunInfoAndHistory: [
        {
          metricKey: 'metric_1',
          history: [
            /* Intentionally reversed timestamp and step here for testing */
            { key: 'metric_1', value: 100, step: 2, timestamp: 1556662044000 },
            { key: 'metric_1', value: 50, step: 1, timestamp: 1556662043000 },
          ],
          runUuid: 'runUuid1',
          runDisplayName: 'runDisplayName1',
        },
        {
          metricKey: 'metric_2',
          history: [
            { key: 'metric_2', value: 55, step: -1, timestamp: 1556662043000 },
            { key: 'metric_2', value: 111, step: 0, timestamp: 1556662044000 },
          ],
          runUuid: 'runUuid1',
          runDisplayName: 'runDisplayName1',
        },
        {
          metricKey: 'metric_1',
          history: [
            { key: 'metric_1', value: 150, step: 3, timestamp: 1556662043000 },
            { key: 'metric_1', value: 200, step: 4, timestamp: 1556662044000 },
          ],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
        {
          metricKey: 'metric_2',
          history: [
            { key: 'metric_2', value: 155, step: -4, timestamp: 1556662043000 },
            { key: 'metric_2', value: 222, step: -3, timestamp: 1556662044000 },
          ],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
      ],
      getMetricHistoryApi: jest.fn(),
      location: location,
      history: history,
      runDisplayNames: ['runDisplayName1', 'runDisplayName2'],
    };

    minimalPropsForBarChart = {
      experimentId: '1',
      runUuids: ['runUuid1', 'runUuid2'],
      metricKey: 'metric_1',
      latestMetricsByRunUuid: {
        runUuid1: { metric_1: 100, metric_2: 200 },
        runUuid2: { metric_1: 111, metric_2: 222 },
      },
      distinctMetricKeys: ['metric_1', 'metric_2'],
      // An array of { metricKey, history, runUuid, runDisplayName }
      metricsWithRunInfoAndHistory: [
        {
          metricKey: 'metric_1',
          history: [{ key: 'metric_1', value: 50, step: 0, timestamp: 1556662043000 }],
          runUuid: 'runUuid1',
          runDisplayName: 'runDisplayName1',
        },
        {
          metricKey: 'metric_2',
          history: [{ key: 'metric_2', value: 55, step: 0, timestamp: 1556662043000 }],
          runUuid: 'runUuid1',
          runDisplayName: 'runDisplayName1',
        },
        {
          metricKey: 'metric_1',
          history: [{ key: 'metric_1', value: 150, step: 0, timestamp: 1556662043000 }],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
        {
          metricKey: 'metric_2',
          history: [{ key: 'metric_2', value: 155, step: 0, timestamp: 1556662043000 }],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
      ],
      getMetricHistoryApi: jest.fn(),
      location: location,
      history: history,
      runDisplayNames: ['runDisplayName1', 'runDisplayName2'],
      deselectedCurves: [],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    expect(wrapper.length).toBe(1);
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForBarChart} />);
    expect(wrapper.length).toBe(1);
  });

  test('predictChartType()', () => {
    expect(
      MetricsPlotPanel.predictChartType(minimalPropsForLineChart.metricsWithRunInfoAndHistory),
    ).toBe(CHART_TYPE_LINE);
    expect(
      MetricsPlotPanel.predictChartType(minimalPropsForBarChart.metricsWithRunInfoAndHistory),
    ).toBe(CHART_TYPE_BAR);
  });

  test('isComparing()', () => {
    const s1 = '?runs=["runUuid1","runUuid2"]&plot_metric_keys=["metric_1","metric_2"]';
    const s2 = '?runs=["runUuid1"]&plot_metric_keys=["metric_1","metric_2"]';
    expect(MetricsPlotPanel.isComparing(s1)).toBe(true);
    expect(MetricsPlotPanel.isComparing(s2)).toBe(false);
  });

  test('getMetrics() should sort the history by timestamp for `Time (Relative)` x-axis', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    instance = wrapper.instance();

    instance.setState({ selectedXAxis: X_AXIS_RELATIVE });
    const metrics = minimalPropsForLineChart.metricsWithRunInfoAndHistory;
    metrics[0].history.sort(); // sort in place before comparison
    expect(instance.getMetrics()).toEqual(metrics);
  });

  test('getMetrics() should sort the history by timestamp for `Time (Wall)` x-axis', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    instance = wrapper.instance();

    instance.setState({ selectedXAxis: X_AXIS_WALL });
    const metrics = minimalPropsForLineChart.metricsWithRunInfoAndHistory;
    metrics[0].history.sort(); // sort in place before comparison
    expect(instance.getMetrics()).toEqual(metrics);
  });

  test('getMetrics() should sort the history by step&timestamp for `Step` x-axis', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    instance = wrapper.instance();

    instance.setState({ selectedXAxis: X_AXIS_STEP });
    const metrics = minimalPropsForLineChart.metricsWithRunInfoAndHistory;
    metrics[0].history.sort(Utils.compareByStepAndTimestamp); // sort in place before comparison
    expect(instance.getMetrics()).toEqual(metrics);
  });

  test('handleYAxisLogScale properly converts layout between log and linear scales', () => {
    const props = {
      ...minimalPropsForLineChart,
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance();
    // Test converting to & from log scale for an empty layout (e.g. a layout without any
    // user-specified zoom)
    instance.handleYAxisLogScaleChange(true);
    expect(instance.getUrlState().layout).toEqual({ yaxis: { type: 'log', autorange: true } });
    instance.handleYAxisLogScaleChange(false);
    expect(instance.getUrlState().layout).toEqual({ yaxis: { type: 'linear', autorange: true } });
    // Test converting to & from log scale for a layout with specified y axis range bounds
    instance.handleLayoutChange({
      'xaxis.range[0]': 2,
      'xaxis.range[1]': 4,
      'yaxis.range[0]': 1,
      'yaxis.range[1]': 100,
    });
    instance.handleYAxisLogScaleChange(true);
    expect(instance.getUrlState().layout).toEqual({
      xaxis: { range: [2, 4], autorange: false },
      yaxis: { range: [0, 2], type: 'log' },
    });
    instance.handleYAxisLogScaleChange(false);
    expect(instance.getUrlState().layout).toEqual({
      xaxis: { range: [2, 4], autorange: false },
      yaxis: { range: [1, 100], type: 'linear' },
    });
    // Test converting to & from log scale for a layout with negative Y axis
    instance.handleLayoutChange({
      'xaxis.range[0]': -5,
      'xaxis.range[1]': 5,
      'yaxis.range[0]': -3,
      'yaxis.range[1]': 6,
    });
    instance.handleYAxisLogScaleChange(true);
    expect(instance.getUrlState().layout).toEqual({
      xaxis: { range: [-5, 5], autorange: false },
      yaxis: { autorange: true, type: 'log' },
    });
    instance.handleYAxisLogScaleChange(false);
    expect(instance.getUrlState().layout).toEqual({
      xaxis: { range: [-5, 5], autorange: false },
      yaxis: { range: [-3, 6], type: 'linear' },
    });
    // Test converting to & from log scale for a layout with zero-valued Y axis bound
    instance.handleLayoutChange({ 'yaxis.range[0]': 0, 'yaxis.range[1]': 6 });
    instance.handleYAxisLogScaleChange(true);
    expect(instance.getUrlState().layout).toEqual({
      xaxis: { range: [-5, 5], autorange: false },
      yaxis: { autorange: true, type: 'log' },
    });
    instance.handleYAxisLogScaleChange(false);
    expect(instance.getUrlState().layout).toEqual({
      xaxis: { range: [-5, 5], autorange: false },
      yaxis: { range: [0, 6], type: 'linear' },
    });
  });

  test('single-click handler in metric comparison plot - line chart', (done) => {
    const props = {
      ...minimalPropsForLineChart,
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance();
    // Verify that clicking doesn't immediately update the run state
    expect(instance.getUrlState().deselectedCurves).toEqual([]);
    instance.handleLegendClick({
      curveNumber: 0,
      data: [{ runId: 'runUuid2', metricName: 'metric_1' }],
    });
    expect(instance.getUrlState().deselectedCurves).toEqual([]);
    // Wait a second, verify first run was deselected
    window.setTimeout(() => {
      expect(instance.getUrlState().deselectedCurves).toEqual(['runUuid2-metric_1']);
      done();
    }, 1000);
  });

  test('single-click handler in metric comparison plot - bar chart', (done) => {
    const props = {
      ...minimalPropsForBarChart,
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance();
    // Verify that clicking doesn't immediately update the run state
    expect(instance.getUrlState().deselectedCurves).toEqual([]);
    instance.handleLegendClick({ curveNumber: 0, data: [{ runId: 'runUuid2', type: 'bar' }] });
    expect(instance.getUrlState().deselectedCurves).toEqual([]);
    // Wait a second, verify first run was deselected
    window.setTimeout(() => {
      expect(instance.getUrlState().deselectedCurves).toEqual(['runUuid2']);
      done();
    }, 1000);
  });

  test('double-click handler in metric comparison plot - line chart', (done) => {
    const props = {
      ...minimalPropsForLineChart,
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance();
    const data = [
      { runId: 'runUuid1', metricName: 'metric_1' },
      { runId: 'runUuid2', metricName: 'metric_2' },
    ];
    // Verify that clicking doesn't immediately update the run state
    expect(instance.getUrlState().deselectedCurves).toEqual([]);
    instance.handleLegendClick({ curveNumber: 1, data: data });
    expect(instance.getUrlState().deselectedCurves).toEqual([]);
    // Double-click, verify that only the clicked run is selected (that the other one is deselected)
    instance.handleLegendClick({ curveNumber: 1, data: data });
    window.setTimeout(() => {
      expect(instance.getUrlState().deselectedCurves).toEqual(['runUuid1-metric_1']);
      done();
    }, 1000);
  });

  test('double-click handler in metric comparison plot - bar chart', (done) => {
    const props = {
      ...minimalPropsForBarChart,
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance();
    const data = [
      { runId: 'runUuid1', type: 'bar' },
      { runId: 'runUuid2', type: 'bar' },
    ];
    // Verify that clicking doesn't immediately update the run state
    expect(instance.getUrlState().deselectedCurves).toEqual([]);
    instance.handleLegendClick({ curveNumber: 1, data: data });
    expect(instance.getUrlState().deselectedCurves).toEqual([]);
    // Double-click, verify that only the clicked run is selected (that the other one is deselected)
    instance.handleLegendClick({ curveNumber: 1, data: data });
    window.setTimeout(() => {
      expect(instance.getUrlState().deselectedCurves).toEqual(['runUuid1']);
      done();
    }, 1000);
  });
});
