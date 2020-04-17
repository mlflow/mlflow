import React from 'react';
import { shallow } from 'enzyme';
import { MetricsPlotView } from './MetricsPlotView';
import { X_AXIS_RELATIVE, X_AXIS_WALL } from './MetricsPlotControls';
import { CHART_TYPE_BAR, CHART_TYPE_LINE } from './MetricsPlotPanel';
import Utils from '../../common/utils/Utils';
import Plot from '../../../node_modules/react-plotly.js/react-plotly';

const metricsForLine = [
  {
    metricKey: 'metric_0',
    history: [
      {
        key: 'metric_0',
        value: 100,
        step: 0,
        timestamp: 1556662043000,
      },
      {
        key: 'metric_0',
        value: 200,
        step: 1,
        timestamp: 1556662044000,
      },
    ],
    runUuid: 'runUuid1',
    runDisplayName: 'RunDisplayName1',
  },
  {
    metricKey: 'metric_1',
    history: [
      {
        key: 'metric_1',
        value: 300,
        step: 0,
        timestamp: 1556662043000,
      },
      {
        key: 'metric_1',
        value: 400,
        step: 1,
        timestamp: 1556662044000,
      },
    ],
    runUuid: 'runUuid2',
    runDisplayName: 'RunDisplayName2',
  },
];

const metricsForBar = [
  {
    metricKey: 'metric_0',
    history: [
      {
        key: 'metric_0',
        value: 100,
        step: 0,
        timestamp: 1556662043000,
      },
    ],
    runUuid: 'runUuid1',
    runDisplayName: 'RunDisplayName1',
  },
  {
    metricKey: 'metric_0',
    history: [
      {
        key: 'metric_0',
        value: 300,
        step: 0,
        timestamp: 1556662043000,
      },
    ],
    runUuid: 'runUuid2',
    runDisplayName: 'RunDisplayName2',
  },
];

describe('unit tests', () => {
  let wrapper;
  let instance;
  let minimalPropsForLineChart;
  let minimalPropsForSmoothedLineChart;
  let minimalPropsForBarChart;

  beforeEach(() => {
    minimalPropsForLineChart = {
      runUuids: ['runUuid1', 'runUuid2'],
      runDisplayNames: ['RunDisplayName1', 'RunDisplayName2'],
      xAxis: X_AXIS_RELATIVE,
      metrics: metricsForLine,
      metricKeys: ['metric_0', 'metric_1'],
      showPoint: false,
      chartType: CHART_TYPE_LINE,
      isComparing: false,
      yAxisLogScale: false,
      lineSmoothness: 1,
      onClick: jest.fn(),
      onLayoutChange: jest.fn(),
      onLegendDoubleClick: jest.fn(),
      onLegendClick: jest.fn(),
      deselectedCurves: [],
    };
    minimalPropsForSmoothedLineChart = {
      ...minimalPropsForLineChart,
      lineSmoothness: 50,
    };
    minimalPropsForBarChart = {
      ...minimalPropsForLineChart,
      metrics: metricsForBar,
      metricKeys: ['metric_0'],
      chartType: CHART_TYPE_BAR,
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<MetricsPlotView {...minimalPropsForLineChart} />);
    expect(wrapper.length).toBe(1);
  });

  test('getPlotPropsForLineChart()', () => {
    wrapper = shallow(<MetricsPlotView {...minimalPropsForLineChart} />);
    instance = wrapper.instance();
    expect(instance.getPlotPropsForLineChart()).toEqual({
      data: [
        {
          metricName: 'metric_0',
          name: 'metric_0',
          runId: 'runUuid1',
          x: [0, 1],
          y: [100, 200],
          text: ['100.00000', '200.00000'],
          type: 'scattergl',
          visible: true,
          mode: 'lines+markers',
          hovertemplate: '%{y}',
          marker: {"opacity": 0},
        },
        {
          metricName: 'metric_1',
          name: 'metric_1',
          runId: 'runUuid2',
          x: [0, 1],
          y: [300, 400],
          text: ['300.00000', '400.00000'],
          type: 'scattergl',
          visible: true,
          mode: 'lines+markers',
          hovertemplate: '%{y}',
          marker: {"opacity": 0},
        },
      ],
      layout: {},
    });
  });

  test('getPlotPropsForLineChart(lineSmoothness = 50)', () => {
    wrapper = shallow(<MetricsPlotView {...minimalPropsForSmoothedLineChart} />);
    instance = wrapper.instance();
    expect(instance.getPlotPropsForLineChart()).toEqual({
      data: [
        {
          metricName: 'metric_0',
          name: 'metric_0',
          runId: 'runUuid1',
          x: [0, 1],
          y: [100, 166.88741721854302],
          text: ['100.00000', '200.00000'],
          type: 'scattergl',
          visible: true,
          mode: 'lines+markers',
          hovertemplate: 'Value: %{text}<br>Smoothed: %{y}',
          marker: {"opacity": 0},
        },
        {
          metricName: 'metric_1',
          name: 'metric_1',
          runId: 'runUuid2',
          x: [0, 1],
          y: [300, 366.887417218543],
          text: ['300.00000', '400.00000'],
          type: 'scattergl',
          visible: true,
          mode: 'lines+markers',
          hovertemplate: 'Value: %{text}<br>Smoothed: %{y}',
          marker: {"opacity": 0},
        },
      ],
      layout: {},
    });
  });

  test('getPlotPropsForBarChart()', () => {
    wrapper = shallow(<MetricsPlotView {...minimalPropsForBarChart} />);
    instance = wrapper.instance();
    expect(instance.getPlotPropsForBarChart()).toEqual({
      data: [
        {
          name: 'RunDisplayName1',
          x: ['metric_0'],
          y: [100],
          type: 'bar',
          runId: 'runUuid1',
        },
        {
          name: 'RunDisplayName2',
          x: ['metric_0'],
          y: [300],
          type: 'bar',
          runId: 'runUuid2',
        },
      ],
      layout: {
        barmode: 'group',
      },
    });
  });

  test('getLineLegend()', () => {
    // how both metric and run name when comparing multiple runs
    expect(MetricsPlotView.getLineLegend('metric_1', 'Run abc', true)).toBe('metric_1, Run abc');
    // only show metric name when there
    expect(MetricsPlotView.getLineLegend('metric_1', 'Run abc', false)).toBe('metric_1');
  });

  test('parseTimestamp()', () => {
    const timestamp = 1556662044000;
    const timestampStr = Utils.formatTimestamp(timestamp);
    const history = [{ timestamp: 1556662043000 }];
    // convert to step when axis is Time (Relative)
    expect(MetricsPlotView.parseTimestamp(timestamp, history, X_AXIS_RELATIVE)).toBe(1);
    // convert to date time string when axis is Time (Wall)
    expect(MetricsPlotView.parseTimestamp(timestamp, history, X_AXIS_WALL)).toBe(timestampStr);
  });

  test('should disable both plotly logo and the link to plotly studio', () => {
    wrapper = shallow(<MetricsPlotView {...minimalPropsForBarChart} />);
    const plot = wrapper.find(Plot);
    expect(plot.props().config.displaylogo).toBe(false);
    expect(plot.props().config.modeBarButtonsToRemove).toContain('sendDataToCloud');
  });
});
