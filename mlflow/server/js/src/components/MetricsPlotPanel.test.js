import React from 'react';
import { shallow } from 'enzyme';
import { MetricsPlotPanel, CHART_TYPE_BAR, CHART_TYPE_LINE } from './MetricsPlotPanel';

describe('unit tests', () => {
  let wrapper;
  let instance;
  let minimalPropsForLineChart;
  let minimalPropsForBarChart;

  beforeEach(() => {
    minimalPropsForLineChart = {
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
            /* Intentionally reversed timestamp here */
            { key: 'metric_1', value: 100, step: 1, timestamp: 1556662044000 },
            { key: 'metric_1', value: 50, step: 2, timestamp: 1556662043000 },
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
          metricKey: 'metric_2',
          history: [
            { key: 'metric_2', value: 150, step: 3, timestamp: 1556662043000 },
            { key: 'metric_2', value: 200, step: 4, timestamp: 1556662044000 },
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
      location: {
        search:
          '?runs=["runUuid1","runUuid2"]&experiment=0' +
          '&plot_metric_keys=["metric_1","metric_2"]',
      },
      history: { push: jest.fn() },
      runDisplayNames: ['runDisplayName1', 'runDisplayName2'],
    };

    minimalPropsForBarChart = {
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
          metricKey: 'metric_2',
          history: [{ key: 'metric_2', value: 150, step: 0, timestamp: 1556662043000 }],
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
      location: {
        search:
          '?runs=["runUuid1","runUuid2"]&experiment=0' +
          '&plot_metric_keys=["metric_1","metric_2"]',
      },
      history: { push: jest.fn() },
      runDisplayNames: ['runDisplayName1', 'runDisplayName2'],
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

  test('getPlotMetricKeysFromUrl()', () => {
    const url0 = '?runs=["runUuid1","runUuid2"]&plot_metric_keys=[]';
    const url1 = '?runs=["runUuid1","runUuid2"]&plot_metric_keys=["metric_1"]';
    const url2 = '?runs=["runUuid1","runUuid2"]&plot_metric_keys=["metric_1","metric_2"]';
    expect(MetricsPlotPanel.getPlotMetricKeysFromUrl(url0)).toEqual([]);
    expect(MetricsPlotPanel.getPlotMetricKeysFromUrl(url1)).toEqual(['metric_1']);
    expect(MetricsPlotPanel.getPlotMetricKeysFromUrl(url2)).toEqual(['metric_1', 'metric_2']);
  });

  test('getMetrics()', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    instance = wrapper.instance();
    // sort reversed history in mock
    const metrics = minimalPropsForLineChart.metricsWithRunInfoAndHistory;
    metrics[0].history.sort();
    expect(instance.getMetrics()).toEqual(metrics);
  });
});
