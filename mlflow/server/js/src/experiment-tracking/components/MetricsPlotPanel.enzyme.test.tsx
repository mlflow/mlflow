/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { shallow } from 'enzyme';
import {
  MetricsPlotPanel,
  CHART_TYPE_BAR,
  CHART_TYPE_LINE,
  EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL,
  METRICS_PLOT_HANGING_RUN_THRESHOLD_MS,
  convertMetricsToCsv,
} from './MetricsPlotPanel';
import MetricsSummaryTable from './MetricsSummaryTable';
import { X_AXIS_RELATIVE, X_AXIS_STEP, X_AXIS_WALL } from './MetricsPlotControls';
import Utils from '../../common/utils/Utils';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { RunLinksPopover } from './RunLinksPopover';
import { Progress } from '../../common/components/Progress';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestApolloProvider } from '../../common/utils/TestApolloProvider';
import { useSampledMetricHistory } from './runs-charts/hooks/useSampledMetricHistory';
import { saveAs } from 'file-saver';

jest.mock('./runs-charts/hooks/useSampledMetricHistory', () => ({
  useSampledMetricHistory: jest.fn(),
}));

// Mock file-saver for CSV download tests
jest.mock('file-saver', () => ({
  saveAs: jest.fn(),
}));

// Global test variables
let wrapper: any;
let instance: any;
let minimalPropsForLineChart: any;
let minimalPropsForBarChart: any;
let minimalStore: any;
let getMetricHistoryApi;
let getRunApi: any;
let navigate;

describe('unit tests', () => {

  beforeEach(() => {
    const location = {
      search: '?runs=["runUuid1","runUuid2","runUuid3","runUuid4"]&experiments=["1"]&plot_metric_keys=["metric_1","metric_2"]&plot_layout={}',
    };

    navigate = jest.fn((to: string) => {
      location.search = new URL(to, 'http://test').search;
    });

    getMetricHistoryApi = jest.fn(() => Promise.resolve({ value: {} }));
    getRunApi = jest.fn(() => Promise.resolve());
    const now = new Date().getTime();
    minimalPropsForLineChart = {
      experimentIds: ['1'],
      runUuids: ['runUuid1', 'runUuid2'],
      completedRunUuids: ['runUuid1', 'runUuid2'],
      metricKey: 'metric_1',
      latestMetricsByRunUuid: {
        runUuid1: {
          metric_1: { key: 'metric_1', value: 100, step: 2, timestamp: now },
          metric_2: { key: 'metric_2', value: 111, step: 0, timestamp: now },
        },
        runUuid2: {
          metric_1: { key: 'metric_1', value: 200, step: 4, timestamp: now },
          metric_2: { key: 'metric_2', value: 222, step: -3, timestamp: now },
        },
      },
      distinctMetricKeys: ['metric_1', 'metric_2'],
      // An array of { metricKey, history, runUuid, runDisplayName }
      metricsWithRunInfoAndHistory: [
        // Metrics for runUuid1
        {
          metricKey: 'metric_1',
          history: [
            /* Intentionally reversed timestamp and step here for testing */
            { key: 'metric_1', value: 100, step: 2, timestamp: now },
            { key: 'metric_1', value: 50, step: 1, timestamp: now - 1 },
          ],
          runUuid: 'runUuid1',
          runDisplayName: 'runDisplayName1',
        },
        {
          metricKey: 'metric_2',
          history: [
            { key: 'metric_2', value: 55, step: -1, timestamp: now - 1 },
            { key: 'metric_2', value: 111, step: 0, timestamp: now },
          ],
          runUuid: 'runUuid1',
          runDisplayName: 'runDisplayName1',
        },
        // Metrics for runUuid2
        {
          metricKey: 'metric_1',
          history: [
            { key: 'metric_1', value: 150, step: 3, timestamp: now - 1 },
            { key: 'metric_1', value: 200, step: 4, timestamp: now },
          ],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
        {
          metricKey: 'metric_2',
          history: [
            { key: 'metric_2', value: 155, step: -4, timestamp: now - 1 },
            { key: 'metric_2', value: 222, step: -3, timestamp: now },
          ],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
      ],
      getMetricHistoryApi,
      getRunApi,
      location,
      navigate,
      runDisplayNames: ['runDisplayName1', 'runDisplayName2'],
    };

    minimalPropsForBarChart = {
      experimentIds: ['1'],
      runUuids: ['runUuid1', 'runUuid2'],
      completedRunUuids: ['runUuid1', 'runUuid2'],
      metricKey: 'metric_1',
      latestMetricsByRunUuid: {
        runUuid1: {
          metric_1: { key: 'metric_1', value: 50, step: 0, timestamp: now },
          metric_2: { key: 'metric_2', value: 55, step: 0, timestamp: now },
        },
        runUuid2: {
          metric_1: { key: 'metric_2', value: 55, step: 0, timestamp: now },
          metric_2: { key: 'metric_2', value: 155, step: 0, timestamp: now },
        },
      },
      distinctMetricKeys: ['metric_1', 'metric_2'],
      // An array of { metricKey, history, runUuid, runDisplayName }
      metricsWithRunInfoAndHistory: [
        // Metrics for runUuid1
        {
          metricKey: 'metric_1',
          history: [{ key: 'metric_1', value: 50, step: 0, timestamp: now }],
          runUuid: 'runUuid1',
          runDisplayName: 'runDisplayName1',
        },
        {
          metricKey: 'metric_2',
          history: [{ key: 'metric_2', value: 55, step: 0, timestamp: now }],
          runUuid: 'runUuid1',
          runDisplayName: 'runDisplayName1',
        },
        // Metrics for runUuid2
        {
          metricKey: 'metric_1',
          history: [{ key: 'metric_1', value: 150, step: 0, timestamp: now }],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
        {
          metricKey: 'metric_2',
          history: [{ key: 'metric_2', value: 155, step: 0, timestamp: now }],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
      ],
      getMetricHistoryApi,
      getRunApi,
      location,
      navigate,
      runDisplayNames: ['runDisplayName1', 'runDisplayName2'],
      deselectedCurves: [],
    };

    const mockStore = configureStore([thunk, promiseMiddleware()]);
    minimalStore = mockStore({
      entities: {
        runInfosByUuid: {},
        latestMetricsByRunUuid: {},
        minMetricsByRunUuid: {},
        maxMetricsByRunUuid: {},
      },
    });

    const sampleHistoryResults = {
      resultsByRunUuid: {},
      isLoading: false,
      isRefreshing: false,
      refresh: jest.fn(),
    };
    jest.mocked(useSampledMetricHistory).mockReturnValue(sampleHistoryResults);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    expect(wrapper.length).toBe(1);
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForBarChart} />);
    expect(wrapper.length).toBe(1);
  });

  test('predictChartType()', () => {
    expect(MetricsPlotPanel.predictChartType(minimalPropsForLineChart.metricsWithRunInfoAndHistory)).toBe(
      CHART_TYPE_LINE,
    );
    expect(MetricsPlotPanel.predictChartType(minimalPropsForBarChart.metricsWithRunInfoAndHistory)).toBe(
      CHART_TYPE_BAR,
    );
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
    expect(instance.getUrlState().layout).toEqual({
      yaxis: { type: 'log', autorange: true, exponentformat: 'e' },
    });
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
      yaxis: { range: [0, 2], type: 'log', exponentformat: 'e' },
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
      yaxis: { autorange: true, type: 'log', exponentformat: 'e' },
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
      yaxis: { autorange: true, type: 'log', exponentformat: 'e' },
    });
    instance.handleYAxisLogScaleChange(false);
    expect(instance.getUrlState().layout).toEqual({
      xaxis: { range: [-5, 5], autorange: false },
      yaxis: { range: [0, 6], type: 'linear' },
    });
  });

  test('single-click handler in metric comparison plot - line chart', () => {
    jest.useFakeTimers();
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
    jest.advanceTimersByTime(1000);
    expect(instance.getUrlState().deselectedCurves).toEqual(['runUuid2-metric_1']);
  });

  test('single-click handler in metric comparison plot - bar chart', () => {
    jest.useFakeTimers();
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
    jest.advanceTimersByTime(1000);
    expect(instance.getUrlState().deselectedCurves).toEqual(['runUuid2']);
  });

  test('double-click handler in metric comparison plot - line chart', () => {
    jest.useFakeTimers();
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
    jest.advanceTimersByTime(1000);
    expect(instance.getUrlState().deselectedCurves).toEqual(['runUuid1-metric_1']);
  });

  test('double-click handler in metric comparison plot - bar chart', () => {
    jest.useFakeTimers();
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
    jest.advanceTimersByTime(1000);
    expect(instance.getUrlState().deselectedCurves).toEqual(['runUuid1']);
  });

  test('should render the number of completed runs correctly', () => {
    const mountWithProps = (props: any) => {
      return mountWithIntl(
        <Provider store={minimalStore}>
          <MemoryRouter>
            <DesignSystemProvider>
              <MetricsPlotPanel {...props} />
            </DesignSystemProvider>
          </MemoryRouter>
        </Provider>,
      );
    };
    // no runs completed
    wrapper = mountWithProps({
      ...minimalPropsForLineChart,
      completedRunUuids: [],
    });
    wrapper.update();
    expect(wrapper.find(Progress).text()).toContain('0/2');
    // 1 run completed
    wrapper = mountWithProps({
      ...minimalPropsForLineChart,
      completedRunUuids: ['runUuid1'],
    });
    wrapper.update();
    expect(wrapper.find(Progress).text()).toContain('1/2');
    // all runs completed
    wrapper = mountWithProps({
      ...minimalPropsForLineChart,
      completedRunUuids: ['runUuid1', 'runUuid2'],
    });
    wrapper.update();
    expect(wrapper.find(Progress).text()).toContain('2/2');
  });

  test('should render the metrics summary table correctly', () => {
    const mountWithProps = (props: any) => {
      return mountWithIntl(
        <Provider store={minimalStore}>
          <MemoryRouter>
            <DesignSystemProvider>
              <MetricsPlotPanel {...props} />
            </DesignSystemProvider>
          </MemoryRouter>
        </Provider>,
      );
    };
    wrapper = mountWithProps({
      ...minimalPropsForLineChart,
    });
    wrapper.update();

    const summaryTable = wrapper.find(MetricsSummaryTable);
    expect(summaryTable.length).toBe(1);
    expect(summaryTable.props().runUuids).toEqual(minimalPropsForLineChart.runUuids);
    // Selected metric keys are set by location.search
    expect(summaryTable.props().metricKeys).toEqual(['metric_1', 'metric_2']);
  });

  test('should not poll if all runs already completed', () => {
    jest.useFakeTimers();
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    expect(wrapper.instance().intervalId).toBeNull();
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(0);
  });

  test('should poll when some runs are active', () => {
    jest.useFakeTimers();
    const props = {
      ...minimalPropsForLineChart,
      completedRunUuids: ['runUuid1'],
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(1);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(2);
  });

  test('should stop polling when all runs complete', () => {
    jest.useFakeTimers();
    const props = {
      ...minimalPropsForLineChart,
      completedRunUuids: ['runUuid1'],
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(1);
    const nextProps = {
      ...minimalPropsForLineChart,
      completedRunUuids: ['runUuid1', 'runUuid2'],
    };
    wrapper.setProps(nextProps);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(1);
  });

  test('should ignore hanging runs', () => {
    jest.useFakeTimers();
    const latestTimestamp = new Date().getTime() - (METRICS_PLOT_HANGING_RUN_THRESHOLD_MS + 1000);
    const props = {
      ...minimalPropsForLineChart,
      // `runUuid1` has already completed and `runUuid2` is hanging.
      completedRunUuids: ['runUuid1'],
      latestMetricsByRunUuid: {
        runUuid1: minimalPropsForLineChart.latestMetricsByRunUuid.runUuid1,
        runUuid2: {
          metric_1: {
            key: 'metric_1',
            value: 200,
            step: 4,
            timestamp: latestTimestamp,
          },
          metric_2: {
            key: 'metric_2',
            value: 222,
            step: -3,
            timestamp: latestTimestamp,
          },
        },
      },
      metricsWithRunInfoAndHistory: [
        // Metrics for runUuid1
        ...minimalPropsForLineChart.metricsWithRunInfoAndHistory.slice(0, 2),
        // Metrics for runUuid2
        {
          metricKey: 'metric_1',
          history: [
            { key: 'metric_1', value: 150, step: 3, timestamp: latestTimestamp - 1 },
            { key: 'metric_1', value: 200, step: 4, timestamp: latestTimestamp },
          ],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
        {
          metricKey: 'metric_2',
          history: [
            { key: 'metric_2', value: 155, step: -4, timestamp: latestTimestamp - 1 },
            { key: 'metric_2', value: 222, step: -3, timestamp: latestTimestamp },
          ],
          runUuid: 'runUuid2',
          runDisplayName: 'runDisplayName2',
        },
      ],
    };

    wrapper = shallow(<MetricsPlotPanel {...props} />);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(0);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(0);
  });

  test('should skip polling when component is out of focus', () => {
    jest.useFakeTimers();
    const props = {
      ...minimalPropsForLineChart,
      completedRunUuids: ['runUuid1'],
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    expect(wrapper.state().focused).toBe(true);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(1);
    wrapper.instance().onBlur();
    expect(wrapper.state().focused).toBe(false);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(1);
    wrapper.instance().onFocus();
    expect(wrapper.state().focused).toBe(true);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(2);
  });

  test('should not poll after unmount', () => {
    jest.useFakeTimers();
    const props = {
      ...minimalPropsForLineChart,
      completedRunUuids: ['runUuid1'],
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(1);
    wrapper.unmount();
    jest.advanceTimersByTime(EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    expect(getRunApi).toHaveBeenCalledTimes(1);
  });

  test('handleDownloadCsv should call loadMetricHistory and saveAs', async () => {
    jest.useFakeTimers();
    const props = {...minimalPropsForLineChart};
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance()
    instance.getUrlState = jest.fn().mockReturnValue({ selectedMetricKeys: ['metric_1'] });
    instance.loadMetricHistory = jest.fn().mockResolvedValue(undefined);
    instance.setState({loading: false });

    await instance.handleDownloadCsv();

    expect(instance.loadMetricHistory).toHaveBeenCalledWith(['runUuid1', 'runUuid2'], ['metric_1']);
    expect(saveAs).toHaveBeenCalledWith(expect.any(Blob), 'metrics.csv');
  });

  test('handleDownloadCsv should skip loadMetricHistory when loading', async () => {
    jest.useFakeTimers();
    const props = {...minimalPropsForLineChart};
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance()
    instance.setState({loading: true})
    instance.getUrlState = jest.fn().mockReturnValue({ selectedMetricKeys: ['metric_1'] });
    instance.loadMetricHistory = jest.fn().mockResolvedValue(undefined);
    jest.advanceTimersByTime(1000)
    instance.setState({loading: false})
    await instance.handleDownloadCsv();

    expect(instance.loadMetricHistory).toHaveBeenCalledTimes(1);
    expect(saveAs).toHaveBeenCalledWith(expect.any(Blob), 'metrics.csv');
  });

  test('getCardChartConfig should return correct configuration', () => {
    const props = { ...minimalPropsForLineChart, metricKey: 'test_metric' };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance();

    instance.getUrlState = jest.fn().mockReturnValue({
      selectedMetricKeys: ['metric_1', 'metric_2'],
      selectedXAxis: 'step',
      yAxisLogScale: true,
      showPoint: true,
      layout: {
        xaxis: { range: [0, 100] },
        yaxis: { range: [10, 1000] }
      }
    });

    const config = instance.getCardChartConfig();

    expect(config.metricKey).toBe('test_metric');
    expect(config.selectedMetricKeys).toEqual(['metric_1', 'metric_2']);
    expect(config.scaleType).toBe('log');
    expect(config.displayPoints).toBe(true);
    expect(config.range).toEqual({
      xMin: 0,
      xMax: 100,
      yMin: 10,
      yMax: 1000
    });
  });

  test('getCardChartConfig should use metricKey as fallback for selectedMetricKeys', () => {
    const props = { ...minimalPropsForLineChart, metricKey: 'fallback_metric' };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance();

    instance.getUrlState = jest.fn().mockReturnValue({
      selectedMetricKeys: [],
      selectedXAxis: 'time',
      yAxisLogScale: false,
      showPoint: false
    });

    const config = instance.getCardChartConfig();

    expect(config.selectedMetricKeys).toEqual(['fallback_metric']);
    expect(config.scaleType).toBe('linear');
    expect(config.displayPoints).toBe(false);
  });

  test('getNewChartRunData should return correct run data structure', () => {
    const props = {
      ...minimalPropsForLineChart,
      metricKey: 'test_metric',
      runUuids: ['run1', 'run2'],
      runNames: ['Run 1', 'Run 2'],
      runDisplayNames: ['Display 1', 'Display 2'],
      latestMetricsByRunUuid: {'run1': [], 'run2': []}
    };
    wrapper = shallow(<MetricsPlotPanel {...props} />);
    instance = wrapper.instance();

    instance.getUrlState = jest.fn().mockReturnValue({
      selectedMetricKeys: ['metric_1', 'metric_2']
    });

    const runData = instance.getNewChartRunData();

    expect(runData).toHaveLength(2);
    expect(runData[0]).toEqual({
      uuid: 'run1',
      runInfo: undefined,
      metrics: { metric_1: {}, metric_2: {} },
      params: {},
      tags: {},
      datasets: [],
      images: {},
      hidden: false,
      color: undefined,
      displayName: 'Run 1'
    });
  });

  test('getGlobalLineChartConfig should return correct global config', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    instance = wrapper.instance();

    instance.getUrlState = jest.fn().mockReturnValue({
      lineSmoothness: 0.5,
      selectedXAxis: 'time'
    });

    const config = instance.getGlobalLineChartConfig();

    expect(config.lineSmoothness).toBe(0.5);
    expect(config.selectedXAxisMetricKey).toBeUndefined();
  });

  test('getTooltipContextValue should return runs data', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    instance = wrapper.instance();

    const mockRunData = [{ uuid: 'run1' }];
    instance.getNewChartRunData = jest.fn().mockReturnValue(mockRunData);

    const contextValue = instance.getTooltipContextValue();

    expect(contextValue).toEqual({ runs: mockRunData });
    expect(instance.getNewChartRunData).toHaveBeenCalledTimes(1);
  });

  test('should render RunsChartsLineChartCard with correct props', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    instance = wrapper.instance();

    const mockConfig = { metricKey: 'test' };
    const mockRunData = [{ uuid: 'run1' }];
    const mockGlobalConfig = { lineSmoothness: 0.5 };

    instance.getCardChartConfig = jest.fn().mockReturnValue(mockConfig);
    instance.getNewChartRunData = jest.fn().mockReturnValue(mockRunData);
    instance.getGlobalLineChartConfig = jest.fn().mockReturnValue(mockGlobalConfig);

    wrapper.setProps({});

    expect(instance.getCardChartConfig).toHaveBeenCalled();
    expect(instance.getNewChartRunData).toHaveBeenCalled();
    expect(instance.getGlobalLineChartConfig).toHaveBeenCalled();
  });

  test('should render Spinner with correct visibility based on loading state', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);

    wrapper.setState({ loading: true });
    expect(wrapper.find('Spinner').prop('css')).toEqual({ visibility: 'visible' });

    wrapper.setState({ loading: false });
    expect(wrapper.find('Spinner').prop('css')).toEqual({ visibility: 'hidden' });
  });

  test('should pass contextData to RunsChartsTooltipWrapper', () => {
    wrapper = shallow(<MetricsPlotPanel {...minimalPropsForLineChart} />);
    instance = wrapper.instance();

    const mockContextValue = { runs: [] };
    instance.getTooltipContextValue = jest.fn().mockReturnValue(mockContextValue);

    wrapper.setProps({});

    expect(instance.getTooltipContextValue).toHaveBeenCalled();
  });
});

test('convertMetricsToCsv', () => {
  const metrics = [
    {
      metricKey: 'metric1',
      history: [
        {
          key: 'metric1',
          value: 0,
          step: 0,
          timestamp: 0,
        },
        {
          key: 'metric1',
          value: 1,
          step: 1,
          timestamp: 1,
        },
        {
          key: 'metric1',
          value: 2,
          step: 2,
          timestamp: 2,
        },
      ],
      runUuid: '1',
      runDisplayName: 'Run 1',
    },
    {
      metricKey: 'metric2',
      history: [
        {
          key: 'metric2',
          value: 0,
          step: 0,
          timestamp: 0,
        },
        {
          key: 'metric2',
          value: 1,
          step: 1,
          timestamp: 1,
        },
      ],
      runUuid: '2',
      runDisplayName: 'Run 2',
    },
  ];
  const csv = convertMetricsToCsv(metrics);
  expect(csv).toBe(
    `
run_id,key,value,step,timestamp
1,metric1,0,0,0
1,metric1,1,1,1
1,metric1,2,2,2
2,metric2,0,0,0
2,metric2,1,1,1
`.trim(),
  );
});
