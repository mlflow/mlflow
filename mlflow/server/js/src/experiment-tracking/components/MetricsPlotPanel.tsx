/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { type ReactNode } from 'react';
import { connect } from 'react-redux';
import Utils from '../../common/utils/Utils';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { getMetricHistoryApi, getRunApi } from '../actions';
import { cloneDeep, difference, every as lodashEvery, flatMap as lodashFlatMap, isNumber, negate } from 'lodash';
import { getRunInfo } from '../reducers/Reducers';
import { MetricsPlotControls, X_AXIS_WALL, X_AXIS_RELATIVE, X_AXIS_STEP } from './MetricsPlotControls';
import MetricsSummaryTable from './MetricsSummaryTable';
import qs from 'qs';
import { withRouterNext } from '../../common/utils/withRouterNext';
import Routes from '../routes';
import { getUUID } from '../../common/utils/ActionUtils';
import { saveAs } from 'file-saver';
import { Spinner } from '@databricks/design-system';
import {
  normalizeMetricsHistoryEntry,
  EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL,
} from '../utils/MetricsUtils';
import type { Location, NavigateFunction } from '../../common/utils/RoutingUtils';
import { RunsChartsCard } from './runs-charts/components/cards/RunsChartsCard';
import {
  RunsChartsCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsLineChartYAxisType,
  RunsChartType,
} from './runs-charts/runs-charts.types';
import { RunsChartsRunData, RunsChartsLineChartXAxisType } from './runs-charts/components/RunsCharts.common';
import { RunsChartsTooltipWrapper } from './runs-charts/hooks/useRunsChartsTooltip';
import { RunsChartsTooltipBody } from './runs-charts/components/RunsChartsTooltipBody';
import { RunsChartsFullScreenModal } from './runs-charts/components/RunsChartsFullScreenModal';
import { getStableColorForRun } from '../utils/RunNameUtils';

export const CHART_TYPE_LINE = 'line';
export const CHART_TYPE_BAR = 'bar';
// Full metrics polling rate is set to 2x sample metrics polling rate
export const EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL = EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL * 2;
// A run is considered as 'hanging' if its status is 'RUNNING' but its latest metric was logged
// prior to this threshold. The metrics plot doesn't automatically update hanging runs.
export const METRICS_PLOT_HANGING_RUN_THRESHOLD_MS = 3600 * 24 * 7 * 1000; // 1 week
const GET_METRIC_HISTORY_MAX_RESULTS = 25000;

// Convert X-axis type from URL to chart config
const convertXAxisType = (selectedXAxis: string | string[] | any): RunsChartsLineChartXAxisType => {
  // Handle union type from URL parsing
  const axisType = Array.isArray(selectedXAxis) ? selectedXAxis[0] : selectedXAxis || 'step';

  switch (axisType) {
    case 'step':
      return RunsChartsLineChartXAxisType.STEP;
    case 'wall':
      return RunsChartsLineChartXAxisType.TIME;
    case 'relative':
      return RunsChartsLineChartXAxisType.TIME_RELATIVE;
    default:
      return RunsChartsLineChartXAxisType.STEP;
  }
};

export const convertMetricsToCsv = (metrics: any) => {
  const header = ['run_id', ...Object.keys(metrics[0].history[0])];
  const rows = metrics.flatMap(({ runUuid, history }: any) =>
    history.map((metric: any) => [runUuid, ...Object.values(metric)]),
  );
  return [header]
    .concat(rows)
    .map((row) => row.join(','))
    .join('\n');
};

type OwnMetricsPlotPanelProps = {
  experimentIds: string[];
  runUuids: string[];
  completedRunUuids: string[];
  metricKey: string;
  latestMetricsByRunUuid: any;
  distinctMetricKeys: string[];
  metricsWithRunInfoAndHistory: any[];
  getMetricHistoryApi: (...args: any[]) => any;
  getRunApi: (...args: any[]) => any;
  location: Location;
  navigate: NavigateFunction;
  runDisplayNames: string[];
  containsInfinities: boolean;
  // Additional props for new chart system
  runNames?: string[];
  runInfos?: any[];
};

type MetricsPlotPanelState = any;

type MetricsPlotPanelProps = OwnMetricsPlotPanelProps & typeof MetricsPlotPanel.defaultProps;

export class MetricsPlotPanel extends React.Component<MetricsPlotPanelProps, MetricsPlotPanelState> {
  _isMounted = false;

  static defaultProps = {
    containsInfinities: false,
  };

  intervalId: any;
  loadMetricHistoryPromise: Promise<any> | null;
  xAxisType: string | null;

  constructor(props: MetricsPlotPanelProps) {
    super(props);
    this.state = {
      historyRequestIds: [],
      focused: true,
      loading: false,
      fullScreenChart: undefined as
        | {
            config: RunsChartsCardConfig;
            title: string | ReactNode;
            subtitle: ReactNode;
          }
        | undefined,
    };
    this.intervalId = null;
    this.loadMetricHistoryPromise = null;
    this.xAxisType = null;
  }

  onFocus = () => {
    this.setState({ focused: true });
  };

  onBlur = () => {
    this.setState({ focused: false });
  };

  clearEventListeners = () => {
    // `window.removeEventListener` does nothing when called with an unregistered event listener:
    // https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/removeEventListener
    window.removeEventListener('focus', this.onFocus);
    window.removeEventListener('blur', this.onBlur);
  };

  clearInterval = () => {
    // `clearInterval` does nothing when called with `null` or `undefine`:
    // https://www.w3.org/TR/2011/WD-html5-20110525/timers.html#dom-windowtimers-cleartimeout
    clearInterval(this.intervalId);
    this.intervalId = null;
  };

  allRunsCompleted = () => {
    return this.props.completedRunUuids.length === this.props.runUuids.length;
  };

  isHangingRunUuid = (activeRunUuid: any) => {
    const metrics = this.props.latestMetricsByRunUuid[activeRunUuid];
    if (!metrics) {
      return false;
    }
    // @ts-expect-error TS(2345): Argument of type '({ timestamp }: { timestamp: any... Remove this comment to see the full error message
    const timestamps = Object.values(metrics).map(({ timestamp }) => timestamp);
    // @ts-expect-error TS(2345): Argument of type 'unknown' is not assignable to pa... Remove this comment to see the full error message
    const latestTimestamp = Math.max(...timestamps);
    return new Date().getTime() - latestTimestamp > METRICS_PLOT_HANGING_RUN_THRESHOLD_MS;
  };

  getActiveRunUuids = () => {
    const { completedRunUuids, runUuids } = this.props;
    const activeRunUuids = difference(runUuids, completedRunUuids);
    return activeRunUuids.filter(negate(this.isHangingRunUuid)); // Exclude hanging runs
  };

  shouldPoll = () => {
    return !(this.allRunsCompleted() || this.getActiveRunUuids().length === 0);
  };

  componentDidMount() {
    this._isMounted = true;
    this.loadMetricHistory(this.props.runUuids, this.getUrlState().selectedMetricKeys);
    if (this.shouldPoll()) {
      // Set event listeners to detect when this component gains/loses focus,
      // e.g., a user switches to a different browser tab or app.
      window.addEventListener('blur', this.onBlur);
      window.addEventListener('focus', this.onFocus);
      this.intervalId = setInterval(() => {
        // Skip polling if this component is out of focus.
        // @ts-expect-error TS(4111): Property 'focused' comes from an index signature, ... Remove this comment to see the full error message
        if (this.state.focused) {
          const activeRunUuids = this.getActiveRunUuids();
          this.loadMetricHistory(activeRunUuids, this.getUrlState().selectedMetricKeys);
          this.loadRuns(activeRunUuids);

          if (!this.shouldPoll()) {
            this.clearEventListeners();
            this.clearInterval();
          }
        }
      }, EXPERIMENT_RUNS_FULL_METRICS_POLLING_INTERVAL);
    }
  }

  componentWillUnmount() {
    this._isMounted = false;
    this.clearEventListeners();
    this.clearInterval();
  }

  getUrlState() {
    return Utils.getMetricPlotStateFromUrl(this.props.location.search);
  }

  static predictChartType(metrics: any) {
    // Show bar chart when every metric has exactly 1 metric history
    if (metrics && metrics.length && lodashEvery(metrics, (metric) => metric.history && metric.history.length === 1)) {
      return CHART_TYPE_BAR;
    }
    return CHART_TYPE_LINE;
  }

  static predictRunChartsCardChartType(metrics: any) {
    if (MetricsPlotPanel.predictChartType(metrics) === CHART_TYPE_BAR) {
      return RunsChartType.BAR;
    }
    return RunsChartType.LINE;
  }

  // Update page URL from component state. Intended to be called after React applies component
  // state updates, e.g. in a setState callback
  updateUrlState = (updatedState: any) => {
    const { runUuids, metricKey, location, navigate } = this.props;
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    const experimentIds = JSON.parse(qs.parse(location.search)['experiments']);
    const newState = {
      ...this.getUrlState(),
      ...updatedState,
    };
    const {
      selectedXAxis,
      selectedMetricKeys,
      showPoint,
      yAxisLogScale,
      lineSmoothness,
      layout,
      deselectedCurves,
      lastLinearYAxisRange,
    } = newState;
    navigate(
      Routes.getMetricPageRoute(
        runUuids,
        metricKey,
        experimentIds,
        selectedMetricKeys,
        layout,
        selectedXAxis,
        yAxisLogScale,
        lineSmoothness,
        showPoint,
        deselectedCurves,
        lastLinearYAxisRange,
      ),
      {
        replace: true,
      },
    );
  };

  loadMetricHistory = (runUuids: any, metricKeys: any) => {
    this.setState({ loading: true });
    const promises = runUuids
      .flatMap((id: any) =>
        metricKeys.map((key: any) => ({
          runUuid: id,
          metricKey: key,
        })),
      )
      // Avoid fetching non existing metrics
      .filter(({ runUuid, metricKey }: any) => this.props.latestMetricsByRunUuid[runUuid].hasOwnProperty(metricKey))
      .map(async ({ runUuid, metricKey }: any) => {
        const requestIds = [];
        const id = getUUID();
        requestIds.push(id);
        const firstPageResp = await this.props.getMetricHistoryApi(
          runUuid,
          metricKey,
          GET_METRIC_HISTORY_MAX_RESULTS,
          undefined,
          id,
        );

        let nextPageToken = firstPageResp.value.next_page_token;
        while (nextPageToken) {
          const uid = getUUID();
          requestIds.push(uid);
          /* eslint-disable no-await-in-loop */
          const nextPageResp = await this.props.getMetricHistoryApi(
            runUuid,
            metricKey,
            GET_METRIC_HISTORY_MAX_RESULTS,
            nextPageToken,
            uid,
          );
          nextPageToken = nextPageResp.value.next_page_token;
        }
        return { requestIds, success: true };
      });
    this.loadMetricHistoryPromise = Promise.all(promises).then((results) => {
      // Ensure we don't set state if component is unmounted
      if (this._isMounted) {
        this.setState({ loading: false });
      }
      return results.flatMap(({ requestIds }) => requestIds);
    });
    return this.loadMetricHistoryPromise;
  };

  loadRuns = (runUuids: any) => {
    const requestIds: any = [];
    runUuids.forEach((runUuid: any) => {
      const id = getUUID();
      this.props.getRunApi(runUuid);
      requestIds.push(id);
    });
    return requestIds;
  };

  getMetrics = () => {
    /* eslint-disable no-param-reassign */
    const state = this.getUrlState();
    const selectedMetricsSet = new Set(state.selectedMetricKeys);
    const { selectedXAxis } = state;
    const { metricsWithRunInfoAndHistory } = this.props;

    // Take only selected metrics
    const metrics = metricsWithRunInfoAndHistory.filter((m) => selectedMetricsSet.has(m.metricKey));

    // Sort metric history based on selected x-axis
    metrics.forEach((metric) => {
      const isStep = selectedXAxis === X_AXIS_STEP && metric.history[0] && isNumber(metric.history[0].step);
      // Metric history can be large. Doing an in-place here to save memory
      metric.history.sort(isStep ? Utils.compareByStepAndTimestamp : Utils.compareByTimestamp);
    });

    return metrics;
  };

  /**
   * Handle changes in the scale type of the y-axis
   * @param yAxisLogScale: Boolean - if true, y-axis should be converted to log scale, and if false,
   * y-axis scale should be converted to a linear scale.
   */
  handleYAxisLogScaleChange = (yAxisLogScale: any) => {
    const state = this.getUrlState();
    const newLayout = cloneDeep(state.layout);
    const newAxisType = yAxisLogScale ? 'log' : 'linear';

    // Handle special case of a linear y-axis scale with negative values converted to log scale &
    // now being restored to linear scale, by restoring the old linear-axis range from
    // state.linearYAxisRange. In particular, we assume that if state.linearYAxisRange
    // is non-empty, it contains a linear y axis range with negative values.
    if (!yAxisLogScale && (state as any).lastLinearYAxisRange && (state as any).lastLinearYAxisRange.length > 0) {
      newLayout.yaxis = {
        type: 'linear',
        range: (state as any).lastLinearYAxisRange,
      };
      this.updateUrlState({ layout: newLayout, lastLinearYAxisRange: [], yAxisLogScale });
      return;
    }

    // Otherwise, if plot previously had no y axis range configured, simply set the axis type to
    // log or linear scale appropriately
    if (!state.layout.yaxis || !state.layout.yaxis.range) {
      newLayout.yaxis = {
        type: newAxisType,
        autorange: true,
        ...(newAxisType === 'log' ? { exponentformat: 'e' } : {}),
      };
      this.updateUrlState({ layout: newLayout, lastLinearYAxisRange: [], yAxisLogScale });
      return;
    }

    // lastLinearYAxisRange contains the last range used for a linear-scale y-axis. We set
    // this state attribute if and only if we're converting from a linear-scale y-axis with
    // negative bounds to a log scale axis, so that we can restore the negative bounds if we
    // subsequently convert back to a linear scale axis. Otherwise, we reset this attribute to an
    // empty array
    let lastLinearYAxisRange = [];

    // At this point, we know the plot previously had a y axis specified with range bounds
    // Convert the range to/from log scale as appropriate
    const oldLayout = state.layout;
    const oldYRange = oldLayout.yaxis.range;
    if (yAxisLogScale) {
      if (oldYRange[0] <= 0) {
        lastLinearYAxisRange = oldYRange;
        // When converting to log scale, handle negative values (which have no log-scale
        // representation as taking the log of a negative number is not possible) as follows:
        // If bottom of old Y range is negative, then tell plotly to infer the log y-axis scale
        // (set 'autorange' to true), and preserve the old range in the lastLinearYAxisRange
        // state attribute so that we can restore it if the user converts back to a linear-scale
        // y axis. We defer to Plotly's autorange here under the assumption that it will produce
        // a reasonable y-axis log scale for plots containing negative values.
        newLayout.yaxis = {
          type: 'log',
          autorange: true,
          exponentformat: 'e',
        };
      } else {
        newLayout.yaxis = {
          type: 'log',
          range: [Math.log(oldYRange[0]) / Math.log(10), Math.log(oldYRange[1]) / Math.log(10)],
          exponentformat: 'e',
        };
      }
    } else {
      // Otherwise, convert from log to linear scale normally
      newLayout.yaxis = {
        type: 'linear',
        range: [Math.pow(10, oldYRange[0]), Math.pow(10, oldYRange[1])],
      };
    }
    this.updateUrlState({ layout: newLayout, lastLinearYAxisRange });
  };

  /**
   * Handle changes in the type of the metric plot's X axis (e.g. changes from wall-clock
   * scale to relative-time scale to step-based scale).
   * @param e: Selection event such that e.target.value is a string containing the new X axis type
   */
  handleXAxisChange = (e: any) => {
    // Set axis value type, & reset axis scaling via autorange
    const state = this.getUrlState();
    const axisType = convertXAxisType(e.target.value);
    const newLayout = {
      ...state.layout,
      xaxis: {
        autorange: true,
        type: axisType,
      },
    };
    this.xAxisType = axisType;
    this.updateUrlState({ selectedXAxis: e.target.value, layout: newLayout });
  };

  getAxisType() {
    const state = this.getUrlState();
    return state.layout && state.layout.yaxis && state.layout.yaxis.type === 'log' ? 'log' : 'linear';
  }

  handleDownloadCsv = async () => {
    const { loading } = this.state;
    if (!loading) {
      const state = this.getUrlState();
      const selectedMetricKeys = state.selectedMetricKeys || [];
      this.loadMetricHistoryPromise = this.loadMetricHistory(this.props.runUuids, selectedMetricKeys);
    }
    await this.loadMetricHistoryPromise;

    // Filter by currently selected metrics (like getMetrics() does)
    const state = this.getUrlState();
    const selectedMetricsSet = new Set(state.selectedMetricKeys);
    const filteredMetrics = this.props.metricsWithRunInfoAndHistory.filter((m) => selectedMetricsSet.has(m.metricKey));

    const csv = convertMetricsToCsv(filteredMetrics);
    const blob = new Blob([csv], { type: 'application/csv;charset=utf-8' });
    saveAs(blob, 'metrics.csv');
  };

  handleMetricsSelectChange = (metricKeys: any) => {
    const existingMetricKeys = this.getUrlState().selectedMetricKeys || [];
    const newMetricKeys = metricKeys.filter((k: any) => !existingMetricKeys.includes(k));
    this.updateUrlState({ selectedMetricKeys: metricKeys });
    this.loadMetricHistory(this.props.runUuids, newMetricKeys).then((requestIds) => {
      this.setState({ loading: false });
      this.setState((prevState: any) => ({
        historyRequestIds: [...prevState.historyRequestIds, ...requestIds],
      }));
    });
  };

  handleShowPointChange = (showPoint: any) => this.updateUrlState({ showPoint });

  handleLineSmoothChange = (lineSmoothness: any) => this.updateUrlState({ lineSmoothness });

  getCardConfig = () => {
    const { metricKey } = this.props;
    const urlState = this.getUrlState();
    const metrics = this.getMetrics();
    const chartType = MetricsPlotPanel.predictRunChartsCardChartType(metrics);

    const empty_config = RunsChartsCardConfig.getEmptyChartCardByType(chartType, false, getUUID());
    const selectedMetricKeys =
      urlState.selectedMetricKeys && urlState.selectedMetricKeys.length > 0 ? urlState.selectedMetricKeys : [metricKey];

    const config = {
      ...empty_config,
      metricKey,
      selectedMetricKeys,
      xAxisKey: convertXAxisType(urlState.selectedXAxis),
      yAxisKey: RunsChartsLineChartYAxisType.METRIC,
      scaleType: urlState.yAxisLogScale ? 'log' : 'linear',
      displayPoints: urlState.showPoint,
      lineSmoothness: urlState.lineSmoothness,
    };

    if (chartType === RunsChartType.LINE && urlState.layout && Object.keys(urlState.layout).length > 0) {
      (config as RunsChartsLineCardConfig).range = {
        xMin: urlState.layout.xaxis?.range?.[0],
        xMax: urlState.layout.xaxis?.range?.[1],
        yMin: urlState.layout.yaxis?.range?.[0],
        yMax: urlState.layout.yaxis?.range?.[1],
      };
    }

    return config;
  };

  getChartRunData = () => {
    const { runUuids, metricKey, runInfos, runNames, runDisplayNames } = this.props;
    const urlState = this.getUrlState();
    const selectedMetricKeys =
      urlState.selectedMetricKeys && urlState.selectedMetricKeys.length > 0 ? urlState.selectedMetricKeys : [metricKey];
    const metrics = selectedMetricKeys.reduce((acc: Record<string, any>, key: string) => {
      acc[key] = {}; // Empty object indicates metric exists but data will be fetched
      return acc;
    }, {});

    return runUuids.map(
      (runUuid, index) =>
        ({
          uuid: runUuid,
          runInfo: runInfos?.[index], // Use actual runInfo from props
          metrics,
          params: {},
          tags: {},
          datasets: [],
          images: {},
          hidden: false,
          color: getStableColorForRun(runUuid),
          displayName: runNames?.[index] || runDisplayNames?.[index] || runUuid,
        } as RunsChartsRunData),
    );
  };

  getGlobalLineChartConfig = () => {
    const { lineSmoothness, selectedXAxis } = this.getUrlState();
    return {
      lineSmoothness,
      xAxisKey: convertXAxisType(selectedXAxis),
      selectedXAxisMetricKey: undefined,
    };
  };

  getTooltipContextValue = () => {
    return {
      runs: this.getChartRunData(),
    };
  };

  setFullScreenChart = (chart: any) => {
    this.setState({ fullScreenChart: chart });
  };

  render() {
    const { runUuids, runDisplayNames, distinctMetricKeys } = this.props;
    const { loading, historyRequestIds } = this.state;
    const state = this.getUrlState();
    const { showPoint, selectedXAxis, selectedMetricKeys, lineSmoothness } = state;
    const yAxisLogScale = this.getAxisType() === 'log';
    const metrics = this.getMetrics();
    const chartType = MetricsPlotPanel.predictChartType(metrics);

    return (
      <div className="mlflow-metrics-plot-container">
        <MetricsPlotControls
          // @ts-expect-error TS(2322): Type '{ numRuns: number; numCompletedRuns: number;... Remove this comment to see the full error message
          numRuns={this.props.runUuids.length}
          numCompletedRuns={this.props.completedRunUuids.length}
          distinctMetricKeys={distinctMetricKeys}
          selectedXAxis={selectedXAxis}
          selectedMetricKeys={selectedMetricKeys}
          handleXAxisChange={this.handleXAxisChange}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
          handleShowPointChange={this.handleShowPointChange}
          handleYAxisLogScaleChange={this.handleYAxisLogScaleChange}
          handleLineSmoothChange={this.handleLineSmoothChange}
          chartType={chartType}
          lineSmoothness={lineSmoothness}
          yAxisLogScale={yAxisLogScale}
          showPoint={showPoint}
          handleDownloadCsv={this.handleDownloadCsv}
          disableSmoothnessControl={this.props.containsInfinities}
        />
        <div className="metrics-plot-data">
          <RequestStateWrapper
            requestIds={historyRequestIds}
            // In this case where there are no history request IDs (e.g. on the
            // initial page load / before we try to load additional metrics),
            // optimistically render the children
            shouldOptimisticallyRender={historyRequestIds.length === 0}
          >
            <Spinner size="large" css={{ visibility: loading ? 'visible' : 'hidden' }} />
            <RunsChartsTooltipWrapper contextData={this.getTooltipContextValue()} component={RunsChartsTooltipBody}>
              <RunsChartsCard
                key={runUuids.join() + this.xAxisType}
                cardConfig={this.getCardConfig()}
                chartRunData={this.getChartRunData()}
                onStartEditChart={() => {}}
                onRemoveChart={() => {}}
                onDownloadFullMetricHistoryCsv={() => {}}
                index={0}
                groupBy={null}
                fullScreen={false}
                autoRefreshEnabled={this.shouldPoll()}
                hideEmptyCharts={false}
                globalLineChartConfig={this.getGlobalLineChartConfig()}
                isInViewport
                isInViewportDeferred
                setFullScreenChart={this.setFullScreenChart}
                onReorderWith={() => {}}
                canMoveUp={false}
                canMoveDown={false}
                canMoveToTop={false}
                canMoveToBottom={false}
              />
            </RunsChartsTooltipWrapper>
            <RunsChartsFullScreenModal
              fullScreenChart={this.state['fullScreenChart']}
              onCancel={() => this.setFullScreenChart(undefined)}
              chartData={this.getChartRunData()}
              tooltipContextValue={this.getTooltipContextValue()}
              tooltipComponent={RunsChartsTooltipBody}
              autoRefreshEnabled={this.shouldPoll()}
              groupBy={null}
            />
            <MetricsSummaryTable
              runUuids={runUuids}
              runDisplayNames={runDisplayNames}
              metricKeys={selectedMetricKeys}
            />
          </RequestStateWrapper>
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { runUuids } = ownProps;
  const completedRunUuids = runUuids.filter((runUuid: any) => getRunInfo(runUuid, state).status !== 'RUNNING');
  const { latestMetricsByRunUuid, metricsByRunUuid } = state.entities;

  // All metric keys from all runUuids, non-distinct
  const metricKeys = lodashFlatMap(runUuids, (runUuid) => {
    const latestMetrics = latestMetricsByRunUuid[runUuid];
    return latestMetrics ? Object.keys(latestMetrics) : [];
  });
  const distinctMetricKeys = [...new Set(metricKeys)].sort();
  const runDisplayNames: any = [];

  let containsInfinities = false;

  // Flat array of all metrics, with history and information of the run it belongs to
  // This is used for underlying MetricsPlotView & predicting chartType for MetricsPlotControls
  const metricsWithRunInfoAndHistory = lodashFlatMap(runUuids, (runUuid) => {
    const runDisplayName = Utils.getRunDisplayName(getRunInfo(runUuid, state), runUuid);
    runDisplayNames.push(runDisplayName);
    const metricsHistory = metricsByRunUuid[runUuid];
    return metricsHistory
      ? Object.keys(metricsHistory).map((metricKey) => {
          const history = metricsHistory[metricKey].map((entry: any) => normalizeMetricsHistoryEntry(entry));
          if (history.some(({ value }: any) => typeof value === 'number' && !isNaN(value) && !isFinite(value))) {
            containsInfinities = true;
          }
          return { metricKey, history, runUuid, runDisplayName };
        })
      : [];
  });

  // Additional data for new chart system
  const runInfos = runUuids.map((runUuid: any) => getRunInfo(runUuid, state));
  const runNames = runInfos.map((runInfo: any, index: number) => {
    return Utils.getRunDisplayName(runInfo, runUuids[index]);
  });

  return {
    runDisplayNames,
    latestMetricsByRunUuid,
    distinctMetricKeys,
    metricsWithRunInfoAndHistory,
    completedRunUuids,
    containsInfinities,
    runInfos,
    runNames,
  };
};

const mapDispatchToProps = { getMetricHistoryApi, getRunApi };

export default withRouterNext(connect(mapStateToProps, mapDispatchToProps)(MetricsPlotPanel));
