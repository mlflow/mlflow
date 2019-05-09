import React from 'react';
import { connect } from 'react-redux';
import Utils from '../utils/Utils';
import RequestStateWrapper from './RequestStateWrapper';
import { getMetricHistoryApi, getUUID } from '../Actions';
import PropTypes from 'prop-types';
import _ from 'lodash';
import { MetricsPlotView } from './MetricsPlotView';
import { getRunTags } from '../reducers/Reducers';
import { MetricsPlotControls, X_AXIS_RELATIVE, X_AXIS_STEP } from './MetricsPlotControls';
import qs from 'qs';
import { withRouter } from 'react-router-dom';
import Routes from '../Routes';

export const CHART_TYPE_LINE = 'line';
export const CHART_TYPE_BAR = 'bar';

class MetricsPlotPanel extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    metricKey: PropTypes.string.isRequired,
    latestMetricsByRunUuid: PropTypes.object.isRequired,
    distinctMetricKeys: PropTypes.arrayOf(String).isRequired,
    metricsWithRunInfoAndHistory: PropTypes.arrayOf(Object).isRequired,
    getMetricHistoryApi: PropTypes.func.isRequired,
    location: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired,
  };

  constructor(props) {
    super(props);
    const plotMetricKeys = MetricsPlotPanel.getPlotMetricKeysFromUrl(props.location.search);
    const selectedMetricKeys = plotMetricKeys.length ? plotMetricKeys : [props.metricKey];
    this.state = {
      selectedXAxis: X_AXIS_RELATIVE,
      selectedMetricKeys,
      showDot: false,
      historyRequestIds: [],
    };
    this.loadMetricHistory(this.props.runUuids, selectedMetricKeys);
  }

  static predictChartType(metrics) {
    // Show bar chart when every metric has exactly 1 metric history
    if (metrics && metrics.length && _.every(metrics, (m) => m.history && m.history.length === 1)) {
      return CHART_TYPE_BAR;
    }
    return CHART_TYPE_LINE;
  }

  static getPlotMetricKeysFromUrl = (search) => JSON.parse(qs.parse(search)['plot_metric_keys']);

  updateUrlWithSelectedMetrics(selectedMetricKeys) {
    const { runUuids, metricKey, location, history } = this.props;
    const params = qs.parse(location.search);
    const experimentId = params['experiment'];
    history.push(Routes.getMetricPageRoute(runUuids, metricKey, experimentId, selectedMetricKeys));
  }

  loadMetricHistory = (runUuids, metricKeys) => {
    const requestIds = [];
    const { latestMetricsByRunUuid } = this.props;
    runUuids.forEach((runUuid) => {
      metricKeys.forEach((metricKey) => {
        if (latestMetricsByRunUuid[runUuid][metricKey]) {
          const id = getUUID();
          this.props.getMetricHistoryApi(runUuid, metricKey, id);
          requestIds.push(id);
        }
      });
    });
    return requestIds;
  };

  getAllMetricKeys = () => {
    const { distinctMetricKeys } = this.props;
    return distinctMetricKeys.map((metricKey) => ({
      title: metricKey,
      value: metricKey,
      key: metricKey,
    }));
  };

  getMetrics = () => {
    const selectedMetricsSet = new Set(this.state.selectedMetricKeys);
    const { selectedXAxis } = this.state;
    const { metricsWithRunInfoAndHistory } = this.props;
    // Handle selected metrics
    const metrics = metricsWithRunInfoAndHistory.filter((m) => selectedMetricsSet.has(m.metricKey));
    // Handle selected chart type
    metrics.forEach((m) =>
      _.sortBy(
        m.history,
        // TODO(Zangr) handle empty step metrics
        selectedXAxis === X_AXIS_STEP && m.history[0] && _.isNumber(m.history[0].step)
          ? ['step', 'timestamp']
          : 'timestamp',
      ),
    );
    return metrics;
  };

  handleXAxisChange = (e) => {
    this.setState({ selectedXAxis: e.target.value });
  };

  handleMetricsSelectChange = (metricValues, metricLabels, { triggerValue }) => {
    const requestIds = this.loadMetricHistory(this.props.runUuids, [triggerValue]);
    this.setState((prevState) => ({
      selectedMetricKeys: metricValues,
      historyRequestIds: [...prevState.historyRequestIds, ...requestIds],
    }));
    this.updateUrlWithSelectedMetrics(metricValues);
  };

  handleShowDotChange = (showDot) => {
    this.setState({ showDot });
  };

  render() {
    const { runUuids } = this.props;
    const { historyRequestIds, showDot, selectedXAxis, selectedMetricKeys } = this.state;
    const metrics = this.getMetrics();
    const chartType = MetricsPlotPanel.predictChartType(metrics);
    return (
      <div className='metrics-plot-container'>
        <MetricsPlotControls
          allMetricKeys={this.getAllMetricKeys()}
          selectedXAxis={selectedXAxis}
          selectedMetricKeys={selectedMetricKeys}
          handleXAxisChange={this.handleXAxisChange}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
          handleShowDotChange={this.handleShowDotChange}
          chartType={chartType}
        />
        <RequestStateWrapper requestIds={historyRequestIds}>
          <MetricsPlotView
            runUuids={runUuids}
            xAxis={selectedXAxis}
            metrics={this.getMetrics()}
            metricKeys={selectedMetricKeys}
            showDot={showDot}
            chartType={chartType}
          />
        </RequestStateWrapper>
      </div>
    );
  }
}

// TODO(Zangr) remove after chart testing is done.
const tuneHistory = (metricsHistory) => {
  metricsHistory.forEach((m, ii) => {
    m.history = m.history.map((entry) => ({
      key: entry.key,
      value: entry.value,
      step: Number.parseInt(entry.step, 10),
      timestamp: Number.parseFloat(entry.timestamp),
    }));

    m.history.forEach((entry, i) => {
      entry.value = Number(entry.key.substr(-1)) * i * i + 5 * i + 30 * entry.value;
    });
  });
};

// TODO(Zangr) remove after chart testing is done.
const tuneHistoryWall = (metricsHistory) => {
  metricsHistory.forEach((m, ii) => {
    m.history = m.history.map((entry) => ({
      key: entry.key,
      value: entry.value,
      step: Number.parseInt(entry.step, 10),
      timestamp: Number.parseFloat(entry.timestamp),
    }));

    m.history.forEach((entry, i) => {
      entry.value = Number(entry.key.substr(-1)) * i * i + 5 * i + 30 * entry.value;
      entry.timestamp += ii * 100 * 1500;
    });
  });
};

// TODO(Zangr) remove after chart testing is done.
const forceSingleHistory = (metrics) => {
  metrics.forEach((m) => (m.history = [m.history[0]]));
};

// TODO(Zangr) remove after chart testing is done.
const forceSingleHistoryExceptOne = (metrics) => {
  metrics.forEach((m, i) => (m.history = m.metricKey.includes('4') ? m.history : [m.history[0]]));
};

const mapStateToProps = (state, ownProps) => {
  const { runUuids } = ownProps;
  const { latestMetricsByRunUuid, metricsByRunUuid } = state.entities;
  const metricKeys = _.flatMap(runUuids, (runUuid) => {
    const latestMetrics = latestMetricsByRunUuid[runUuid];
    return latestMetrics ? Object.keys(latestMetrics) : [];
  });
  const distinctMetricKeys = [...new Set(metricKeys)].sort();
  const metricsWithRunInfoAndHistory = _.flatMap(runUuids, (runUuid) => {
    const runName = Utils.getRunDisplayName(getRunTags(runUuid, state), runUuid);
    const metricsHistory = metricsByRunUuid[runUuid];
    return metricsHistory
      ? Object.keys(metricsHistory).map((metricKey, index) => {
        const history = _.sortBy(metricsHistory[metricKey], 'timestamp');
        return { metricKey, history, runUuid, runName, index };
      })
      : [];
  });
  // tuneHistory(metricsWithRunInfoAndHistory); // TODO(Zangr) remove tuning
  tuneHistoryWall(metricsWithRunInfoAndHistory); // TODO(Zangr) remove tuning
  // forceSingleHistory(metricsWithRunInfoAndHistory); // TODO(Zangr) remove tuning
  // forceSingleHistoryExceptOne(metricsWithRunInfoAndHistory); // TODO(Zangr) remove tunining

  return {
    latestMetricsByRunUuid,
    distinctMetricKeys,
    metricsWithRunInfoAndHistory,
  };
};

const mapDispatchToProps = { getMetricHistoryApi };

export default withRouter(
  connect(
    mapStateToProps,
    mapDispatchToProps,
  )(MetricsPlotPanel),
);
