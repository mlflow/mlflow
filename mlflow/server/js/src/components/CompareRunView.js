import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getParams, getRunInfo } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './CompareRunView.css';
import { RunInfo } from '../sdk/MlflowMessages';
import HtmlTableView from './HtmlTableView';
import Routes from '../Routes';
import { Link } from 'react-router-dom';
import Utils from '../utils/Utils';
import { getLatestMetrics } from '../reducers/MetricReducer';

class CompareRunView extends Component {
  static propTypes = {
    runInfos: PropTypes.arrayOf(RunInfo).required,
    metricLists: PropTypes.arrayOf(Array).required,
    paramLists: PropTypes.arrayOf(Array).required,
  };

  render() {
    const tableStyles = {
      'tr': {
        display: 'flex',
        justifyContent: 'flex-start',
      },
      'td': {
        flex: '1',
      },
      'th': {
        flex: '1',
      },
      'td-first': {
        width: '500px',
      },
      'th-first': {
        width: '500px',
      },
    };
    return (
      <div className="CompareRunView">
        <h1>Comparing {this.props.runInfos.length} Runs</h1>
        <div className="run-metadata-container">
          <div className="run-metadata-label">Run UUID:</div>
          <div className="run-metadata-row">
            {this.props.runInfos.map((r) => <div className="run-metadata-item">{r.getRunUuid()}</div>)}
          </div>
        </div>
        <div className="run-metadata-container last-run-metadata-container">
          <div className="run-metadata-label">Start Time:</div>
          <div className="run-metadata-row">
            {this.props.runInfos.map((run) => {
               const startTime = run.getStartTime() ? Utils.formatTimestamp(run.getStartTime()) : '(unknown)';
               return <div className="run-metadata-item">{startTime}</div>;
             }
            )}
          </div>
        </div>
        <h2>Parameters</h2>
        <HtmlTableView
          columns={["Name", "", ""]}
          values={Private.getParamRows(this.props.runInfos, this.props.paramLists)}
          styles={tableStyles}
        />
        <h2>Metrics</h2>
        <HtmlTableView
          columns={["Name", "", ""]}
          values={Private.getLatestMetricRows(this.props.runInfos, this.props.metricLists)}
          styles={tableStyles}
        />

      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const runInfos = [];
  const metricLists = [];
  const paramLists = [];
  const { runUuids } = ownProps;
  runUuids.forEach((runUuid) => {
    runInfos.push(getRunInfo(runUuid, state));
    metricLists.push(Object.values(getLatestMetrics(runUuid, state)));
    paramLists.push(Object.values(getParams(runUuid, state)));
  });
  return { runInfos, metricLists, paramLists };
};

export default connect(mapStateToProps)(CompareRunView);

class Private {
  static getParamRows(runInfos, paramLists) {
    const rows = [];
    // Map of parameter key to a map of (runUuid -> value)
    const paramKeyValueList = [];
    paramLists.forEach((paramList) => {
      const curKeyValueObj = {};
      paramList.forEach((param) => {
        curKeyValueObj[param.key] = param.value;
      });
      paramKeyValueList.push(curKeyValueObj);
    });

    const mergedParams = Utils.mergeRuns(runInfos.map((r) => r.run_uuid), paramKeyValueList);

    Object.keys(mergedParams).sort().forEach((paramKey) => {
      const curRow = [];
      curRow.push(paramKey);
      runInfos.forEach((r) => {
        const curUuid = r.run_uuid;
        curRow.push(mergedParams[paramKey][curUuid]);
      });
      rows.push(curRow)
    });
    return rows;
  }

  static getLatestMetricRows(runInfos, metricLists) {
    const rows = [];
    // Map of parameter key to a map of (runUuid -> value)
    const metricKeyValueList = [];
    metricLists.forEach((metricList) => {
      const curKeyValueObj = {};
      metricList.forEach((metric) => {
        curKeyValueObj[metric.key] = Utils.formatMetric(metric.value);
      });
      metricKeyValueList.push(curKeyValueObj);
    });

    const mergedMetrics = Utils.mergeRuns(runInfos.map((r) => r.run_uuid), metricKeyValueList);

    const runUuids = runInfos.map((r) => r.run_uuid);
    Object.keys(mergedMetrics).sort().forEach((metricKey) => {
      // Figure out which runUuids actually have this metric.
      const runUuidsWithMetric = Object.keys(mergedMetrics[metricKey]);
      const curRow = [];
      curRow.push(<Link to={Routes.getMetricPageRoute(runUuidsWithMetric, metricKey)}>{metricKey}</Link>);
      runInfos.forEach((r) => {
        const curUuid = r.run_uuid;
        curRow.push(Math.round(mergedMetrics[metricKey][curUuid] * 1e4)/1e4);
      });
      rows.push(curRow)
    });
    return rows;
  }
}
