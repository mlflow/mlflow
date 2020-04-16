import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getParams, getRunInfo, getRunTags } from '../../experiment-tracking/reducers/Reducers';
import { connect } from 'react-redux';
import '../../experiment-tracking/components/CompareRunView.css';
import { RunInfo } from '../../experiment-tracking/sdk/MlflowMessages';
import CompareRunScatter from '../../experiment-tracking/components/CompareRunScatter';
import CompareRunContour from '../../experiment-tracking/components/CompareRunContour';
import Routes from '../../experiment-tracking/routes';
import { Link } from 'react-router-dom';
import { getLatestMetrics } from '../../experiment-tracking/reducers/MetricReducer';
import CompareRunUtil from '../../experiment-tracking/components/CompareRunUtil';
import Utils from '../../common/utils/Utils';
import { Tabs } from 'antd';
import ParallelCoordinatesPlotPanel from
    '../../experiment-tracking/components/ParallelCoordinatesPlotPanel';
import {modelListPageRoute, getModelPageRoute, getModelVersionPageRoute} from "../routes";

const TabPane = Tabs.TabPane;

export class CompareModelVersionsView extends Component {
  static propTypes = {
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    runUuids: PropTypes.arrayOf(String).isRequired,
    metricLists: PropTypes.arrayOf(Array).isRequired,
    paramLists: PropTypes.arrayOf(Array).isRequired,
    // Array of user-specified run names. Elements may be falsy (e.g. empty string or undefined) if
    // a run was never given a name.
    runNames: PropTypes.arrayOf(String).isRequired,
    // Array of names to use when displaying runs. No element in this array should be falsy;
    // we expect this array to contain user-specified run names, or default display names
    // ("Run <uuid>") for runs without names.
    runDisplayNames: PropTypes.arrayOf(String).isRequired,
    modelName: PropTypes.string.isRequired,
    runsToVersions: PropTypes.object.isRequired,
  };

  render() {
    const { runInfos, runNames, modelName} = this.props;
    const chevron = <i className='fas fa-chevron-right breadcrumb-chevron' />;
    const breadcrumbItemClass = 'truncate-text single-line breadcrumb-title';
    return (
      <div className="CompareModelVersionsView">
        <h1 className='breadcrumb-header'>
          <Link to={modelListPageRoute} className={breadcrumbItemClass}>Registered Models</Link>
          {chevron}
          <Link to={getModelPageRoute(modelName)} className={breadcrumbItemClass}>{modelName}</Link>
          {chevron}
          <span className={breadcrumbItemClass}>
            {"Comparing " + this.props.runInfos.length + " Versions"}</span>
        </h1>
        <div className="responsive-table-container">
          <table className="compare-table table">
            <thead>
            <tr>
              <th scope="row" className="row-header">Run ID:</th>
              {this.props.runInfos.map(r =>
                <th scope="column" className="data-value" key={r.getRunUuid()}>
                  <Link to={Routes.getRunPageRoute(r.getExperimentId(), r.getRunUuid())}>
                    {r.getRunUuid()}
                  </Link>
                </th>
              )}
            </tr>
            </thead>
            <tbody>
            <tr>
              <th scope="row" className="data-value">Model Version:</th>
              {Object.keys(this.props.runsToVersions).map((run) => {
                const version = this.props.runsToVersions[run];
                return (<td className="meta-info" key={run}>
                  <Link to={getModelVersionPageRoute(modelName, version)}>
                    {version}
                  </Link>
                </td>);
              }
              )}
            </tr>
            <tr>
              <th scope="row" className="data-value">Run Name:</th>
              {runNames.map((runName, i) => {
                return (<td
                  className="meta-info"
                  key={runInfos[i].getRunUuid()}
                  >
                    <div
                      className="truncate-text single-line"
                      style={styles.compareRunTableCellContents}
                    >
                      {runName}
                    </div>
                  </td>);
              }
              )}

            </tr>
            <tr>
              <th scope="row" className="data-value">Start Time:</th>
              {this.props.runInfos.map((run) => {
                const startTime =
                  run.getStartTime() ? Utils.formatTimestamp(run.getStartTime()) : '(unknown)';
                return <td className="meta-info" key={run.getRunUuid()}>{startTime}</td>;
              }
              )}
            </tr>
            <tr>
              <th scope="rowgroup"
                  className="inter-title"
                  colSpan={this.props.runInfos.length + 1}>
                <h2>Parameters</h2>
              </th>
            </tr>
            {this.renderDataRows(this.props.paramLists)}
            <tr>
              <th scope="rowgroup"
                  className="inter-title"
                  colSpan={this.props.runInfos.length + 1}>
                <h2>Metrics</h2>
              </th>
            </tr>
            {this.renderDataRows(this.props.metricLists, (key, data) => {
              return <Link
                to={Routes.getMetricPageRoute(
                  this.props.runInfos.map(info => info.run_uuid)
                    .filter((uuid, idx) => data[idx] !== undefined),
                  key,
                  // TODO: Refactor so that the breadcrumb on the linked page is for model registry
                  this.props.runInfos[0].experiment_id)}
                title="Plot chart">
                {key}
                <i className="fas fa-chart-line" style={{paddingLeft: "6px"}}/>
              </Link>;
            }, Utils.formatMetric)}
            </tbody>
          </table>
        </div>
        <Tabs>
          <TabPane tab="Scatter Plot" key="1">
            <CompareRunScatter
              runUuids={this.props.runUuids}
              runDisplayNames={this.props.runDisplayNames}
            />
          </TabPane>
          <TabPane tab="Contour Plot" key="2">
            <CompareRunContour
              runUuids={this.props.runUuids}
              runDisplayNames={this.props.runDisplayNames}
            />
          </TabPane>
          <TabPane tab="Parallel Coordinates Plot" key="3">
            <ParallelCoordinatesPlotPanel runUuids={this.props.runUuids}/>
          </TabPane>
        </Tabs>
      </div>
    );
  }

  // eslint-disable-next-line no-unused-vars
  renderDataRows(list, headerMap = (key, data) => key, formatter = (value) => value) {
    const keys = CompareRunUtil.getKeys(list);
    const data = {};
    keys.forEach(k => data[k] = []);
    list.forEach((records, i) => {
      keys.forEach(k => data[k].push(undefined));
      records.forEach(r => data[r.key][i] = r.value);
    });

    return keys.map(k => {
      return <tr key={k}>
        <th scope="row" className="rowHeader">{headerMap(k, data[k])}</th>
        {data[k].map((value, i) =>
          <td className="data-value" key={this.props.runInfos[i].getRunUuid()}>
            <span className="truncate-text single-line" style={styles.compareRunTableCellContents}>
              {value === undefined ? "" : formatter(value)}
            </span>
          </td>
        )}
      </tr>;
    });
  }
}

const styles = {
  compareRunTableCellContents: {
    maxWidth: '200px',
  },
};

const mapStateToProps = (state, ownProps) => {
  const runInfos = [];
  const metricLists = [];
  const paramLists = [];
  const runNames = [];
  const runDisplayNames = [];
  const runUuids = [];
  const { modelName, runsToVersions } = ownProps;
  for (const runUuid in runsToVersions) {
    if ({}.hasOwnProperty.call(runsToVersions, runUuid)) {
      runInfos.push(getRunInfo(runUuid, state));
      metricLists.push(Object.values(getLatestMetrics(runUuid, state)));
      paramLists.push(Object.values(getParams(runUuid, state)));
      const runTags = getRunTags(runUuid, state);
      runDisplayNames.push(Utils.getRunDisplayName(runTags, runUuid));
      runNames.push(Utils.getRunName(runTags));
      runUuids.push(runUuid);
    }
  }

  return { runInfos, metricLists, paramLists, runNames, runDisplayNames, runUuids, modelName };
};

export default connect(mapStateToProps)(CompareModelVersionsView);
