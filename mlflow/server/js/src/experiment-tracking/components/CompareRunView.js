import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './CompareRunView.css';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
import { CompareRunScatter } from './CompareRunScatter';
import CompareRunContour from './CompareRunContour';
import Routes from '../routes';
import { Link } from 'react-router-dom';
import { getLatestMetrics } from '../reducers/MetricReducer';
import { BreadcrumbTitle } from './BreadcrumbTitle';
import CompareRunUtil from './CompareRunUtil';
import Utils from '../../common/utils/Utils';
import { Tabs } from 'antd';
import ParallelCoordinatesPlotPanel from './ParallelCoordinatesPlotPanel';

const { TabPane } = Tabs;

export class CompareRunView extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    experimentId: PropTypes.string.isRequired,
    runInfos: PropTypes.arrayOf(PropTypes.instanceOf(RunInfo)).isRequired,
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricLists: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    paramLists: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    // Array of user-specified run names. Elements may be falsy (e.g. empty string or undefined) if
    // a run was never given a name.
    runNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    // Array of names to use when displaying runs. No element in this array should be falsy;
    // we expect this array to contain user-specified run names, or default display names
    // ("Run <uuid>") for runs without names.
    runDisplayNames: PropTypes.arrayOf(PropTypes.string).isRequired,
  };

  componentDidMount() {
    const pageTitle = `Comparing ${this.props.runInfos.length} MLflow Runs`;
    Utils.updatePageTitle(pageTitle);
  }

  render() {
    const { experiment } = this.props;
    const experimentId = experiment.getExperimentId();
    const { runInfos, runNames } = this.props;
    return (
      <div className='CompareRunView'>
        <div className='header-container'>
          <BreadcrumbTitle
            experiment={experiment}
            title={'Comparing ' + this.props.runInfos.length + ' Runs'}
          />
        </div>
        <div className='responsive-table-container'>
          <table className='compare-table table'>
            <thead>
              <tr>
                <th scope='row' className='row-header'>
                  Run ID:
                </th>
                {this.props.runInfos.map((r) => (
                  <th scope='column' className='data-value' key={r.run_uuid}>
                    <Link to={Routes.getRunPageRoute(r.getExperimentId(), r.getRunUuid())}>
                      {r.getRunUuid()}
                    </Link>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope='row' className='data-value'>
                  Run Name:
                </th>
                {runNames.map((runName, i) => {
                  return (
                    <td className='meta-info' key={runInfos[i].run_uuid}>
                      <div
                        className='truncate-text single-line'
                        style={styles.compareRunTableCellContents}
                      >
                        {runName}
                      </div>
                    </td>
                  );
                })}
              </tr>
              <tr>
                <th scope='row' className='data-value'>
                  Start Time:
                </th>
                {this.props.runInfos.map((run) => {
                  const startTime = run.getStartTime()
                    ? Utils.formatTimestamp(run.getStartTime())
                    : '(unknown)';
                  return (
                    <td className='meta-info' key={run.run_uuid}>
                      {startTime}
                    </td>
                  );
                })}
              </tr>
              <tr>
                <th
                  scope='rowgroup'
                  className='inter-title'
                  colSpan={this.props.runInfos.length + 1}
                >
                  <h2>Parameters</h2>
                </th>
              </tr>
              {this.renderDataRows(this.props.paramLists, true)}
              <tr>
                <th
                  scope='rowgroup'
                  className='inter-title'
                  colSpan={this.props.runInfos.length + 1}
                >
                  <h2>Metrics</h2>
                </th>
              </tr>
              {this.renderDataRows(
                this.props.metricLists,
                false,
                (key, data) => {
                  return (
                    <Link
                      to={Routes.getMetricPageRoute(
                        this.props.runInfos
                          .map((info) => info.run_uuid)
                          .filter((uuid, idx) => data[idx] !== undefined),
                        key,
                        experimentId,
                      )}
                      title='Plot chart'
                    >
                      {key}
                      <i className='fas fa-chart-line' style={{ paddingLeft: '6px' }} />
                    </Link>
                  );
                },
                Utils.formatMetric,
              )}
            </tbody>
          </table>
        </div>
        <Tabs>
          <TabPane tab='Scatter Plot' key='1'>
            <CompareRunScatter
              runUuids={this.props.runUuids}
              runDisplayNames={this.props.runDisplayNames}
            />
          </TabPane>
          <TabPane tab='Contour Plot' key='2'>
            <CompareRunContour
              runUuids={this.props.runUuids}
              runDisplayNames={this.props.runDisplayNames}
            />
          </TabPane>
          <TabPane tab='Parallel Coordinates Plot' key='3'>
            <ParallelCoordinatesPlotPanel runUuids={this.props.runUuids} />
          </TabPane>
        </Tabs>
      </div>
    );
  }

  // eslint-disable-next-line no-unused-vars
  renderDataRows(
    list,
    highlightChanges = false,
    headerMap = (key, data) => key,
    formatter = (value) => value,
  ) {
    const keys = CompareRunUtil.getKeys(list);
    const data = {};
    keys.forEach((k) => (data[k] = []));
    list.forEach((records, i) => {
      keys.forEach((k) => data[k].push(undefined));
      records.forEach((r) => (data[r.key][i] = r.value));
    });

    return keys.map((k) => {
      let row_class = undefined;
      if (highlightChanges) {
        const all_equal = data[k].every((x) => x === data[k][0]);
        if (!all_equal) {
          row_class = 'row-changed';
        }
      }

      return (
        <tr key={k} className={row_class}>
          <th scope='row' className='rowHeader'>
            {headerMap(k, data[k])}
          </th>
          {data[k].map((value, i) => (
            <td className='data-value' key={this.props.runInfos[i].run_uuid}>
              <span
                className='truncate-text single-line'
                style={styles.compareRunTableCellContents}
              >
                {value === undefined ? '' : formatter(value)}
              </span>
            </td>
          ))}
        </tr>
      );
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
  const { experimentId, runUuids } = ownProps;
  const experiment = getExperiment(experimentId, state);
  runUuids.forEach((runUuid) => {
    runInfos.push(getRunInfo(runUuid, state));
    metricLists.push(Object.values(getLatestMetrics(runUuid, state)));
    paramLists.push(Object.values(getParams(runUuid, state)));
    const runTags = getRunTags(runUuid, state);
    runDisplayNames.push(Utils.getRunDisplayName(runTags, runUuid));
    runNames.push(Utils.getRunName(runTags));
  });
  return { experiment, runInfos, metricLists, paramLists, runNames, runDisplayNames };
};

export default connect(mapStateToProps)(CompareRunView);
