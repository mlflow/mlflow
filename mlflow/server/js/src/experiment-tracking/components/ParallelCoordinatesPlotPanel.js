import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import ParallelCoordinatesPlotView from './ParallelCoordinatesPlotView';
import { ParallelCoordinatesPlotControls } from './ParallelCoordinatesPlotControls';
import {
  getAllParamKeysByRunUuids,
  getAllMetricKeysByRunUuids,
  getSharedMetricKeysByRunUuids,
} from '../reducers/Reducers';
import _ from 'lodash';
import { Empty } from 'antd';

import './ParallelCoordinatesPlotPanel.css';

export class ParallelCoordinatesPlotPanel extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of all parameter keys across all runs
    allParamKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of all metric keys across all runs
    allMetricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of metric keys for which all runs have values
    sharedMetricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    // A subset of allParamKeys where the values, potentially undefined,
    // of the parameters differ between runs
    diffParamKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
  };

  state = {
    // Default to select differing parameters. Sort alphabetically (to match
    // highlighted params in param table), then cap at first 10
    selectedParamKeys: this.props.diffParamKeys.sort().slice(0, 10),
    // Default to select the first metric key.
    // Note that there will be no color scaling if no metric is selected.
    selectedMetricKeys: this.props.sharedMetricKeys.slice(0, 1),
  };

  handleParamsSelectChange = (paramValues) => {
    this.setState({ selectedParamKeys: paramValues });
  };

  handleMetricsSelectChange = (metricValues) => {
    this.setState({ selectedMetricKeys: metricValues });
  };

  render() {
    const { runUuids, allParamKeys, allMetricKeys } = this.props;
    const { selectedParamKeys, selectedMetricKeys } = this.state;
    return (
      <div className='parallel-coordinates-plot-panel'>
        <ParallelCoordinatesPlotControls
          paramKeys={allParamKeys}
          metricKeys={allMetricKeys}
          selectedParamKeys={selectedParamKeys}
          selectedMetricKeys={selectedMetricKeys}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
          handleParamsSelectChange={this.handleParamsSelectChange}
        />
        {!_.isEmpty(selectedParamKeys) || !_.isEmpty(selectedMetricKeys) ? (
          <ParallelCoordinatesPlotView
            runUuids={runUuids}
            paramKeys={selectedParamKeys}
            metricKeys={selectedMetricKeys}
          />
        ) : (
          <Empty style={{ width: '100%', height: '100%' }} />
        )}
      </div>
    );
  }
}

export const getDiffParams = (allParamKeys, runUuids, paramsByRunUuid) => {
  const diffParamKeys = [];
  allParamKeys.forEach((param) => {
    // collect all values for this param
    const paramVals = runUuids.map(
      (runUuid) => paramsByRunUuid[runUuid][param] && paramsByRunUuid[runUuid][param].value,
    );
    if (!paramVals.every((x, i, arr) => x === arr[0])) diffParamKeys.push(param);
  });
  return diffParamKeys;
};

const mapStateToProps = (state, ownProps) => {
  const { runUuids } = ownProps;
  const allParamKeys = getAllParamKeysByRunUuids(runUuids, state);
  const allMetricKeys = getAllMetricKeysByRunUuids(runUuids, state);
  const sharedMetricKeys = getSharedMetricKeysByRunUuids(runUuids, state);
  const { paramsByRunUuid } = state.entities;
  const diffParamKeys = getDiffParams(allParamKeys, runUuids, paramsByRunUuid);

  return {
    allParamKeys,
    allMetricKeys,
    sharedMetricKeys,
    diffParamKeys,
  };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotPanel);
