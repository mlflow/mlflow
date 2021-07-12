import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import ParallelCoordinatesPlotView from './ParallelCoordinatesPlotView';
import { ParallelCoordinatesPlotControls } from './ParallelCoordinatesPlotControls';
import {
  getAllParamKeysByRunUuids,
  getAllMetricKeysByRunUuids,
  getSharedMetricKeysByRunUuids,
  getSharedParamKeysByRunUuids,
} from '../reducers/Reducers';
import _ from 'lodash';
import { Empty } from 'antd';

import './ParallelCoordinatesPlotPanel.css';

export class ParallelCoordinatesPlotPanel extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of all parameter keys across runs
    allParamKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of all metric keys across runs
    allMetricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of parameter keys shared by all runs
    sharedParamKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of metric keys shared by all runs
    sharedMetricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
  };

  state = {
    // Default to select all parameters
    selectedParamKeys: this.props.sharedParamKeys,
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
      <div className='parallel-coorinates-plot-panel'>
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

const mapStateToProps = (state, ownProps) => {
  const { runUuids } = ownProps;
  const allParamKeys = getAllParamKeysByRunUuids(runUuids, state);
  const allMetricKeys = getAllMetricKeysByRunUuids(runUuids, state);
  const sharedParamKeys = getSharedParamKeysByRunUuids(runUuids, state);
  const sharedMetricKeys = getSharedMetricKeysByRunUuids(runUuids, state);
  return { allParamKeys, allMetricKeys, sharedParamKeys, sharedMetricKeys };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotPanel);
