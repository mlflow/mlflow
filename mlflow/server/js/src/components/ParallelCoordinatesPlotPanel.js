import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import ParallelCoordinatesPlotView from './ParallelCoordinatesPlotView';
import { ParallelCoordinatesPlotControls } from './ParallelCoordinatesPlotControls';
import { getSharedMetricKeysByRunUuids, getSharedParamKeysByRunUuids } from '../reducers/Reducers';
import _ from 'lodash';
import rows from '../pcp.json';

import './ParallelCoordinatesPlotPanel.css';

class ParallelCoordinatesPlotPanel extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    sharedParamKeys: PropTypes.arrayOf(String).isRequired,
    sharedMetricKeys: PropTypes.arrayOf(String).isRequired,
  };

  static defaultProps = {
    sharedParamKeys: [],
    sharedMetricKeys: [],
  };

  state = {
    // TODO(Zangr) handle empty cases, show notice to user
    selectedParamKeys: this.props.sharedParamKeys, // default select all parameters
    selectedMetricKeys: this.props.sharedMetricKeys.slice(0, 1),
  };

  handleParamsSelectChange = (paramValues) => {
    this.setState({ selectedParamKeys: paramValues });
  };

  handleMetricsSelectChange = (metricValues) => {
    this.setState({ selectedMetricKeys: metricValues });
  };

  render() {
    // TODO(rzang) remove mock after testing
    const { runUuids, sharedParamKeys, sharedMetricKeys } = this.props;
    const { selectedParamKeys, selectedMetricKeys } = this.state;
    return (
      <div className='parallel-coorinates-plot-panel'>
        <ParallelCoordinatesPlotControls
          paramKeys={sharedParamKeys}
          metricKeys={sharedMetricKeys}
          selectedParamKeys={selectedParamKeys}
          selectedMetricKeys={selectedMetricKeys}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
          handleParamsSelectChange={this.handleParamsSelectChange}
        />
        {(!_.isEmpty(selectedParamKeys) || !_.isEmpty(selectedMetricKeys)) ? (
          <ParallelCoordinatesPlotView
            runUuids={runUuids}
            paramKeys={selectedParamKeys}
            metricKeys={selectedMetricKeys}
          />
        ) : null}
      </div>
    );
  }
}

// TODO(Zangr) remove mock after testing
const mockParamKeys = [
  'blockHeight',
  'blockWidth',
  'cycMaterial',
  'blockMaterial',
  'totalWeight',
  'blockHeight',
  'assemblyPW',
  'HstW',
];

// TODO(Zangr) remove mock after testing
const mockMetricKeys = ['minHW', 'minWD', 'rfBlock'];

// TODO(Zangr) remove mock after testing
const injectMockMetricsAndParamsIntoState = (state) => {
  const { latestMetricsByRunUuid, paramsByRunUuid } = state.entities;
  Object.keys(paramsByRunUuid).forEach((runUuid, i) => {
    mockParamKeys.forEach((paramsKey, j) => {
      const label = ['Category A', 'Category B', 'Category C', 'Category D'][Math.floor(Math.random() * 4)];
      const value = j === 2 ? label : rows[i][paramsKey];
      paramsByRunUuid[runUuid][`param_${j}`] = {
        key: `param_${j}`,
        value,
      };
    });
  });
  Object.keys(latestMetricsByRunUuid).forEach((runUuid, i) => {
    mockMetricKeys.forEach((metricKey, j) => {
      const label = ['Category A', 'Category B', 'Category C', 'Category D'][Math.floor(Math.random() * 4)];
      const value = j === 2 ? label : rows[i][metricKey];
      latestMetricsByRunUuid[runUuid][`metric_${j}`] = {
        key: `metric_${j}`,
        value,
      };
    });
  });
};

const mapStateToProps = (state, ownProps) => {
  // TODO(Zangr) remove mock after testing
  injectMockMetricsAndParamsIntoState(state);

  const { runUuids } = ownProps;
  const sharedParamKeys = getSharedParamKeysByRunUuids(runUuids, state);
  const sharedMetricKeys = getSharedMetricKeysByRunUuids(runUuids, state);
  return { sharedParamKeys, sharedMetricKeys };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotPanel);
