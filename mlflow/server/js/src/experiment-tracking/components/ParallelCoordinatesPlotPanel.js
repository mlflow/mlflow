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
    runUuids: PropTypes.arrayOf(String).isRequired,
    // An array of all parameter keys across runs
    allParamKeys: PropTypes.arrayOf(String).isRequired,
    // An array of all metric keys across runs
    allMetricKeys: PropTypes.arrayOf(String).isRequired,
    // An array of parameter keys shared by all runs
    sharedParamKeys: PropTypes.arrayOf(String).isRequired,
    // An array of metric keys shared by all runs
    sharedMetricKeys: PropTypes.arrayOf(String).isRequired,
    // A subset of sharedParamKeys where the values of the parameters differ between runs
    diffParamKeys: PropTypes.arrayOf(String).isRequired,
  };

  constructor(props) {
    super(props);
    const { sharedParamKeys, sharedMetricKeys } = props;

    this.state = {
      // Default to select all parameters
      selectedParamKeys: sharedParamKeys,
      // Default to select the first metric key.
      // Note that there will be no color scaling if no metric is selected.
      selectedMetricKeys: sharedMetricKeys.slice(0, 1),
      selectAll: this.getSelectAllState(sharedParamKeys),
      selectDiff: this.getSelectDiffState(sharedParamKeys),
    };
  }

  getSelectAllState = (selectedParamKeys) => {
    const { allParamKeys } = this.props;

    const allParamsSelected = _.isEqual(selectedParamKeys, allParamKeys);
    return {
      indeterminate: !allParamsSelected && selectedParamKeys.length > 0,
      checked: allParamsSelected,
    };
  };

  getSelectDiffState = (selectedParamKeys) => {
    const { diffParamKeys } = this.props;

    const allDiffParamsSelected = _.isEqual(selectedParamKeys, diffParamKeys);
    const diffParamsSelected = _.intersection(selectedParamKeys, diffParamKeys).length > 0;
    return {
      indeterminate: !allDiffParamsSelected && diffParamsSelected,
      checked: allDiffParamsSelected,
      disabled: diffParamKeys.length === 0,
    };
  };

  handleParamsSelectChange = (paramValues) => {
    this.setState({
      selectedParamKeys: paramValues,
      selectAll: this.getSelectAllState(paramValues),
      selectDiff: this.getSelectDiffState(paramValues),
    });
  };

  handleMetricsSelectChange = (metricValues) => {
    this.setState({ selectedMetricKeys: metricValues });
  };

  handleSelectAllChange = () => {
    const { indeterminate, checked } = this.state.selectAll;
    const { allParamKeys } = this.props;

    if (indeterminate) {
      this.setState({
        selectedParamKeys: allParamKeys,
        selectAll: { indeterminate: false, checked: true },
        selectDiff: this.getSelectDiffState(allParamKeys),
      });
    } else if (checked) {
      this.setState({
        selectedParamKeys: [],
        selectAll: { checked: false },
        selectDiff: this.getSelectDiffState([]),
      });
    } else {
      this.setState({
        selectedParamKeys: allParamKeys,
        selectAll: { checked: true },
        selectDiff: this.getSelectDiffState(allParamKeys),
      });
    }
  };

  handleSelectDiffChange = () => {
    const { indeterminate, checked } = this.state.selectDiff;
    const { diffParamKeys } = this.props;

    if (indeterminate) {
      this.setState({
        selectedParamKeys: diffParamKeys,
        selectAll: this.getSelectAllState(diffParamKeys),
        selectDiff: { indeterminate: false, checked: true },
      });
    } else if (checked) {
      this.setState({
        selectedParamKeys: [],
        selectAll: this.getSelectAllState([]),
        selectDiff: { checked: false },
      });
    } else {
      this.setState({
        selectedParamKeys: diffParamKeys,
        selectAll: this.getSelectAllState(diffParamKeys),
        selectDiff: { checked: true },
      });
    }
  };

  render() {
    const { runUuids, allParamKeys, allMetricKeys, sharedParamKeys } = this.props;
    const { selectedParamKeys, selectedMetricKeys, selectAll, selectDiff } = this.state;
    return (
      <div className='parallel-coorinates-plot-panel'>
        <ParallelCoordinatesPlotControls
          paramKeys={allParamKeys}
          metricKeys={allMetricKeys}
          selectedParamKeys={selectedParamKeys}
          sharedParamKeys={sharedParamKeys}
          selectedMetricKeys={selectedMetricKeys}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
          handleParamsSelectChange={this.handleParamsSelectChange}
          handleSelectAllChange={this.handleSelectAllChange}
          handleSelectDiffChange={this.handleSelectDiffChange}
          selectAll={selectAll}
          selectDiff={selectDiff}
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
  const missingParamKeys = _.difference(allParamKeys, sharedParamKeys);

  const { paramsByRunUuid } = state.entities;
  const diffParamKeys = sharedParamKeys.filter((paramKey) => {
    return runUuids.some(
      (runUuid) =>
        paramsByRunUuid[runUuid][paramKey].value !== paramsByRunUuid[runUuids[0]][paramKey].value,
    );
  });

  return {
    allParamKeys,
    allMetricKeys,
    sharedParamKeys,
    sharedMetricKeys,
    diffParamKeys: [...missingParamKeys, ...diffParamKeys],
  };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotPanel);
