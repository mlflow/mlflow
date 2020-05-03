import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import { TreeSelect } from 'antd';
import _ from 'lodash';

export class ParallelCoordinatesPlotControls extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    // An array of available parameter keys to select
    paramKeys: PropTypes.arrayOf(String).isRequired,
    // An array of available metric keys to select
    metricKeys: PropTypes.arrayOf(String).isRequired,
    selectedParamKeys: PropTypes.arrayOf(String).isRequired,
    selectedMetricKeys: PropTypes.arrayOf(String).isRequired,
    sharedParamKeys: PropTypes.arrayOf(String).isRequired,
    missingParamKeys: PropTypes.arrayOf(String).isRequired,
    diffParamKeys: PropTypes.arrayOf(String).isRequired,
    constParamKeys: PropTypes.arrayOf(String).isRequired,
    handleParamsSelectChange: PropTypes.func.isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
  };

  static handleFilterChange = (text, option) =>
    option.props.title.toUpperCase().includes(text.toUpperCase());

  render() {
    const {
      metricKeys,
      selectedParamKeys,
      selectedMetricKeys,
      missingParamKeys,
      diffParamKeys,
      constParamKeys,
      handleParamsSelectChange,
      handleMetricsSelectChange,
    } = this.props;

    const keyToNode = (k) => ({
      title: k,
      value: k,
      label: k,
    });

    return (
      <div className='plot-controls'>
        <div>Parameters:</div>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder='Please select parameters'
          value={selectedParamKeys}
          showCheckedStrategy={TreeSelect.SHOW_CHILD}
          treeCheckable
          treeDefaultExpandAll
          treeData={[
            {
              title: 'Different Parameters',
              value: 'diff',
              key: 'diff',
              children: [...diffParamKeys, ...missingParamKeys].map(keyToNode),
            },
            {
              title: 'Constant Parameters',
              value: 'constant',
              key: 'constant',
              children: constParamKeys.map(keyToNode),
            },
          ]}
          onChange={handleParamsSelectChange}
          filterTreeNode={ParallelCoordinatesPlotControls.handleFilterChange}
        />
        <div style={{ marginTop: 20 }}>Metrics:</div>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder='Please select metrics'
          value={selectedMetricKeys}
          showCheckedStrategy={TreeSelect.SHOW_PARENT}
          treeCheckable
          treeData={metricKeys.map((k) => ({ title: k, value: k, label: k }))}
          onChange={handleMetricsSelectChange}
          filterTreeNode={ParallelCoordinatesPlotControls.handleFilterChange}
        />
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { runUuids, sharedParamKeys } = ownProps;
  const { paramsByRunUuid: params } = state.entities;

  const diffParamKeys = sharedParamKeys.filter((paramKey) => {
    return runUuids.some(
      (runUuid) => params[runUuid][paramKey].value !== params[runUuids[0]][paramKey].value,
    );
  });

  const constParamKeys = _.difference(sharedParamKeys, diffParamKeys);
  return { diffParamKeys, constParamKeys };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotControls);
