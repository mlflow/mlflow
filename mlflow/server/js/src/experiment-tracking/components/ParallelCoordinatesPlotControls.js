import React from 'react';
import PropTypes from 'prop-types';
import { Checkbox, TreeSelect } from 'antd';

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
    handleParamsSelectChange: PropTypes.func.isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
    handleSelectAll: PropTypes.func.isRequired,
    handleSelectDiff: PropTypes.func.isRequired,
    selectAll: PropTypes.object.isRequired,
    selectDiff: PropTypes.object.isRequired,
  };

  static handleFilterChange = (text, option) =>
    option.props.title.toUpperCase().includes(text.toUpperCase());

  render() {
    const {
      paramKeys,
      metricKeys,
      selectedParamKeys,
      selectedMetricKeys,
      handleParamsSelectChange,
      handleMetricsSelectChange,
      handleSelectAll,
      handleSelectDiff,
      selectAll,
      selectDiff,
    } = this.props;

    const keyToNode = (k) => ({
      title: k,
      value: k,
      label: k,
    });

    return (
      <div className='plot-controls'>
        <div>Parameters:</div>
        <div style={{ marginTop: 5, marginBottom: 5 }}>
          <Checkbox
            indeterminate={selectAll.indeterminate}
            onChange={handleSelectAll}
            checked={selectAll.checked}
          >
            Select All
          </Checkbox>
          <Checkbox
            indeterminate={selectDiff.indeterminate}
            onChange={handleSelectDiff}
            checked={selectDiff.checked}
            disabled={selectDiff.disabled}
          >
            Select Diff Only
          </Checkbox>
        </div>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder='Please select parameters'
          value={selectedParamKeys}
          showCheckedStrategy={TreeSelect.SHOW_CHILD}
          treeCheckable
          treeDefaultExpandAll
          treeData={paramKeys.map(keyToNode)}
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
