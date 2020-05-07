import React from 'react';
import PropTypes from 'prop-types';
import { Checkbox, TreeSelect } from 'antd';

export class ParallelCoordinatesPlotControls extends React.Component {
  static propTypes = {
    // An array of available parameter keys to select
    paramKeys: PropTypes.arrayOf(String).isRequired,
    // An array of available metric keys to select
    metricKeys: PropTypes.arrayOf(String).isRequired,
    selectedParamKeys: PropTypes.arrayOf(String).isRequired,
    selectedMetricKeys: PropTypes.arrayOf(String).isRequired,
    sharedParamKeys: PropTypes.arrayOf(String).isRequired,
    handleParamsSelectChange: PropTypes.func.isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
    handleSelectAllChange: PropTypes.func.isRequired,
    handleSelectDiffChange: PropTypes.func.isRequired,
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
      handleSelectAllChange,
      handleSelectDiffChange,
      selectAll,
      selectDiff,
    } = this.props;

    return (
      <div className='plot-controls'>
        <div>Parameters:</div>
        <div style={{ marginTop: 5, marginBottom: 5 }}>
          <Checkbox
            indeterminate={selectAll.indeterminate}
            checked={selectAll.checked}
            onChange={handleSelectAllChange}
          >
            Select All
          </Checkbox>
          <Checkbox
            indeterminate={selectDiff.indeterminate}
            checked={selectDiff.checked}
            disabled={selectDiff.disabled}
            onChange={handleSelectDiffChange}
          >
            Select Diff Only
          </Checkbox>
        </div>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder='Please select parameters'
          value={selectedParamKeys}
          showCheckedStrategy={TreeSelect.SHOW_PARENT}
          treeCheckable
          treeData={paramKeys.map((k) => ({ title: k, value: k, label: k }))}
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
