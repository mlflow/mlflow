import React from 'react';
import PropTypes from 'prop-types';
import { TreeSelect } from 'antd';

export class ParallelCoordinatesPlotControls extends React.Component {
  static propTypes = {
    paramKeys: PropTypes.arrayOf(String).isRequired,
    metricKeys: PropTypes.arrayOf(String).isRequired,
    selectedParamKeys: PropTypes.arrayOf(String).isRequired,
    selectedMetricKeys: PropTypes.arrayOf(String).isRequired,
    handleParamsSelectChange: PropTypes.func.isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
  };

  handleFilterChange = (text, option) =>
    option.props.title.toUpperCase().includes(text.toUpperCase());

  render() {
    const {
      paramKeys,
      metricKeys,
      selectedParamKeys,
      selectedMetricKeys,
      handleParamsSelectChange,
      handleMetricsSelectChange
    } = this.props;
    return (
      <div className='plot-controls'>
        <h3>Parameters:</h3>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder='Please select parameters'
          value={selectedParamKeys}
          showCheckedStrategy={TreeSelect.SHOW_PARENT}
          treeCheckable
          treeData={paramKeys.map((k) => ({ title: k, value: k, label: k}))}
          onChange={handleParamsSelectChange}
          filterTreeNode={this.handleFilterChange}
        />
        {/* TODO(Zangr) remove BR */}
        <br/><br/><br/>
        <h3>Metrics:</h3>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder='Please select metrics'
          value={selectedMetricKeys}
          showCheckedStrategy={TreeSelect.SHOW_PARENT}
          treeCheckable
          treeData={metricKeys.map((k) => ({ title: k, value: k, label: k}))}
          onChange={handleMetricsSelectChange}
          filterTreeNode={this.handleFilterChange}
        />
      </div>
    );
  }
}
