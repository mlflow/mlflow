import React from 'react';
import PropTypes from 'prop-types';
import { TreeSelect } from 'antd';

export class ParallelCoordinatesPlotControl extends React.Component {
  static propTypes = {
    allMetricKeys: PropTypes.arrayOf(String).isRequired,
    selectedMetricKeys: PropTypes.arrayOf(String).isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
  };

  handleMetricsSelectFilterChange = (text, option) =>
    option.props.title.toUpperCase().includes(text.toUpperCase());

  render() {
    return (
      <div className='plot-controls'>
        <h3>Metrics:</h3>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder='Please select metric'
          value={this.props.selectedMetricKeys}
          showCheckedStrategy={TreeSelect.SHOW_PARENT}
          treeCheckable
          treeData={this.props.allMetricKeys}
          onChange={this.props.handleMetricsSelectChange}
          filterTreeNode={this.handleMetricsSelectFilterChange}
        />
      </div>
    );
  }
}
