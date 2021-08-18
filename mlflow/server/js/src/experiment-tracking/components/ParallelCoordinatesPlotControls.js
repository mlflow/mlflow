import React from 'react';
import PropTypes from 'prop-types';
import { TreeSelect } from 'antd';
import { FormattedMessage } from 'react-intl';

export class ParallelCoordinatesPlotControls extends React.Component {
  static propTypes = {
    // An array of available parameter keys to select
    paramKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of available metric keys to select
    metricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    selectedParamKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    selectedMetricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    handleParamsSelectChange: PropTypes.func.isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
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
    } = this.props;
    return (
      <div className='plot-controls'>
        <div>
          <FormattedMessage
            defaultMessage='Parameters:'
            description='Label text for parameters in parallel coordinates plot in MLflow'
          />
        </div>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder={
            <FormattedMessage
              defaultMessage='Please select parameters'
              description='Placeholder text for parameters in parallel coordinates plot in MLflow'
            />
          }
          value={selectedParamKeys}
          showCheckedStrategy={TreeSelect.SHOW_PARENT}
          treeCheckable
          treeData={paramKeys.map((k) => ({ title: k, value: k, label: k }))}
          onChange={handleParamsSelectChange}
          filterTreeNode={ParallelCoordinatesPlotControls.handleFilterChange}
        />
        <div style={{ marginTop: 20 }}>
          <FormattedMessage
            defaultMessage='Metrics:'
            description='Label text for metrics in parallel coordinates plot in MLflow'
          />
        </div>
        <TreeSelect
          className='metrics-select'
          searchPlaceholder={
            <FormattedMessage
              defaultMessage='Please select metrics'
              description='Placeholder text for metrics in parallel coordinates plot in MLflow'
            />
          }
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
