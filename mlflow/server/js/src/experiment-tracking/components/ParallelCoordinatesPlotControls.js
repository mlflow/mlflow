import React from 'react';
import PropTypes from 'prop-types';
import { Button } from '../../shared/building_blocks/Button';
import { Spacer } from '../../shared/building_blocks/Spacer';
import { TreeSelect } from 'antd';
import { FormattedMessage } from 'react-intl';
import { FlexBar } from '../../shared/building_blocks/FlexBar';

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
      onClearParamsSelect,
    } = this.props;
    return (
      <div className='plot-controls'>
        <div>
          <FlexBar
            left={
              <Spacer size='small' direction='horizontal'>
                <FormattedMessage
                  defaultMessage='Parameters:'
                  description='Label text for parameters in parallel coordinates plot in MLflow'
                />
              </Spacer>
            }
            right={
              <Spacer size='small' direction='horizontal'>
                <Button dataTestId='clear-button' onClick={onClearParamsSelect}>
                  <FormattedMessage
                    defaultMessage='Clear All'
                    description='String for the clear button to clear any selected parameters'
                  />
                </Button>
              </Spacer>
            }
          />
        </div>
        <TreeSelect
          className='metrics-select'
          placeholder={
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
          placeholder={
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
