import React from 'react';
import _ from 'lodash';
import { Radio, Switch, TreeSelect } from 'antd';
import PropTypes from 'prop-types';
import { CHART_TYPE_LINE } from './MetricsPlotPanel';
import { LineSmoothSlider } from './LineSmoothSlider';

const RadioGroup = Radio.Group;
export const X_AXIS_WALL = 'wall';
export const X_AXIS_STEP = 'step';
export const X_AXIS_RELATIVE = 'relative';

export class MetricsPlotControls extends React.Component {
  static propTypes = {
    allMetricKeys: PropTypes.arrayOf(String).isRequired,
    selectedMetricKeys: PropTypes.arrayOf(String).isRequired,
    selectedXAxis: PropTypes.string.isRequired,
    handleXAxisChange: PropTypes.func.isRequired,
    handleShowDotChange: PropTypes.func.isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
    handleYAxisLogScaleChange: PropTypes.func.isRequired,
    handleLineSmoothChange: PropTypes.func.isRequired,
    chartType: PropTypes.string.isRequired,
  };

  handleMetricsSelectFilterChange = (text, option) =>
    option.props.title.toUpperCase().includes(text.toUpperCase());

  render() {
    const { chartType } = this.props;
    return (
      <div className='plot-controls'>
        <h2>Plot Settings</h2>
        {chartType === CHART_TYPE_LINE ? (
          <div>
            <h3>Points:</h3>
            <Switch
              checkedChildren='On'
              unCheckedChildren='Off'
              onChange={this.props.handleShowDotChange}
            />
            <h3>Line Smoothness</h3>
            <LineSmoothSlider
              min={0}
              max={1.3}
              handleLineSmoothChange={_.debounce(this.props.handleLineSmoothChange, 500)}
            />
            <h3>X-axis:</h3>
            <RadioGroup
              onChange={this.props.handleXAxisChange}
              value={this.props.selectedXAxis}
            >
              <Radio className='x-axis-radio' value={X_AXIS_STEP}>
                Step
              </Radio>
              <Radio className='x-axis-radio' value={X_AXIS_WALL}>
                Time (Wall)
              </Radio>
              <Radio className='x-axis-radio' value={X_AXIS_RELATIVE}>
                Time (Relative)
              </Radio>
            </RadioGroup>
          </div>
        ) : null}
        <h3>Y-axis:</h3>
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
        <h3>Log Scale:</h3>
        <Switch
          checkedChildren='On'
          unCheckedChildren='Off'
          onChange={this.props.handleYAxisLogScaleChange}
        />
      </div>
    );
  }
}
