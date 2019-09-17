import React from 'react';
import _ from 'lodash';
import { Radio, Switch, TreeSelect, Icon, Tooltip } from 'antd';
import PropTypes from 'prop-types';
import { CHART_TYPE_LINE } from './MetricsPlotPanel';
import { LineSmoothSlider } from './LineSmoothSlider';

const RadioGroup = Radio.Group;
export const X_AXIS_WALL = 'wall';
export const X_AXIS_STEP = 'step';
export const X_AXIS_RELATIVE = 'relative';

export class MetricsPlotControls extends React.Component {
  static propTypes = {
    // An array of distinct metric keys to be shown as options
    distinctMetricKeys: PropTypes.arrayOf(String).isRequired,
    // An array of metric keys selected by user or indicated by URL
    selectedMetricKeys: PropTypes.arrayOf(String).isRequired,
    selectedXAxis: PropTypes.string.isRequired,
    handleXAxisChange: PropTypes.func.isRequired,
    handleShowPointChange: PropTypes.func.isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
    handleYAxisLogScaleChange: PropTypes.func.isRequired,
    handleLineSmoothChange: PropTypes.func.isRequired,
    chartType: PropTypes.string.isRequired,
  };

  handleMetricsSelectFilterChange = (text, option) =>
    option.props.title.toUpperCase().includes(text.toUpperCase());

  getAllMetricKeys = () => {
    const { distinctMetricKeys } = this.props;
    return distinctMetricKeys.map((metricKey) => ({
      title: metricKey,
      value: metricKey,
      key: metricKey,
    }));
  };

  render() {
    const { chartType } = this.props;
    const lineSmoothnessTooltipText =
      'Make the line between points "smoother" based on generalized Catmull-Rom splines. ' +
      'Smoothing can be useful for displaying the ' +
      'overall trend when the logging frequency is high.';
    return (
      <div className='plot-controls'>
        {chartType === CHART_TYPE_LINE ? (
          <div>
            <div className='inline-control'>
              <div className='control-label'>Points:</div>
              <Switch
                className='show-point-toggle'
                checkedChildren='On'
                unCheckedChildren='Off'
                onChange={this.props.handleShowPointChange}
              />
            </div>
            <div className='block-control'>
              <div className='control-label'>
                Line Smoothness {' '}
                <Tooltip title={lineSmoothnessTooltipText}>
                  <Icon type='question-circle' />
                </Tooltip>
              </div>
              <LineSmoothSlider
                className='smoothness-toggle'
                min={0}
                max={1.3}
                handleLineSmoothChange={_.debounce(this.props.handleLineSmoothChange, 500)}
              />
            </div>
            <div className='block-control'>
              <div className='control-label'>X-axis:</div>
              <RadioGroup onChange={this.props.handleXAxisChange} value={this.props.selectedXAxis}>
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
          </div>
        ) : null}
        <div className='block-control'>
          <div className='control-label'>Y-axis:</div>
          <TreeSelect
            className='metrics-select'
            searchPlaceholder='Please select metric'
            value={this.props.selectedMetricKeys}
            showCheckedStrategy={TreeSelect.SHOW_PARENT}
            treeCheckable
            treeData={this.getAllMetricKeys()}
            onChange={this.props.handleMetricsSelectChange}
            filterTreeNode={this.handleMetricsSelectFilterChange}
          />
        </div>
        <div className='inline-control'>
          <div className='control-label'>Y-axis Log Scale:</div>
          <Switch
            checkedChildren='On'
            unCheckedChildren='Off'
            onChange={this.props.handleYAxisLogScaleChange}
          />
        </div>
      </div>
    );
  }
}
