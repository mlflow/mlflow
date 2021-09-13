import React from 'react';
import _ from 'lodash';
import { Radio, Switch, TreeSelect, Icon, Tooltip } from 'antd';
import PropTypes from 'prop-types';
import { CHART_TYPE_LINE } from './MetricsPlotPanel';
import { LineSmoothSlider } from './LineSmoothSlider';

import { FormattedMessage, injectIntl } from 'react-intl';

const RadioGroup = Radio.Group;
export const X_AXIS_WALL = 'wall';
export const X_AXIS_STEP = 'step';
export const X_AXIS_RELATIVE = 'relative';
export const MAX_LINE_SMOOTHNESS = 100;

export class MetricsPlotControlsImpl extends React.Component {
  static propTypes = {
    // An array of distinct metric keys to be shown as options
    distinctMetricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    // An array of metric keys selected by user or indicated by URL
    selectedMetricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    selectedXAxis: PropTypes.string.isRequired,
    handleXAxisChange: PropTypes.func.isRequired,
    handleShowPointChange: PropTypes.func.isRequired,
    handleMetricsSelectChange: PropTypes.func.isRequired,
    handleYAxisLogScaleChange: PropTypes.func.isRequired,
    handleLineSmoothChange: PropTypes.func.isRequired,
    chartType: PropTypes.string.isRequired,
    initialLineSmoothness: PropTypes.number.isRequired,
    yAxisLogScale: PropTypes.bool.isRequired,
    showPoint: PropTypes.bool.isRequired,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
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
    const { chartType, yAxisLogScale, initialLineSmoothness, showPoint } = this.props;
    const wrapperStyle = chartType === CHART_TYPE_LINE ? styles.linechartControlsWrapper : {};
    const lineSmoothnessTooltipText = (
      <FormattedMessage
        // eslint-disable-next-line max-len
        defaultMessage='Make the line between points "smoother" based on Exponential Moving Average. Smoothing can be useful for displaying the overall trend when the logging frequency is high.'
        description='Helpful tooltip message to help with line smoothness for the metrics plot'
      />
    );
    return (
      <div className='plot-controls' style={wrapperStyle}>
        {chartType === CHART_TYPE_LINE ? (
          <div>
            <div className='inline-control'>
              <div className='control-label'>
                <FormattedMessage
                  defaultMessage='Points:'
                  // eslint-disable-next-line max-len
                  description='Label for the toggle button to toggle to show points or not for the metric experiment run'
                />
              </div>
              <Switch
                className='show-point-toggle'
                checkedChildren={this.props.intl.formatMessage({
                  defaultMessage: 'On',
                  description: 'Toggle on option to toggle show points for metric experiment run',
                })}
                unCheckedChildren={this.props.intl.formatMessage({
                  defaultMessage: 'Off',
                  description: 'Toggle off option to toggle show points for metric experiment run',
                })}
                defaultChecked={showPoint}
                onChange={this.props.handleShowPointChange}
              />
            </div>
            <div className='block-control'>
              <div className='control-label'>
                <FormattedMessage
                  defaultMessage='Line Smoothness'
                  description='Label for the smoothness slider for the graph plot for metrics'
                />{' '}
                <Tooltip title={lineSmoothnessTooltipText}>
                  <Icon type='question-circle' />
                </Tooltip>
              </div>
              <LineSmoothSlider
                className='smoothness-toggle'
                min={1}
                max={MAX_LINE_SMOOTHNESS}
                handleLineSmoothChange={_.debounce(this.props.handleLineSmoothChange, 100)}
                defaultValue={initialLineSmoothness}
              />
            </div>
            <div className='block-control'>
              <div className='control-label'>
                <FormattedMessage
                  defaultMessage='X-axis:'
                  // eslint-disable-next-line max-len
                  description='Label for the radio button to toggle the control on the X-axis of the metric graph for the experiment'
                />
              </div>
              <RadioGroup onChange={this.props.handleXAxisChange} value={this.props.selectedXAxis}>
                <Radio className='x-axis-radio' value={X_AXIS_STEP}>
                  <FormattedMessage
                    defaultMessage='Step'
                    // eslint-disable-next-line max-len
                    description='Radio button option to choose the step control option for the X-axis for metric graph on the experiment runs'
                  />
                </Radio>
                <Radio className='x-axis-radio' value={X_AXIS_WALL}>
                  <FormattedMessage
                    defaultMessage='Time (Wall)'
                    // eslint-disable-next-line max-len
                    description='Radio button option to choose the time wall control option for the X-axis for metric graph on the experiment runs'
                  />
                </Radio>
                <Radio className='x-axis-radio' value={X_AXIS_RELATIVE}>
                  <FormattedMessage
                    defaultMessage='Time (Relative)'
                    // eslint-disable-next-line max-len
                    description='Radio button option to choose the time relative control option for the X-axis for metric graph on the experiment runs'
                  />
                </Radio>
              </RadioGroup>
            </div>
          </div>
        ) : null}
        <div className='block-control'>
          <div className='control-label'>
            <FormattedMessage
              defaultMessage='Y-axis:'
              // eslint-disable-next-line max-len
              description='Label where the users can choose the metric of the experiment run to be plotted on the Y-axis'
            />
          </div>
          <TreeSelect
            className='metrics-select'
            searchPlaceholder={this.props.intl.formatMessage({
              defaultMessage: 'Please select metric',
              description:
                // eslint-disable-next-line max-len
                'Placeholder text where one can select metrics from the list of available metrics to render on the graph',
            })}
            value={this.props.selectedMetricKeys}
            showCheckedStrategy={TreeSelect.SHOW_PARENT}
            treeCheckable
            treeData={this.getAllMetricKeys()}
            onChange={this.props.handleMetricsSelectChange}
            filterTreeNode={this.handleMetricsSelectFilterChange}
          />
        </div>
        <div className='inline-control'>
          <div className='control-label'>
            <FormattedMessage
              defaultMessage='Y-axis Log Scale:'
              // eslint-disable-next-line max-len
              description='Label for the radio button to toggle the Log scale on the Y-axis of the metric graph for the experiment'
            />
          </div>
          <Switch
            checkedChildren={this.props.intl.formatMessage({
              defaultMessage: 'On',
              description: 'Toggle on option to toggle log scale graph for metric experiment run',
            })}
            unCheckedChildren={this.props.intl.formatMessage({
              defaultMessage: 'Off',
              description: 'Toggle off option to toggle log scale graph for metric experiment run',
            })}
            defaultChecked={yAxisLogScale}
            onChange={this.props.handleYAxisLogScaleChange}
          />
        </div>
      </div>
    );
  }
}

const styles = {
  linechartControlsWrapper: {
    // Make controls aligned to plotly line chart
    justifyContent: 'center',
  },
};

export const MetricsPlotControls = injectIntl(MetricsPlotControlsImpl);
