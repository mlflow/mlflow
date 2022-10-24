import React from 'react';
import _ from 'lodash';
import {
  Button,
  Select,
  Switch,
  Tooltip,
  Radio,
  QuestionMarkBorderIcon,
} from '@databricks/design-system';
import { Progress } from '../../common/components/Progress';
import PropTypes from 'prop-types';
import { CHART_TYPE_LINE, METRICS_PLOT_POLLING_INTERVAL_MS } from './MetricsPlotPanel';
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
    numRuns: PropTypes.number.isRequired,
    numCompletedRuns: PropTypes.number.isRequired,
    handleDownloadCsv: PropTypes.func.isRequired,
    disableSmoothnessControl: PropTypes.bool.isRequired,
  };

  static defaultProps = {
    disableSmoothnessControl: false,
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
    const {
      chartType,
      yAxisLogScale,
      initialLineSmoothness,
      showPoint,
      numRuns,
      numCompletedRuns,
      disableSmoothnessControl,
    } = this.props;

    const lineSmoothnessTooltipText = (
      <FormattedMessage
        // eslint-disable-next-line max-len
        defaultMessage='Make the line between points "smoother" based on Exponential Moving Average. Smoothing can be useful for displaying the overall trend when the logging frequency is high.'
        description='Helpful tooltip message to help with line smoothness for the metrics plot'
      />
    );
    const completedRunsTooltipText = (
      <FormattedMessage
        // eslint-disable-next-line max-len
        defaultMessage='MLflow UI automatically fetches metric histories for active runs and updates the metrics plot with a {interval} second interval.'
        description='Helpful tooltip message to explain the automatic metrics plot update'
        values={{ interval: Math.round(METRICS_PLOT_POLLING_INTERVAL_MS / 1000) }}
      />
    );
    return (
      <div
        className='plot-controls'
        css={[
          styles.controlsWrapper,
          chartType === CHART_TYPE_LINE && styles.centeredControlsWrapper,
        ]}
      >
        {chartType === CHART_TYPE_LINE ? (
          <div>
            <div className='inline-control'>
              <div className='control-label'>
                <FormattedMessage
                  defaultMessage='Completed Runs'
                  description='Label for the progress bar to show the number of completed runs'
                />{' '}
                <Tooltip title={completedRunsTooltipText}>
                  <QuestionMarkBorderIcon />
                </Tooltip>
                <Progress
                  percent={Math.round((100 * numCompletedRuns) / numRuns)}
                  format={() => `${numCompletedRuns}/${numRuns}`}
                />
              </div>
            </div>
            <div className='inline-control'>
              <div className='control-label'>
                <FormattedMessage
                  defaultMessage='Points:'
                  // eslint-disable-next-line max-len
                  description='Label for the toggle button to toggle to show points or not for the metric experiment run'
                />
              </div>
              <Switch
                data-testid='show-point-toggle'
                defaultChecked={showPoint}
                onChange={this.props.handleShowPointChange}
              />
            </div>
            {!disableSmoothnessControl && (
              <div className='block-control'>
                <div className='control-label'>
                  <FormattedMessage
                    defaultMessage='Line Smoothness'
                    description='Label for the smoothness slider for the graph plot for metrics'
                  />{' '}
                  <Tooltip title={lineSmoothnessTooltipText}>
                    <QuestionMarkBorderIcon />
                  </Tooltip>
                </div>
                <LineSmoothSlider
                  data-testid='smoothness-toggle'
                  min={1}
                  max={MAX_LINE_SMOOTHNESS}
                  handleLineSmoothChange={_.debounce(this.props.handleLineSmoothChange, 100)}
                  defaultValue={initialLineSmoothness}
                />
              </div>
            )}
            <div className='block-control'>
              <div className='control-label'>
                <FormattedMessage
                  defaultMessage='X-axis:'
                  // eslint-disable-next-line max-len
                  description='Label for the radio button to toggle the control on the X-axis of the metric graph for the experiment'
                />
              </div>
              <RadioGroup
                css={styles.xAxisControls}
                onChange={this.props.handleXAxisChange}
                value={this.props.selectedXAxis}
              >
                <Radio value={X_AXIS_STEP} data-testid='x-axis-radio'>
                  <FormattedMessage
                    defaultMessage='Step'
                    // eslint-disable-next-line max-len
                    description='Radio button option to choose the step control option for the X-axis for metric graph on the experiment runs'
                  />
                </Radio>
                <Radio value={X_AXIS_WALL} data-testid='x-axis-radio'>
                  <FormattedMessage
                    defaultMessage='Time (Wall)'
                    // eslint-disable-next-line max-len
                    description='Radio button option to choose the time wall control option for the X-axis for metric graph on the experiment runs'
                  />
                </Radio>
                <Radio value={X_AXIS_RELATIVE} data-testid='x-axis-radio'>
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
          <Select
            placeholder={this.props.intl.formatMessage({
              defaultMessage: 'Please select metric',
              description:
                // eslint-disable-next-line max-len
                'Placeholder text where one can select metrics from the list of available metrics to render on the graph',
            })}
            value={this.props.selectedMetricKeys}
            onChange={this.props.handleMetricsSelectChange}
            mode='multiple'
            css={styles.axisSelector}
          >
            {this.getAllMetricKeys().map((key) => (
              <Select.Option value={key.value} key={key.key}>
                {key.title}
              </Select.Option>
            ))}
          </Select>
        </div>
        <div className='inline-control'>
          <div className='control-label'>
            <FormattedMessage
              defaultMessage='Y-axis Log Scale:'
              // eslint-disable-next-line max-len
              description='Label for the radio button to toggle the Log scale on the Y-axis of the metric graph for the experiment'
            />
          </div>
          <Switch defaultChecked={yAxisLogScale} onChange={this.props.handleYAxisLogScaleChange} />
        </div>
        <div className='inline-control'>
          <Button
            css={{
              textAlign: 'justify',
              textAlignLast: 'left',
            }}
            onClick={this.props.handleDownloadCsv}
          >
            <FormattedMessage
              defaultMessage='Download CSV'
              // eslint-disable-next-line max-len
              description='String for the download csv button to download metrics from this run offline in a CSV format'
            />
            <i className='fas fa-download' />
          </Button>
        </div>
      </div>
    );
  }
}

const styles = {
  xAxisControls: (theme) => ({
    label: { marginTop: theme.spacing.xs, marginBottom: theme.spacing.xs },
  }),
  controlsWrapper: { minWidth: '20%', maxWidth: '30%' },
  axisSelector: { width: '100%' },
  centeredControlsWrapper: {
    // Make controls aligned to plotly line chart
    justifyContent: 'center',
  },
};

export const MetricsPlotControls = injectIntl(MetricsPlotControlsImpl);
