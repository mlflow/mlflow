/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Button, LegacySelect, Switch, LegacyTooltip, Radio, QuestionMarkIcon } from '@databricks/design-system';
import { Progress } from '../../common/components/Progress';
import { CHART_TYPE_LINE } from './MetricsPlotPanel';
import { EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL } from '../utils/MetricsUtils';

import { FormattedMessage, injectIntl } from 'react-intl';
import { LineSmoothSlider } from './LineSmoothSlider';

const RadioGroup = Radio.Group;
export const X_AXIS_WALL = 'wall';
export const X_AXIS_STEP = 'step';
export const X_AXIS_RELATIVE = 'relative';
export const MAX_LINE_SMOOTHNESS = 100;

type Props = {
  distinctMetricKeys: string[];
  selectedMetricKeys: string[];
  selectedXAxis: string;
  handleXAxisChange: (...args: any[]) => any;
  handleShowPointChange: (...args: any[]) => any;
  handleMetricsSelectChange: (...args: any[]) => any;
  handleYAxisLogScaleChange: (...args: any[]) => any;
  handleLineSmoothChange: (value: number) => void;
  chartType: string;
  lineSmoothness: number;
  yAxisLogScale: boolean;
  showPoint: boolean;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
  numRuns: number;
  numCompletedRuns: number;
  handleDownloadCsv: (...args: any[]) => any;
  disableSmoothnessControl: boolean;
};

class MetricsPlotControlsImpl extends React.Component<Props> {
  static defaultProps = {
    disableSmoothnessControl: false,
  };

  handleMetricsSelectFilterChange = (text: any, option: any) =>
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
    const { chartType, yAxisLogScale, lineSmoothness, showPoint, numRuns, numCompletedRuns, disableSmoothnessControl } =
      this.props;

    const lineSmoothnessTooltipText = (
      <FormattedMessage
        // eslint-disable-next-line max-len
        defaultMessage='Make the line between points "smoother" based on Exponential Moving Average. Smoothing can be useful for displaying the overall trend when the logging frequency is high.'
        description="Helpful tooltip message to help with line smoothness for the metrics plot"
      />
    );
    const completedRunsTooltipText = (
      <FormattedMessage
        // eslint-disable-next-line max-len
        defaultMessage="MLflow UI automatically fetches metric histories for active runs and updates the metrics plot with a {interval} second interval."
        description="Helpful tooltip message to explain the automatic metrics plot update"
        values={{ interval: Math.round(EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL / 1000) }}
      />
    );
    return (
      <div
        className="plot-controls"
        css={[styles.controlsWrapper, chartType === CHART_TYPE_LINE && styles.centeredControlsWrapper]}
      >
        {chartType === CHART_TYPE_LINE ? (
          <div>
            <div className="inline-control">
              <div className="control-label">
                <FormattedMessage
                  defaultMessage="Completed Runs"
                  description="Label for the progress bar to show the number of completed runs"
                />{' '}
                <LegacyTooltip title={completedRunsTooltipText}>
                  <QuestionMarkIcon />
                </LegacyTooltip>
                <Progress
                  percent={Math.round((100 * numCompletedRuns) / numRuns)}
                  format={() => `${numCompletedRuns}/${numRuns}`}
                />
              </div>
            </div>
            <div className="inline-control">
              <div className="control-label">
                <FormattedMessage
                  defaultMessage="Points:"
                  // eslint-disable-next-line max-len
                  description="Label for the toggle button to toggle to show points or not for the metric experiment run"
                />
              </div>
              <Switch
                componentId="codegen_mlflow_app_src_experiment-tracking_components_metricsplotcontrols.tsx_120"
                data-testid="show-point-toggle"
                defaultChecked={showPoint}
                onChange={this.props.handleShowPointChange}
              />
            </div>
            {!disableSmoothnessControl && (
              <div className="block-control">
                <div className="control-label">
                  <FormattedMessage
                    defaultMessage="Line Smoothness"
                    description="Label for the smoothness slider for the graph plot for metrics"
                  />{' '}
                  <LegacyTooltip title={lineSmoothnessTooltipText}>
                    <QuestionMarkIcon />
                  </LegacyTooltip>
                </div>
                <LineSmoothSlider
                  data-testid="smoothness-toggle"
                  min={1}
                  max={MAX_LINE_SMOOTHNESS}
                  onChange={this.props.handleLineSmoothChange}
                  value={lineSmoothness}
                />
              </div>
            )}
            <div className="block-control">
              <div className="control-label">
                <FormattedMessage
                  defaultMessage="X-axis:"
                  // eslint-disable-next-line max-len
                  description="Label for the radio button to toggle the control on the X-axis of the metric graph for the experiment"
                />
              </div>
              <RadioGroup
                componentId="codegen_mlflow_app_src_experiment-tracking_components_metricsplotcontrols.tsx_154"
                name="metrics-plot-x-axis-radio-group"
                css={styles.xAxisControls}
                onChange={this.props.handleXAxisChange}
                value={this.props.selectedXAxis}
              >
                <Radio value={X_AXIS_STEP} data-testid="x-axis-radio">
                  <FormattedMessage
                    defaultMessage="Step"
                    // eslint-disable-next-line max-len
                    description="Radio button option to choose the step control option for the X-axis for metric graph on the experiment runs"
                  />
                </Radio>
                <Radio value={X_AXIS_WALL} data-testid="x-axis-radio">
                  <FormattedMessage
                    defaultMessage="Time (Wall)"
                    // eslint-disable-next-line max-len
                    description="Radio button option to choose the time wall control option for the X-axis for metric graph on the experiment runs"
                  />
                </Radio>
                <Radio value={X_AXIS_RELATIVE} data-testid="x-axis-radio">
                  <FormattedMessage
                    defaultMessage="Time (Relative)"
                    // eslint-disable-next-line max-len
                    description="Radio button option to choose the time relative control option for the X-axis for metric graph on the experiment runs"
                  />
                </Radio>
              </RadioGroup>
            </div>
          </div>
        ) : null}
        <div className="block-control">
          <div className="control-label">
            <FormattedMessage
              defaultMessage="Y-axis:"
              // eslint-disable-next-line max-len
              description="Label where the users can choose the metric of the experiment run to be plotted on the Y-axis"
            />
          </div>
          <LegacySelect
            placeholder={this.props.intl.formatMessage({
              defaultMessage: 'Please select metric',
              description:
                // eslint-disable-next-line max-len
                'Placeholder text where one can select metrics from the list of available metrics to render on the graph',
            })}
            value={this.props.selectedMetricKeys}
            onChange={this.props.handleMetricsSelectChange}
            mode="multiple"
            css={styles.axisSelector}
          >
            {this.getAllMetricKeys().map((key) => (
              <LegacySelect.Option value={key.value} key={key.key}>
                {key.title}
              </LegacySelect.Option>
            ))}
          </LegacySelect>
        </div>
        <div className="inline-control">
          <div className="control-label">
            <FormattedMessage
              defaultMessage="Y-axis Log Scale:"
              // eslint-disable-next-line max-len
              description="Label for the radio button to toggle the Log scale on the Y-axis of the metric graph for the experiment"
            />
          </div>
          <Switch
            componentId="codegen_mlflow_app_src_experiment-tracking_components_metricsplotcontrols.tsx_220"
            defaultChecked={yAxisLogScale}
            onChange={this.props.handleYAxisLogScaleChange}
          />
        </div>
        <div className="inline-control">
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_metricsplotcontrols.tsx_222"
            css={{
              textAlign: 'justify',
              textAlignLast: 'left',
            }}
            onClick={this.props.handleDownloadCsv}
          >
            <FormattedMessage
              defaultMessage="Download data"
              // eslint-disable-next-line max-len
              description="String for the download csv button to download metrics from this run offline in a CSV format"
            />
            <i className="fa fa-download" />
          </Button>
        </div>
      </div>
    );
  }
}

const styles = {
  xAxisControls: (theme: any) => ({
    label: { marginTop: theme.spacing.xs, marginBottom: theme.spacing.xs },
  }),
  controlsWrapper: { minWidth: '20%', maxWidth: '30%' },
  axisSelector: { width: '100%' },
  centeredControlsWrapper: {
    // Make controls aligned to plotly line chart
    justifyContent: 'center',
  },
};

// @ts-expect-error TS(2769): No overload matches this call.
export const MetricsPlotControls = injectIntl(MetricsPlotControlsImpl);
