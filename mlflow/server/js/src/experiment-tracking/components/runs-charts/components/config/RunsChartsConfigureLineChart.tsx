import {
  Radio,
  LegacySelect,
  Switch,
  Tooltip,
  QuestionMarkIcon,
  Form,
  useDesignSystemTheme,
  ThemeType,
  SegmentedControlGroup,
  SegmentedControlButton,
  InfoIcon,
  Input,
  FormUI,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCallback, useEffect, useState } from 'react';
import type { ChartRange, RunsChartsCardConfig, RunsChartsLineCardConfig } from '../../runs-charts.types';
import { RunsChartsConfigureField, RunsChartsRunNumberSelect } from './RunsChartsConfigure.common';
import {
  shouldEnableDeepLearningUIPhase3,
  shouldEnableManualRangeControls,
} from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { RunsChartsLineChartXAxisType } from '@mlflow/mlflow/src/experiment-tracking/components/runs-charts/components/RunsCharts.common';
import { LineSmoothSlider } from '@mlflow/mlflow/src/experiment-tracking/components/LineSmoothSlider';
import { isUndefined } from 'lodash';

const renderMetricSelectorV1 = ({
  metricKeyList,
  metricKey,
  updateMetric,
}: {
  metricKeyList: string[];
  metricKey?: string;
  updateMetric: (metricKey: string) => void;
}) => {
  const emptyMetricsList = metricKeyList.length === 0;

  return (
    <LegacySelect
      css={styles.selectFull}
      value={emptyMetricsList ? 'No metrics available' : metricKey}
      onChange={updateMetric}
      disabled={emptyMetricsList}
      dangerouslySetAntdProps={{ showSearch: true }}
    >
      {metricKeyList.map((metric) => (
        <LegacySelect.Option key={metric} value={metric} data-testid={`metric-${metric}`}>
          {metric}
        </LegacySelect.Option>
      ))}
    </LegacySelect>
  );
};

const renderMetricSelectorV2 = ({
  metricKeyList,
  selectedMetricKeys,
  updateSelectedMetrics,
}: {
  metricKeyList: string[];
  selectedMetricKeys?: string[];
  updateSelectedMetrics: (metricKeys: string[]) => void;
}) => {
  const emptyMetricsList = metricKeyList.length === 0;

  return (
    <LegacySelect
      mode="multiple"
      placeholder={
        emptyMetricsList ? (
          <FormattedMessage
            defaultMessage="No metrics available"
            description="Text shown in a disabled multi-selector when there are no selectable metrics."
          />
        ) : (
          <FormattedMessage
            defaultMessage="Select metrics"
            description="Placeholder text for a metric multi-selector when configuring a line chart"
          />
        )
      }
      css={styles.selectFull}
      value={emptyMetricsList ? [] : selectedMetricKeys}
      onChange={updateSelectedMetrics}
      disabled={emptyMetricsList}
      dangerouslySetAntdProps={{ showSearch: true }}
    >
      {metricKeyList.map((metric) => (
        <LegacySelect.Option key={metric} value={metric} data-testid={`metric-${metric}`}>
          {metric}
        </LegacySelect.Option>
      ))}
    </LegacySelect>
  );
};

const renderXAxisMetricSelector = ({
  theme,
  metricKeyList,
  selectedXAxisMetricKey,
  updateSelectedXAxisMetricKey,
}: {
  theme: ThemeType;
  metricKeyList: string[];
  selectedXAxisMetricKey?: string;
  updateSelectedXAxisMetricKey: (metricKey: string) => void;
}) => {
  const emptyMetricsList = metricKeyList.length === 0;

  return (
    <Radio value={RunsChartsLineChartXAxisType.METRIC}>
      <FormattedMessage
        defaultMessage="Metric"
        description="Label for a radio button that configures the x-axis on a line chart. This option makes the X-axis a custom metric that the user selects."
      />
      <LegacySelect
        css={{
          marginTop: theme.spacing.xs,
          width: '100%',
        }}
        value={selectedXAxisMetricKey || undefined}
        placeholder={
          emptyMetricsList ? (
            <FormattedMessage
              defaultMessage="No metrics available"
              description="Text shown in a disabled metric selector when there are no selectable metrics."
            />
          ) : (
            <FormattedMessage
              defaultMessage="Select metric"
              description="Placeholder text for a metric selector when configuring a line chart"
            />
          )
        }
        onClick={(e: React.MouseEvent<HTMLElement>) => {
          // this is to prevent the radio button
          // from automatically closing the selector
          e.preventDefault();
          e.stopPropagation();
        }}
        onChange={updateSelectedXAxisMetricKey}
        disabled={emptyMetricsList}
        dangerouslySetAntdProps={{ showSearch: true }}
      >
        {metricKeyList.map((metric) => (
          <LegacySelect.Option key={metric} value={metric} data-testid={`metric-${metric}`}>
            {metric}
          </LegacySelect.Option>
        ))}
      </LegacySelect>
    </Radio>
  );
};

const safeLog = (x: number | undefined) => {
  if (isUndefined(x)) {
    return x;
  }
  if (x <= 0) {
    return undefined;
  }
  return Math.log10(x);
};

const safePow = (x: number | undefined) => {
  if (isUndefined(x)) {
    return x;
  }
  return Math.pow(10, x);
};

/**
 * Form containing configuration controls for runs compare charts.
 */
export const RunsChartsConfigureLineChart = ({
  state,
  onStateChange,
  metricKeyList,
}: {
  metricKeyList: string[];
  state: Partial<RunsChartsLineCardConfig>;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsLineCardConfig) => void;
}) => {
  const shouldEnableMetricsOnXAxis = shouldEnableDeepLearningUIPhase3();
  const usingManualRangeControls = shouldEnableManualRangeControls();
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const runSelectOptions = [5, 10, 20, 50, 100];

  const [localAxisRange, setLocalAxisRange] = useState<ChartRange>({
    xMin: state.xAxisScaleType === 'log' ? safePow(state.range?.xMin) : state.range?.xMin,
    xMax: state.xAxisScaleType === 'log' ? safePow(state.range?.xMax) : state.range?.xMax,
    yMin: state.scaleType === 'log' ? safePow(state.range?.yMin) : state.range?.yMin,
    yMax: state.scaleType === 'log' ? safePow(state.range?.yMax) : state.range?.yMax,
  });

  /**
   * Callback for updating metric key
   */
  const updateMetric = useCallback(
    (metricKey: string) => {
      onStateChange((current) => ({ ...(current as RunsChartsLineCardConfig), metricKey }));
    },
    [onStateChange],
  );

  const updateSelectedMetrics = useCallback(
    (metricKeys: string[]) => {
      onStateChange((current) => ({
        ...(current as RunsChartsLineCardConfig),
        metricKey: metricKeys[0],
        selectedMetricKeys: metricKeys,
      }));
    },
    [onStateChange],
  );

  const updateXAxisKey = useCallback(
    (xAxisKey: RunsChartsLineCardConfig['xAxisKey']) => {
      onStateChange((current) => {
        const config = current as RunsChartsLineCardConfig;
        return {
          ...config,
          xAxisKey,
          selectedXAxisMetricKey: '',
          range: {
            ...config.range,
            xMin: undefined,
            xMax: undefined,
          },
        };
      });
    },
    [onStateChange],
  );

  const isInvalidLogValue = (value: number | undefined) => !isUndefined(value) && value <= 0;

  const updateXAxisScaleType = useCallback(
    (isLogType: boolean) => {
      if (usingManualRangeControls) {
        onStateChange((current) => {
          const config = current as RunsChartsLineCardConfig;

          let newXMin = isLogType ? safeLog(localAxisRange.xMin) : localAxisRange.xMin;
          let newXMax = isLogType ? safeLog(localAxisRange.xMax) : localAxisRange.xMax;
          if (isLogType && isInvalidLogValue(localAxisRange.xMin) && localAxisRange.xMax && localAxisRange.xMax > 1) {
            // when switching to log type, if only xMin is invalid, set xMin to 1.
            setLocalAxisRange((prev) => ({
              ...prev,
              xMin: 1,
            }));
            newXMin = 0;
          } else if (isLogType && (isInvalidLogValue(localAxisRange.xMin) || isInvalidLogValue(localAxisRange.xMax))) {
            setLocalAxisRange((prev) => ({
              ...prev,
              xMin: undefined,
              xMax: undefined,
            }));
            newXMin = undefined;
            newXMax = undefined;
          }
          return {
            ...config,
            xAxisScaleType: isLogType ? 'log' : 'linear',
            range: {
              ...config.range,
              xMin: newXMin,
              xMax: newXMax,
            },
          };
        });
      } else {
        onStateChange((current) => ({
          ...(current as RunsChartsLineCardConfig),
          xAxisScaleType: isLogType ? 'log' : 'linear',
        }));
      }
    },
    [onStateChange, localAxisRange.xMin, localAxisRange.xMax, usingManualRangeControls],
  );

  const updateSelectedXAxisMetricKey = useCallback(
    (selectedXAxisMetricKey: string) => {
      onStateChange((current) => ({
        ...(current as RunsChartsLineCardConfig),
        selectedXAxisMetricKey,
        xAxisKey: RunsChartsLineChartXAxisType.METRIC,
      }));
    },
    [onStateChange],
  );

  const updateYAxisType = useCallback(
    (isLogType: boolean) => {
      if (usingManualRangeControls) {
        onStateChange((current) => {
          const config = current as RunsChartsLineCardConfig;

          let newYMin = isLogType ? safeLog(localAxisRange.yMin) : localAxisRange.yMin;
          let newYMax = isLogType ? safeLog(localAxisRange.yMax) : localAxisRange.yMax;
          if (isLogType && isInvalidLogValue(localAxisRange.yMin) && localAxisRange.yMax && localAxisRange.yMax > 1) {
            // when switching to log type, if only yMin is invalid, set yMin to 1.
            setLocalAxisRange((prev) => ({
              ...prev,
              yMin: 1,
            }));
            newYMin = 0; // This is the logged value of 1.
          } else if (isLogType && (isInvalidLogValue(localAxisRange.yMin) || isInvalidLogValue(localAxisRange.yMax))) {
            setLocalAxisRange((prev) => ({
              ...prev,
              yMin: undefined,
              yMax: undefined,
            }));
            newYMin = undefined;
            newYMax = undefined;
          }
          return {
            ...config,
            scaleType: isLogType ? 'log' : 'linear',
            range: {
              ...config.range,
              yMin: newYMin,
              yMax: newYMax,
            },
          };
        });
      } else {
        onStateChange((current) => ({
          ...(current as RunsChartsLineCardConfig),
          scaleType: isLogType ? 'log' : 'linear',
        }));
      }
    },
    [onStateChange, localAxisRange.yMin, localAxisRange.yMax, usingManualRangeControls],
  );

  const updateSmoothing = useCallback(
    (lineSmoothness: number) => {
      onStateChange((current) => ({
        ...(current as RunsChartsLineCardConfig),
        lineSmoothness: lineSmoothness,
      }));
    },
    [onStateChange],
  );

  /**
   * Callback for updating run count
   */
  const updateVisibleRunCount = useCallback(
    (runsCountToCompare: number) => {
      onStateChange((current) => ({
        ...(current as RunsChartsLineCardConfig),
        runsCountToCompare,
      }));
    },
    [onStateChange],
  );

  /**
   * If somehow metric key is not predetermined, automatically
   * select the first one so it's not empty
   */
  useEffect(() => {
    if (!state.metricKey && metricKeyList?.[0]) {
      updateMetric(metricKeyList[0]);
    }
  }, [state.metricKey, updateMetric, metricKeyList]);

  // for backwards compatibility, if selectedMetricKeys
  // is not present, set it using metricKey.
  useEffect(() => {
    if (isUndefined(state.selectedMetricKeys) && !isUndefined(state.metricKey) && state.metricKey !== '') {
      updateSelectedMetrics([state.metricKey]);
    }
  }, [state.selectedMetricKeys, state.metricKey, updateSelectedMetrics]);

  const updateXAxisWhenConfirmed = (xMin: number | undefined, xMax: number | undefined) => {
    if (inTransitionState(xMin, xMax)) {
      return;
    }
    onStateChange((current) => {
      const config = current as RunsChartsLineCardConfig;
      return {
        ...config,
        range: {
          ...config.range,
          xMin: config.xAxisScaleType === 'log' ? safeLog(xMin) : xMin,
          xMax: config.xAxisScaleType === 'log' ? safeLog(xMax) : xMax,
        },
      };
    });
  };

  const updateYAxisWhenConfirmed = (yMin: number | undefined, yMax: number | undefined) => {
    if (inTransitionState(yMin, yMax)) {
      return;
    }
    onStateChange((current) => {
      const config = current as RunsChartsLineCardConfig;
      return {
        ...config,
        range: {
          ...config.range,
          yMin: config.scaleType === 'log' ? safeLog(yMin) : yMin,
          yMax: config.scaleType === 'log' ? safeLog(yMax) : yMax,
        },
      };
    });
  };

  const updateXAxisScaleMin = (xMin: string) => {
    const newXMin = xMin ? Number(xMin) : undefined;
    setLocalAxisRange((prev) => ({ ...prev, xMin: newXMin }));
    updateXAxisWhenConfirmed(newXMin, localAxisRange.xMax);
  };
  const updateXAxisScaleMax = (xMax: string) => {
    const newXMax = xMax ? Number(xMax) : undefined;
    setLocalAxisRange((prev) => ({ ...prev, xMax: newXMax }));
    updateXAxisWhenConfirmed(localAxisRange.xMin, newXMax);
  };
  const updateYAxisScaleMin = (yMin: string) => {
    const newYMin = yMin ? Number(yMin) : undefined;
    setLocalAxisRange((prev) => ({ ...prev, yMin: newYMin }));
    updateYAxisWhenConfirmed(newYMin, localAxisRange.yMax);
  };
  const updateYAxisScaleMax = (yMax: string) => {
    const newYMax = yMax ? Number(yMax) : undefined;
    setLocalAxisRange((prev) => ({ ...prev, yMax: newYMax }));
    updateYAxisWhenConfirmed(localAxisRange.yMin, newYMax);
  };

  const inTransitionState = (a: number | undefined, b: number | undefined) => {
    if (isUndefined(a) && isUndefined(b)) {
      return false;
    } else if (!isUndefined(a) && !isUndefined(b)) {
      return false;
    } else {
      return true;
    }
  };

  const hintAndInvalidMessage = (
    scaleType: 'log' | 'linear' | undefined,
    value: number | undefined,
    hintTitle: string,
  ) => {
    if (scaleType === 'log' && isInvalidLogValue(value)) {
      return (
        <FormUI.Message
          message={
            <FormattedMessage
              defaultMessage="Invalid log value"
              description="Experiment tracking > runs charts > line chart configuration > invalid log value message"
            />
          }
          type="warning"
        />
      );
    }
    return <FormUI.Hint>{hintTitle}</FormUI.Hint>;
  };

  return (
    <>
      <RunsChartsConfigureField title="Metric (Y-axis)">
        {renderMetricSelectorV2({
          metricKeyList,
          selectedMetricKeys: state.selectedMetricKeys,
          updateSelectedMetrics,
        })}
      </RunsChartsConfigureField>
      <RunsChartsConfigureField title="X-axis">
        <Radio.Group
          name="runs-charts-field-group-x-axis"
          value={state.xAxisKey}
          onChange={({ target: { value } }) => updateXAxisKey(value)}
        >
          <Radio value={RunsChartsLineChartXAxisType.STEP}>
            <FormattedMessage
              defaultMessage="Step"
              description="Label for a radio button that configures the x-axis on a line chart. This option is for the step number that the metrics were logged."
            />
          </Radio>
          <Radio value={RunsChartsLineChartXAxisType.TIME}>
            <FormattedMessage
              defaultMessage="Time (wall)"
              description="Label for a radio button that configures the x-axis on a line chart. This option is for the absolute time that the metrics were logged."
            />
            <Tooltip
              title={
                <FormattedMessage
                  defaultMessage="Absolute date and time"
                  description="A tooltip line chart configuration for the step function of wall time"
                />
              }
              placement="right"
            >
              {' '}
              <QuestionMarkIcon css={styles.timeStepQuestionMarkIcon} />
            </Tooltip>
          </Radio>
          <Radio value={RunsChartsLineChartXAxisType.TIME_RELATIVE}>
            <FormattedMessage
              defaultMessage="Time (relative)"
              description="Label for a radio button that configures the x-axis on a line chart. This option is for relative time since the first metric was logged."
            />
            <Tooltip
              title={
                <FormattedMessage
                  defaultMessage="Amount of time that has passed since the first metric value was logged"
                  description="A tooltip line chart configuration for the step function of relative time"
                />
              }
              placement="right"
            >
              {' '}
              <QuestionMarkIcon css={styles.timeStepQuestionMarkIcon} />
            </Tooltip>
          </Radio>
          {shouldEnableMetricsOnXAxis &&
            renderXAxisMetricSelector({
              theme,
              metricKeyList,
              selectedXAxisMetricKey: state.selectedXAxisMetricKey,
              updateSelectedXAxisMetricKey,
            })}
        </Radio.Group>
      </RunsChartsConfigureField>
      {state.xAxisKey === RunsChartsLineChartXAxisType.STEP && (
        <>
          {usingManualRangeControls && (
            <RunsChartsConfigureField title="X-axis scale">
              <div css={{ display: 'flex', gap: theme.spacing.sm }}>
                <div>
                  <Input
                    componentId="mlflow.charts.line_chart_configure.x_axis_min"
                    aria-label="x-axis-min"
                    name="min"
                    type="number"
                    value={localAxisRange.xMin}
                    onChange={(e) => updateXAxisScaleMin(e.target.value)}
                    max={localAxisRange.xMax}
                  />
                  {hintAndInvalidMessage(state.xAxisScaleType, localAxisRange.xMin, 'Min')}
                </div>
                <div>
                  <Input
                    componentId="mlflow.charts.line_chart_configure.x_axis_max"
                    aria-label="x-axis-max"
                    name="max"
                    type="number"
                    value={localAxisRange.xMax}
                    onChange={(e) => updateXAxisScaleMax(e.target.value)}
                    min={localAxisRange.xMin}
                  />
                  {hintAndInvalidMessage(state.xAxisScaleType, localAxisRange.xMax, 'Max')}
                </div>
              </div>
            </RunsChartsConfigureField>
          )}
          <RunsChartsConfigureField title="X-axis log scale">
            <Switch
              aria-label="x-axis-log"
              checked={state.xAxisScaleType === 'log'}
              onChange={updateXAxisScaleType}
              label="Enabled"
            />
          </RunsChartsConfigureField>
        </>
      )}

      {usingManualRangeControls && (
        <RunsChartsConfigureField title="Y-axis scale">
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <div>
              <Input
                componentId="mlflow.charts.line_chart_configure.y_axis_min"
                aria-label="y-axis-min"
                name="min"
                type="number"
                value={localAxisRange.yMin}
                onChange={(e) => updateYAxisScaleMin(e.target.value)}
                max={localAxisRange.yMax}
              />
              {hintAndInvalidMessage(state.scaleType, localAxisRange.yMin, 'Min')}
            </div>
            <div>
              <Input
                componentId="mlflow.charts.line_chart_configure.y_axis_max"
                aria-label="y-axis-max"
                name="max"
                type="number"
                value={localAxisRange.yMax}
                onChange={(e) => updateYAxisScaleMax(e.target.value)}
                min={localAxisRange.yMin}
              />
              {hintAndInvalidMessage(state.scaleType, localAxisRange.yMax, 'Max')}
            </div>
          </div>
        </RunsChartsConfigureField>
      )}
      <RunsChartsConfigureField title="Y-axis log scale">
        <Switch
          aria-label="y-axis-log"
          checked={state.scaleType === 'log'}
          onChange={updateYAxisType}
          label="Enabled"
        />
      </RunsChartsConfigureField>
      <RunsChartsConfigureField
        title={intl.formatMessage({
          defaultMessage: 'Display points',
          description: 'Runs charts > line chart > display points > label',
        })}
      >
        <SegmentedControlGroup
          name={intl.formatMessage({
            defaultMessage: 'Display points',
            description: 'Runs charts > line chart > display points > label',
          })}
          value={state.displayPoints}
          onChange={({ target }) => {
            onStateChange((current) => ({
              ...(current as RunsChartsLineCardConfig),
              displayPoints: target.value,
            }));
          }}
        >
          <SegmentedControlButton
            value={undefined}
            aria-label={[
              intl.formatMessage({
                defaultMessage: 'Display points',
                description: 'Runs charts > line chart > display points > label',
              }),
              intl.formatMessage({
                defaultMessage: 'Auto',
                description: 'Runs charts > line chart > display points > auto setting label',
              }),
            ].join(': ')}
          >
            <FormattedMessage
              defaultMessage="Auto"
              description="Runs charts > line chart > display points > auto setting label"
            />{' '}
            <Tooltip
              title={
                <FormattedMessage
                  defaultMessage="Show points on line charts if there are fewer than 60 data points per trace"
                  description="Runs charts > line chart > display points > auto tooltip"
                />
              }
            >
              <InfoIcon />
            </Tooltip>
          </SegmentedControlButton>
          <SegmentedControlButton
            value
            aria-label={[
              intl.formatMessage({
                defaultMessage: 'Display points',
                description: 'Runs charts > line chart > display points > label',
              }),
              intl.formatMessage({
                defaultMessage: 'On',
                description: 'Runs charts > line chart > display points > on setting label',
              }),
            ].join(': ')}
          >
            <FormattedMessage
              defaultMessage="On"
              description="Runs charts > line chart > display points > on setting label"
            />
          </SegmentedControlButton>
          <SegmentedControlButton
            value={false}
            aria-label={[
              intl.formatMessage({
                defaultMessage: 'Display points',
                description: 'Runs charts > line chart > display points > label',
              }),
              intl.formatMessage({
                defaultMessage: 'Off',
                description: 'Runs charts > line chart > display points > off setting label',
              }),
            ].join(': ')}
          >
            <FormattedMessage
              defaultMessage="Off"
              description="Runs charts > line chart > display points > off setting label"
            />
          </SegmentedControlButton>
        </SegmentedControlGroup>
      </RunsChartsConfigureField>
      <RunsChartsConfigureField title="Line smoothness">
        <LineSmoothSlider
          data-testid="smoothness-toggle"
          min={0}
          max={100}
          onChange={updateSmoothing}
          defaultValue={state.lineSmoothness ? state.lineSmoothness : 0}
        />
      </RunsChartsConfigureField>
      <RunsChartsRunNumberSelect
        value={state.runsCountToCompare}
        onChange={updateVisibleRunCount}
        options={runSelectOptions}
      />
    </>
  );
};

const styles = {
  selectFull: { width: '100%' },
  timeStepQuestionMarkIcon: () => ({
    svg: { width: 12, height: 12 },
  }),
};
