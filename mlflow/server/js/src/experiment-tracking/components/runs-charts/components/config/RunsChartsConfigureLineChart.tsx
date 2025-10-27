import type { ThemeType } from '@databricks/design-system';
import {
  Radio,
  LegacySelect,
  Switch,
  LegacyTooltip,
  QuestionMarkIcon,
  useDesignSystemTheme,
  SegmentedControlGroup,
  SegmentedControlButton,
  InfoSmallIcon,
  Input,
  FormUI,
  Typography,
  SimpleSelect,
  SimpleSelectOption,
  SimpleSelectOptionGroup,
  Tooltip,
  PlusIcon,
  Button,
  Spacer,
  CloseIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCallback, useEffect, useState } from 'react';
import {
  RunsChartsLineChartYAxisType,
  type ChartRange,
  type RunsChartsCardConfig,
  type RunsChartsLineCardConfig,
} from '../../runs-charts.types';
import { RunsChartsConfigureField } from './RunsChartsConfigure.common';
import { shouldEnableChartExpressions } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { RunsChartsLineChartXAxisType } from '@mlflow/mlflow/src/experiment-tracking/components/runs-charts/components/RunsCharts.common';
import { LineSmoothSlider } from '@mlflow/mlflow/src/experiment-tracking/components/LineSmoothSlider';
import { isUndefined } from 'lodash';
import { RunsChartsYAxisMetricAndExpressionSelector } from '../RunsChartsYAxisMetricAndExpressionSelector';

const USE_GLOBAL_SETTING_KEY = '_GLOBAL';

const renderXAxisMetricSelector = ({
  theme,
  metricKeyList,
  selectedXAxisMetricKey,
  updateSelectedXAxisMetricKey,
  disabled = false,
}: {
  theme: ThemeType;
  metricKeyList: string[];
  selectedXAxisMetricKey?: string;
  updateSelectedXAxisMetricKey: (metricKey: string) => void;
  disabled?: boolean;
}) => {
  const emptyMetricsList = metricKeyList.length === 0;

  return (
    <Radio value={RunsChartsLineChartXAxisType.METRIC} disabled={disabled}>
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
        disabled={emptyMetricsList || disabled}
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
  const usingChartExpressions = shouldEnableChartExpressions();
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
    (xAxisKey: RunsChartsLineCardConfig['xAxisKey'], useGlobal = false) => {
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
          useGlobalXaxisKey: useGlobal ?? config.useGlobalXaxisKey,
          selectedYAxisMetricKey: RunsChartsLineChartYAxisType.METRIC,
        };
      });
    },
    [onStateChange],
  );

  const isInvalidLogValue = (value: number | undefined) => !isUndefined(value) && value <= 0;

  const updateXAxisScaleType = useCallback(
    (isLogType: boolean) => {
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
    },
    [onStateChange, localAxisRange.xMin, localAxisRange.xMax],
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
    },
    [onStateChange, localAxisRange.yMin, localAxisRange.yMax],
  );

  const updateIgnoreOutliers = useCallback(
    (ignoreOutliers: boolean) => {
      onStateChange((current) => ({
        ...(current as RunsChartsLineCardConfig),
        ignoreOutliers,
      }));
    },
    [onStateChange],
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

  const invalidMessage = (scaleType: 'log' | 'linear' | undefined, value: number | undefined) => {
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
    return null;
  };

  const METRIC_AXIS_PREFIX = 'metric-';
  return (
    <>
      <Typography.Title level={4} color="secondary">
        X-axis
      </Typography.Title>
      <RunsChartsConfigureField title="Type" compact>
        {usingChartExpressions ? (
          <SimpleSelect
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_436"
            id="x-axis-type"
            width="100%"
            value={state.useGlobalXaxisKey ? USE_GLOBAL_SETTING_KEY : state.xAxisKey}
            contentProps={{ matchTriggerWidth: true, textOverflowMode: 'ellipsis' }}
            onChange={({ target: { value } }) => {
              if (value.startsWith(METRIC_AXIS_PREFIX)) {
                updateSelectedXAxisMetricKey(value.slice(METRIC_AXIS_PREFIX.length));
              } else if (value === RunsChartsLineChartXAxisType.STEP) {
                updateXAxisKey(RunsChartsLineChartXAxisType.STEP);
              } else if (value === RunsChartsLineChartXAxisType.TIME) {
                updateXAxisKey(RunsChartsLineChartXAxisType.TIME);
              } else if (value === RunsChartsLineChartXAxisType.TIME_RELATIVE) {
                updateXAxisKey(RunsChartsLineChartXAxisType.TIME_RELATIVE);
              } else if (value === USE_GLOBAL_SETTING_KEY) {
                updateXAxisKey(RunsChartsLineChartXAxisType.STEP, true);
              }
            }}
          >
            <SimpleSelectOption value={USE_GLOBAL_SETTING_KEY}>
              <FormattedMessage
                defaultMessage="Use workspace settings"
                description="Label for a radio button that configures the x-axis on a line chart. This option is for using global workspace settings."
              />
            </SimpleSelectOption>
            <SimpleSelectOption value={RunsChartsLineChartXAxisType.STEP}>
              <FormattedMessage
                defaultMessage="Step"
                description="Label for a radio button that configures the x-axis on a line chart. This option is for the step number that the metrics were logged."
              />
            </SimpleSelectOption>
            <SimpleSelectOption value={RunsChartsLineChartXAxisType.TIME}>
              <FormattedMessage
                defaultMessage="Time (wall)"
                description="Label for a radio button that configures the x-axis on a line chart. This option is for the absolute time that the metrics were logged."
              />
              <Tooltip
                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_474"
                content={
                  <FormattedMessage
                    defaultMessage="Absolute date and time"
                    description="A tooltip line chart configuration for the step function of wall time"
                  />
                }
                side="right"
              >
                <span>
                  {' '}
                  <QuestionMarkIcon css={styles.timeStepQuestionMarkIcon} />
                </span>
              </Tooltip>
            </SimpleSelectOption>
            <SimpleSelectOption value={RunsChartsLineChartXAxisType.TIME_RELATIVE}>
              <FormattedMessage
                defaultMessage="Time (relative)"
                description="Label for a radio button that configures the x-axis on a line chart. This option is for relative time since the first metric was logged."
              />
              <Tooltip
                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_494"
                content={
                  <FormattedMessage
                    defaultMessage="Amount of time that has passed since the first metric value was logged"
                    description="A tooltip line chart configuration for the step function of relative time"
                  />
                }
                side="right"
              >
                <span>
                  {' '}
                  <QuestionMarkIcon css={styles.timeStepQuestionMarkIcon} />
                </span>
              </Tooltip>
            </SimpleSelectOption>
            {metricKeyList.length > 0 && (
              <SimpleSelectOptionGroup label="Metrics">
                {metricKeyList.map((metric) => (
                  <SimpleSelectOption
                    key={metric}
                    value={`${METRIC_AXIS_PREFIX}${metric}`}
                    data-testid={`${METRIC_AXIS_PREFIX}${metric}`}
                  >
                    {metric}
                  </SimpleSelectOption>
                ))}
              </SimpleSelectOptionGroup>
            )}
          </SimpleSelect>
        ) : (
          <Radio.Group
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_524"
            name="runs-charts-field-group-x-axis"
            value={state.useGlobalXaxisKey ? USE_GLOBAL_SETTING_KEY : state.xAxisKey}
            onChange={({ target: { value } }) => {
              if (value === USE_GLOBAL_SETTING_KEY) {
                updateXAxisKey(RunsChartsLineChartXAxisType.STEP, true);
              } else {
                updateXAxisKey(value);
              }
            }}
          >
            <Radio value={USE_GLOBAL_SETTING_KEY}>
              <FormattedMessage
                defaultMessage="Use workspace settings"
                description="Label for a radio button that configures the x-axis on a line chart. This option is for using global workspace settings."
              />
            </Radio>

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
              <LegacyTooltip
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
              </LegacyTooltip>
            </Radio>
            <Radio value={RunsChartsLineChartXAxisType.TIME_RELATIVE}>
              <FormattedMessage
                defaultMessage="Time (relative)"
                description="Label for a radio button that configures the x-axis on a line chart. This option is for relative time since the first metric was logged."
              />
              <LegacyTooltip
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
              </LegacyTooltip>
            </Radio>
            {renderXAxisMetricSelector({
              theme,
              metricKeyList,
              selectedXAxisMetricKey: state.selectedXAxisMetricKey,
              updateSelectedXAxisMetricKey,
            })}
          </Radio.Group>
        )}
      </RunsChartsConfigureField>
      {state.xAxisKey === RunsChartsLineChartXAxisType.STEP && (
        <>
          <RunsChartsConfigureField title="X-axis scale" compact>
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
                  placeholder="Min"
                />
                {invalidMessage(state.xAxisScaleType, localAxisRange.xMin)}
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
                  placeholder="Max"
                />
                {invalidMessage(state.xAxisScaleType, localAxisRange.xMax)}
              </div>
            </div>
            <div style={{ padding: theme.spacing.xs }} />
            <Switch
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_628"
              aria-label="x-axis-log"
              checked={state.xAxisScaleType === 'log'}
              onChange={updateXAxisScaleType}
              label="Log scale"
              activeLabel="On"
              inactiveLabel="Off"
              disabledLabel="Disabled"
            />
          </RunsChartsConfigureField>
        </>
      )}
      <Typography.Title level={4} color="secondary" css={{ paddingTop: theme.spacing.lg }}>
        Y-axis
      </Typography.Title>
      <RunsChartsYAxisMetricAndExpressionSelector
        state={state}
        onStateChange={onStateChange}
        metricKeyList={metricKeyList}
        updateSelectedMetrics={updateSelectedMetrics}
      />
      <RunsChartsConfigureField title="Y-axis scale" compact>
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
              placeholder="Min"
            />
            {invalidMessage(state.scaleType, localAxisRange.yMin)}
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
              placeholder="Max"
            />
            {invalidMessage(state.scaleType, localAxisRange.yMax)}
          </div>
        </div>
        <Spacer size="xs" />
        <Switch
          componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_682"
          aria-label="y-axis-log"
          checked={state.scaleType === 'log'}
          onChange={updateYAxisType}
          label="Log scale"
          activeLabel="On"
          inactiveLabel="Off"
          disabledLabel="Disabled"
        />
        <Spacer size="xs" />
        <div css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between' }}>
          <div>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Ignore outliers"
                description="Runs charts > line chart > ignore outliers > label"
              />
            </Typography.Text>
            <Tooltip
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_703"
              delayDuration={0}
              content={
                <FormattedMessage
                  defaultMessage="Only display data points between the p5 and p95 of the data. This can help with chart readability in cases where outliers significantly affect the Y-axis range"
                  description="A tooltip describing the 'Ignore Outliers' configuration option for line charts"
                />
              }
              side="right"
            >
              <span>
                {' '}
                <QuestionMarkIcon css={styles.timeStepQuestionMarkIcon} />
              </span>
            </Tooltip>
          </div>
          <Switch
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_716"
            aria-label="y-axis-ignore-outliers"
            checked={state.ignoreOutliers}
            onChange={updateIgnoreOutliers}
            // Empty label so that the active/inactive labels are actually displayed
            label=" "
            activeLabel={intl.formatMessage({
              defaultMessage: 'On',
              description: 'Runs charts > line chart > ignore outliers > on setting label',
            })}
            inactiveLabel={intl.formatMessage({
              defaultMessage: 'Off',
              description: 'Runs charts > line chart > ignore outliers > off setting label',
            })}
            disabledLabel={intl.formatMessage({
              defaultMessage: 'Disabled',
              description: 'Runs charts > line chart > ignore outliers > disabled label',
            })}
          />
        </div>
      </RunsChartsConfigureField>
      <Typography.Title level={4} color="secondary" css={{ paddingTop: theme.spacing.lg }}>
        Advanced
      </Typography.Title>
      <RunsChartsConfigureField
        title={intl.formatMessage({
          defaultMessage: 'Display points',
          description: 'Runs charts > line chart > display points > label',
        })}
        compact
      >
        <SegmentedControlGroup
          componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_747"
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
            <LegacyTooltip
              title={
                <FormattedMessage
                  defaultMessage="Show points on line charts if there are fewer than 60 data points per trace"
                  description="Runs charts > line chart > display points > auto tooltip"
                />
              }
            >
              <InfoSmallIcon />
            </LegacyTooltip>
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
      <RunsChartsConfigureField
        title={
          <>
            <FormattedMessage
              defaultMessage="Line smoothing"
              description="Runs charts > line chart > configuration > label for line smoothing slider control. The control allows changing data trace line smoothness from 1 to 100, where 1 is the original data trace and 100 is the smoothest trace. Line smoothing helps eliminate noise in the data."
            />
          </>
        }
        compact
      >
        <Radio.Group
          componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigurelinechart.tsx_838"
          name="use-global-line-smoothness"
          value={Boolean(state.useGlobalLineSmoothing)}
          onChange={({ target }) => {
            onStateChange((current) => ({
              ...(current as RunsChartsLineCardConfig),
              useGlobalLineSmoothing: target.value === true,
            }));
          }}
        >
          <Radio value>Use workspace settings</Radio>
          <Radio value={false}>Custom</Radio>
        </Radio.Group>

        <LineSmoothSlider
          data-testid="smoothness-toggle"
          min={0}
          max={100}
          onChange={updateSmoothing}
          value={state.lineSmoothness ? state.lineSmoothness : 0}
          disabled={state.useGlobalLineSmoothing}
        />
      </RunsChartsConfigureField>
    </>
  );
};

const styles = {
  selectFull: { width: '100%' },
  timeStepQuestionMarkIcon: () => ({
    svg: { width: 12, height: 12 },
  }),
};
