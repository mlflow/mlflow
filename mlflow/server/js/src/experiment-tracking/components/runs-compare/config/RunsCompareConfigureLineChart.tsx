import { Radio, Select, Switch, Tooltip, QuestionMarkIcon, Form } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useCallback, useEffect } from 'react';
import type { RunsCompareCardConfig, RunsCompareLineCardConfig } from '../runs-compare.types';
import { RunsCompareConfigureField, RunsCompareRunNumberSelect } from './RunsCompareConfigure.common';
import { LineSmoothSlider } from '../../LineSmoothSlider';
import { shouldEnableDeepLearningUI } from 'common/utils/FeatureUtils';

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
    <Select
      css={styles.selectFull}
      value={emptyMetricsList ? 'No metrics available' : metricKey}
      onChange={updateMetric}
      disabled={emptyMetricsList}
      dangerouslySetAntdProps={{ showSearch: true }}
    >
      {metricKeyList.map((metric) => (
        <Select.Option key={metric} value={metric} data-testid={`metric-${metric}`}>
          {metric}
        </Select.Option>
      ))}
    </Select>
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
    <Select
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
        <Select.Option key={metric} value={metric} data-testid={`metric-${metric}`}>
          {metric}
        </Select.Option>
      ))}
    </Select>
  );
};

/**
 * Form containing configuration controls for runs compare charts.
 */
export const RunsCompareConfigureLineChart = ({
  state,
  onStateChange,
  metricKeyList,
}: {
  metricKeyList: string[];
  state: Partial<RunsCompareLineCardConfig>;
  onStateChange: (setter: (current: RunsCompareCardConfig) => RunsCompareLineCardConfig) => void;
}) => {
  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const runSelectOptions = usingV2ChartImprovements ? [5, 10, 20, 50, 100] : [5, 10, 20];

  /**
   * Callback for updating metric key
   */
  const updateMetric = useCallback(
    (metricKey: string) => {
      onStateChange((current) => ({ ...(current as RunsCompareLineCardConfig), metricKey }));
    },
    [onStateChange],
  );

  const updateSelectedMetrics = useCallback(
    (metricKeys: string[]) => {
      onStateChange((current) => ({
        ...(current as RunsCompareLineCardConfig),
        metricKey: metricKeys[0],
        selectedMetricKeys: metricKeys,
      }));
    },
    [onStateChange],
  );

  const updateXAxisKey = useCallback(
    (xAxisKey: RunsCompareLineCardConfig['xAxisKey']) => {
      onStateChange((current) => ({
        ...(current as RunsCompareLineCardConfig),
        xAxisKey,
      }));
    },
    [onStateChange],
  );

  const updateYAxisType = useCallback(
    (isLogType: boolean) =>
      onStateChange((current) => ({
        ...(current as RunsCompareLineCardConfig),
        scaleType: isLogType ? 'log' : 'linear',
      })),
    [onStateChange],
  );

  const updateSmoothing = useCallback(
    (lineSmoothness: number) => {
      onStateChange((current) => ({
        ...(current as RunsCompareLineCardConfig),
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
        ...(current as RunsCompareLineCardConfig),
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
    if (
      usingV2ChartImprovements &&
      state.selectedMetricKeys === undefined &&
      state.metricKey !== undefined &&
      state.metricKey !== ''
    ) {
      updateSelectedMetrics([state.metricKey]);
    }
  }, [state.selectedMetricKeys, state.metricKey, updateSelectedMetrics, usingV2ChartImprovements]);

  return (
    <>
      <RunsCompareConfigureField title="Metric (Y-axis)">
        {usingV2ChartImprovements
          ? renderMetricSelectorV2({
              metricKeyList,
              selectedMetricKeys: state.selectedMetricKeys,
              updateSelectedMetrics,
            })
          : renderMetricSelectorV1({ metricKeyList, metricKey: state.metricKey, updateMetric })}
      </RunsCompareConfigureField>
      <RunsCompareConfigureField title="X-axis">
        <Radio.Group value={state.xAxisKey} onChange={({ target: { value } }) => updateXAxisKey(value)}>
          <Radio value="step">Step</Radio>
          <Radio value="time">
            Time (wall)
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
          <Radio value="time-relative">
            Time (relative)
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
        </Radio.Group>
      </RunsCompareConfigureField>
      <RunsCompareConfigureField title="Y-axis log scale">
        <Switch checked={state.scaleType === 'log'} onChange={updateYAxisType} label="Enabled" />
      </RunsCompareConfigureField>
      <RunsCompareConfigureField title="Line smoothness">
        <LineSmoothSlider
          data-testid="smoothness-toggle"
          min={0}
          max={100}
          handleLineSmoothChange={updateSmoothing}
          defaultValue={state.lineSmoothness ? state.lineSmoothness : 0}
        />
      </RunsCompareConfigureField>
      <RunsCompareRunNumberSelect
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
