import {
  Button,
  CloseIcon,
  Input,
  LegacySelect,
  PlusIcon,
  Radio,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { RunsChartsCardConfig, RunsChartsLineCardConfig, RunsChartsLineChartYAxisType } from '../runs-charts.types';
import { RunsChartsConfigureField } from './config/RunsChartsConfigure.common';
import { FormattedMessage, useIntl } from 'react-intl';
import { shouldEnableChartExpressions } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { useCallback } from 'react';

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
      css={{ width: '100%' }}
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
      css={{ width: '100%' }}
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

export const RunsChartsYAxisMetricAndExpressionSelector = ({
  state,
  onStateChange,
  metricKeyList,
  updateSelectedMetrics,
}: {
  state: Partial<RunsChartsLineCardConfig>;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsLineCardConfig) => void;
  metricKeyList: string[];
  updateSelectedMetrics: (metricKeys: string[]) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const usingChartExpressions = shouldEnableChartExpressions();

  const updateYAxisExpression = (expression: string, index: number) => {
    onStateChange((current) => {
      const config = current as RunsChartsLineCardConfig;
      const yAxisExpressions = config.yAxisExpressions || [];
      yAxisExpressions[index] = expression;
      return {
        ...config,
        yAxisExpressions,
      };
    });
  };

  const addNewYAxisExpression = () => {
    onStateChange((current) => {
      const config = current as RunsChartsLineCardConfig;
      const yAxisExpressions = config.yAxisExpressions || [];
      return {
        ...config,
        yAxisExpressions: [...yAxisExpressions, ''],
      };
    });
  };

  const removeYAxisExpression = (index: number) => {
    onStateChange((current) => {
      const config = current as RunsChartsLineCardConfig;
      const yAxisExpressions = config.yAxisExpressions || [];
      yAxisExpressions.splice(index, 1);
      return {
        ...config,
        yAxisExpressions,
      };
    });
  };

  const updateYAxisKey = useCallback(
    (yAxisKey: RunsChartsLineCardConfig['yAxisKey']) => {
      onStateChange((current) => {
        const config = current as RunsChartsLineCardConfig;
        return {
          ...config,
          yAxisKey,
          range: {
            ...config.range,
            yMin: undefined,
            yMax: undefined,
          },
        };
      });
    },
    [onStateChange],
  );

  return (
    <>
      {usingChartExpressions && (
        <RunsChartsConfigureField title="Metric type" compact>
          <Radio.Group
            name="runs-charts-field-group-metric-type-y-axis"
            value={state.yAxisKey || RunsChartsLineChartYAxisType.METRIC}
            onChange={({ target: { value } }) => updateYAxisKey(value)}
          >
            <Radio value={RunsChartsLineChartYAxisType.METRIC} key={RunsChartsLineChartYAxisType.METRIC}>
              <FormattedMessage
                defaultMessage="Logged metrics"
                description="Experiment tracking > runs charts > line chart configuration > logged metrics label"
              />
            </Radio>
            <Radio value={RunsChartsLineChartYAxisType.EXPRESSION} key={RunsChartsLineChartYAxisType.EXPRESSION}>
              <FormattedMessage
                defaultMessage="Custom expression"
                description="Experiment tracking > runs charts > line chart configuration > custom expression label"
              />
            </Radio>
          </Radio.Group>
        </RunsChartsConfigureField>
      )}
      {usingChartExpressions && state.yAxisKey === RunsChartsLineChartYAxisType.EXPRESSION ? (
        <RunsChartsConfigureField title="Expression" compact>
          <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.sm }}>
            {state.yAxisExpressions?.map((expression, index) => {
              return (
                <span css={{ display: 'flex', width: '100%', gap: theme.spacing.sm }}>
                  <Input
                    style={{ fontFamily: 'monospace' }}
                    value={expression}
                    onChange={(e) => updateYAxisExpression(e.target.value, index)}
                  />
                  <Button
                    componentId="mlflow.charts.line_chart_configure.expressions_remove"
                    icon={<CloseIcon />}
                    onClick={() => removeYAxisExpression(index)}
                  />
                </span>
              );
            })}
            <Button
              componentId="mlflow.charts.line_chart_configure.expressions_add_new"
              icon={<PlusIcon />}
              onClick={addNewYAxisExpression}
            >
              <FormattedMessage
                defaultMessage="Add new"
                description="Experiment tracking > runs charts > line chart configuration > add new expression button"
              />
            </Button>
          </div>
        </RunsChartsConfigureField>
      ) : (
        <RunsChartsConfigureField title="Metric" compact>
          {renderMetricSelectorV2({
            metricKeyList,
            selectedMetricKeys: state.selectedMetricKeys,
            updateSelectedMetrics,
          })}
        </RunsChartsConfigureField>
      )}
    </>
  );
};
