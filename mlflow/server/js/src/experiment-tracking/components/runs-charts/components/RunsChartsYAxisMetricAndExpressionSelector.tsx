import {
  Button,
  CloseIcon,
  Input,
  LegacySelect,
  PlusIcon,
  Radio,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type {
  RunsChartsCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsLineChartExpression,
} from '../runs-charts.types';
import { RunsChartsLineChartYAxisType } from '../runs-charts.types';
import { RunsChartsConfigureField } from './config/RunsChartsConfigure.common';
import { FormattedMessage } from 'react-intl';
import { shouldEnableChartExpressions } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { useCallback, useEffect, useState } from 'react';
import { useChartExpressionParser } from '../hooks/useChartExpressionParser';
import { RunsChartsLineChartXAxisType } from './RunsCharts.common';

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

const ExpressionInput = ({
  chartExpression,
  index,
  updateYAxisExpression,
  removeYAxisExpression,
  metricKeyList,
}: {
  chartExpression: RunsChartsLineChartExpression;
  index: number;
  updateYAxisExpression: (expression: RunsChartsLineChartExpression, index: number) => void;
  removeYAxisExpression: (index: number) => void;
  metricKeyList: string[];
}) => {
  const { theme } = useDesignSystemTheme();
  const { compileExpression } = useChartExpressionParser();
  const [isValidExpression, setIsValidExpression] = useState(true);
  const validateAndUpdate = (expression: string) => {
    const compiledExpression = compileExpression(expression, metricKeyList);
    if (compiledExpression === undefined) {
      setIsValidExpression(false);
      updateYAxisExpression({ rpn: [], variables: [], expression }, index);
    } else {
      setIsValidExpression(true);
      updateYAxisExpression(compiledExpression, index);
    }
  };

  return (
    <span css={{ display: 'flex', width: '100%', gap: theme.spacing.sm }}>
      <Input
        componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsyaxismetricandexpressionselector.tsx_122"
        value={chartExpression.expression}
        onChange={(e) => validateAndUpdate(e.target.value)}
        validationState={isValidExpression ? undefined : 'error'}
      />
      <Button
        componentId="mlflow.charts.line-chart-expressions-remove"
        icon={<CloseIcon />}
        onClick={() => removeYAxisExpression(index)}
      />
    </span>
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
  const usingChartExpressions =
    shouldEnableChartExpressions() && state.xAxisKey !== RunsChartsLineChartXAxisType.METRIC;

  const DEBOUNCE_DELAY = 300; // in ms

  const [temporaryChartExpressions, setTemporaryChartExpressions] = useState<RunsChartsLineChartExpression[]>(
    state.yAxisExpressions || [],
  );

  const updateYAxisExpressionTemporary = (expression: RunsChartsLineChartExpression, index: number) => {
    setTemporaryChartExpressions((current) => {
      const newExpressions = [...current];
      newExpressions[index] = expression;
      return newExpressions;
    });
  };

  const addNewYAxisExpressionTemporary = () => {
    setTemporaryChartExpressions((current) => {
      return [...current, { rpn: [], variables: [], expression: '' } as RunsChartsLineChartExpression];
    });
  };

  const removeYAxisExpressionTemporary = (index: number) => {
    setTemporaryChartExpressions((current) => {
      const newExpressions = [...current];
      newExpressions.splice(index, 1);
      return newExpressions;
    });
  };

  useEffect(() => {
    const updateYAxisExpression = (yAxisExpressions: RunsChartsLineChartExpression[]) => {
      onStateChange((current) => {
        const config = current as RunsChartsLineCardConfig;
        return {
          ...config,
          yAxisExpressions,
        };
      });
    };
    const handler = setTimeout(() => {
      updateYAxisExpression(temporaryChartExpressions);
    }, DEBOUNCE_DELAY);

    return () => {
      clearTimeout(handler);
    };
  }, [temporaryChartExpressions, onStateChange]);

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
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsyaxismetricandexpressionselector.tsx_221"
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
            {temporaryChartExpressions.map((chartExpression, index) => {
              return (
                <ExpressionInput
                  key={index}
                  chartExpression={chartExpression}
                  index={index}
                  updateYAxisExpression={updateYAxisExpressionTemporary}
                  removeYAxisExpression={removeYAxisExpressionTemporary}
                  metricKeyList={metricKeyList}
                />
              );
            })}
            <Button
              componentId="mlflow.charts.line-chart-expressions-add-new"
              icon={<PlusIcon />}
              onClick={addNewYAxisExpressionTemporary}
            >
              Add new
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
