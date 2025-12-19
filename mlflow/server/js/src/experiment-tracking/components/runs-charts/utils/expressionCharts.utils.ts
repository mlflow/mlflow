import { intersection } from 'lodash';
import type { MetricEntity } from '../../../types';
import type { RunsChartsRunData } from '../components/RunsCharts.common';
import { RunsChartsLineChartXAxisType } from '../components/RunsCharts.common';
import type { RunsMetricsLinePlotProps } from '../components/RunsMetricsLinePlot';
import type { RunsChartsLineChartExpression } from '../runs-charts.types';

const getCompareValue = (element: MetricEntity, xAxisKey: RunsMetricsLinePlotProps['xAxisKey']) => {
  if (xAxisKey === RunsChartsLineChartXAxisType.STEP) {
    // If xAxisKey is steps, we match along step
    return element.step;
  } else if (
    xAxisKey === RunsChartsLineChartXAxisType.TIME ||
    xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE
  ) {
    // If xAxisKey is time, we match along timestamp
    return element.timestamp;
  }
  return null;
};

export const getExpressionChartsSortedMetricHistory = ({
  expression,
  runEntry,
  evaluateExpression,
  xAxisKey,
}: {
  expression: RunsChartsLineChartExpression;
  runEntry: Omit<RunsChartsRunData, 'metrics' | 'params' | 'tags' | 'images'>;
  evaluateExpression: (
    expression: RunsChartsLineChartExpression,
    variables: Record<string, number>,
  ) => number | undefined;
  xAxisKey: RunsMetricsLinePlotProps['xAxisKey'];
}) => {
  const xDataPointsPerVariable = expression.variables.map((variable) => {
    const elements = runEntry.metricsHistory?.[variable];
    if (elements !== undefined) {
      return elements.flatMap((element) => {
        const compareValue = getCompareValue(element, xAxisKey);
        if (compareValue !== null) {
          return [compareValue];
        }
        return [];
      });
    }
    return [];
  });
  const xDataPoints = intersection(...xDataPointsPerVariable);
  xDataPoints.sort((a, b) => a - b); // ascending

  // For each xDataPoint, extract variables and evaluate expression
  return xDataPoints.flatMap((xDataPoint) => {
    const variables = expression.variables.reduce((obj, variable) => {
      const elements = runEntry.metricsHistory?.[variable];
      const value = elements?.find((element) => getCompareValue(element, xAxisKey) === xDataPoint)?.value;
      if (value !== undefined) {
        obj[variable] = value;
      }
      return obj;
    }, {} as Record<string, number>);

    const expressionValue = evaluateExpression(expression, variables);
    if (expressionValue !== undefined) {
      return [
        {
          value: expressionValue,
          key: expression.expression,
          ...(xAxisKey === RunsChartsLineChartXAxisType.STEP || xAxisKey === RunsChartsLineChartXAxisType.METRIC
            ? { step: xDataPoint, timestamp: 0 }
            : { timestamp: xDataPoint, step: 0 }),
        },
      ];
    }
    return [];
  });
};
