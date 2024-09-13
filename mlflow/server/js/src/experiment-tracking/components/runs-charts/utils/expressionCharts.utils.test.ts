import { renderHook } from '@testing-library/react';
import { RunsChartsLineChartXAxisType } from '../components/RunsCharts.common';
import { useChartExpressionParser } from '../hooks/useChartExpressionParser';
import { getExpressionChartsSortedMetricHistory } from './expressionCharts.utils';

describe('getExpressionChartsSortedMetricHistory', () => {
  const { result } = renderHook(() => useChartExpressionParser());
  const { evaluateExpression } = result.current;

  it('should return an empty array if runEntry has no metrics history', () => {
    const expression = {
      expression: 'metric1 + metric2',
      rpn: ['metric1', 'metric2', '+'],
      variables: ['metric1', 'metric2'],
    };
    const runEntry = {
      // No metrics history
      uuid: 'run-uuid',
      displayName: 'run-name',
    };
    const evaluateExpression = jest.fn();
    const xAxisKey = RunsChartsLineChartXAxisType.STEP;

    const result = getExpressionChartsSortedMetricHistory({
      expression,
      runEntry,
      evaluateExpression,
      xAxisKey,
    });

    expect(result).toEqual([]);
  });

  it('should return an empty array if no common xDataPoints found', () => {
    const expression = {
      expression: 'metric1 + metric2',
      rpn: ['metric1', 'metric2', '+'],
      variables: ['metric1', 'metric2'],
    };
    const runEntry = {
      uuid: 'run-uuid',
      displayName: 'run-name',
      metricsHistory: {
        metric1: [
          { key: 'metric1', value: 1, step: 1, timestamp: 0 },
          { key: 'metric1', value: 2, step: 2, timestamp: 1 },
        ],
        metric2: [
          { key: 'metric2', value: 3, step: 3, timestamp: 0 },
          { key: 'metric2', value: 4, step: 4, timestamp: 1 },
        ],
      },
    };
    const evaluateExpression = jest.fn();
    const xAxisKey = RunsChartsLineChartXAxisType.STEP;

    const result = getExpressionChartsSortedMetricHistory({
      expression,
      runEntry,
      evaluateExpression,
      xAxisKey,
    });

    expect(result).toEqual([]);
  });

  it('should evaluate the expression for each xDataPoint and return the results', () => {
    const expression = {
      expression: 'metric1 + metric2',
      rpn: ['metric1', 'metric2', '+'],
      variables: ['metric1', 'metric2'],
    };
    const runEntry = {
      uuid: 'run-uuid',
      displayName: 'run-name',
      metricsHistory: {
        metric1: [
          { value: 1, step: 1, timestamp: 0, key: 'metric1' },
          { value: 2, step: 2, timestamp: 0, key: 'metric1' },
        ],
        metric2: [
          { value: 3, step: 1, timestamp: 0, key: 'metric2' },
          { value: 4, step: 2, timestamp: 0, key: 'metric2' },
        ],
      },
    };

    const xAxisKey = RunsChartsLineChartXAxisType.STEP;

    const result = getExpressionChartsSortedMetricHistory({
      expression,
      runEntry,
      evaluateExpression,
      xAxisKey,
    });

    expect(result).toEqual([
      { value: 4, key: 'metric1 + metric2', step: 1, timestamp: 0 },
      { value: 6, key: 'metric1 + metric2', step: 2, timestamp: 0 },
    ]);
  });

  it('should group by timestamp when xAxisKey is TIME', () => {
    const expression = {
      expression: 'metric1 + metric2',
      rpn: ['metric1', 'metric2', '+'],
      variables: ['metric1', 'metric2'],
    };
    const runEntry = {
      uuid: 'run-uuid',
      displayName: 'run-name',
      metricsHistory: {
        metric1: [
          { value: 1, step: 1, timestamp: 0, key: 'metric1' },
          { value: 2, step: 2, timestamp: 1, key: 'metric1' },
        ],
        metric2: [
          { value: 3, step: 1, timestamp: 1, key: 'metric2' },
          { value: 4, step: 2, timestamp: 0, key: 'metric2' },
        ],
      },
    };
    const xAxisKey = RunsChartsLineChartXAxisType.TIME;

    const result = getExpressionChartsSortedMetricHistory({
      expression,
      runEntry,
      evaluateExpression,
      xAxisKey,
    });

    expect(result).toEqual([
      { value: 5, key: 'metric1 + metric2', step: 0, timestamp: 0 },
      { value: 5, key: 'metric1 + metric2', step: 0, timestamp: 1 },
    ]);
  });

  it('should group by timestamp when xAxisKey is TIME_RELATIVE', () => {
    const expression = {
      expression: 'metric1 + metric2',
      rpn: ['metric1', 'metric2', '+'],
      variables: ['metric1', 'metric2'],
    };
    const runEntry = {
      uuid: 'run-uuid',
      displayName: 'run-name',
      metricsHistory: {
        metric1: [
          { value: 1, step: 1, timestamp: 0, key: 'metric1' },
          { value: 2, step: 2, timestamp: 1, key: 'metric1' },
        ],
        metric2: [
          { value: 3, step: 1, timestamp: 1, key: 'metric2' },
          { value: 4, step: 2, timestamp: 0, key: 'metric2' },
        ],
      },
    };
    const xAxisKey = RunsChartsLineChartXAxisType.TIME_RELATIVE;

    const result = getExpressionChartsSortedMetricHistory({
      expression,
      runEntry,
      evaluateExpression,
      xAxisKey,
    });

    expect(result).toEqual([
      { value: 5, key: 'metric1 + metric2', step: 0, timestamp: 0 },
      { value: 5, key: 'metric1 + metric2', step: 0, timestamp: 1 },
    ]);
  });
});
