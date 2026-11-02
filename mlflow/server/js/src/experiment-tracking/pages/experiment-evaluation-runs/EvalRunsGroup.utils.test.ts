import { describe, expect, test } from '@jest/globals';
import type { RunEntity } from '../../types';
import { getEvalRunsGroupMetricMean } from './EvalRunsGroup.utils';

const runWithMetric = (value?: number) =>
  ({
    data: { metrics: value === undefined ? [] : [{ key: 'accuracy', value }] },
  }) as unknown as RunEntity;

describe('evaluation-run group metric calculations', () => {
  test('averages the runs that recorded the metric', () => {
    expect(getEvalRunsGroupMetricMean([runWithMetric(0.8), runWithMetric(), runWithMetric(1)], 'accuracy')).toBe(0.9);
    expect(getEvalRunsGroupMetricMean([runWithMetric()], 'accuracy')).toBeUndefined();
  });

  test('includes every run of the flat group in the average', () => {
    expect(getEvalRunsGroupMetricMean([runWithMetric(0.6), runWithMetric(0.9)], 'accuracy')).toBe(0.75);
  });
});
