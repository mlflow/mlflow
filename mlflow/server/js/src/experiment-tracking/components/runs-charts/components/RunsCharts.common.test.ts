import { MetricEntity } from '../../../types';
import { removeOutliersFromMetricHistory } from './RunsCharts.common';

describe('removeOutliersFromMetricHistory', () => {
  it.each([
    {
      label: 'all same and two outliers',
      makeData: (): MetricEntity[] => {
        const history = Array(20).fill({ step: 0, value: 100, key: 'metric', timestamp: 0 });
        history.push({ step: 0, value: 1000, key: 'metric', timestamp: 0 });
        history.unshift({ step: 0, value: 0, key: 'metric', timestamp: 0 });
        return history;
      },
      predicate: ({ value }: MetricEntity) => value === 100,
    },
    {
      label: 'linear progression',
      makeData: (): MetricEntity[] => Array(100).map((_, i) => ({ step: 0, value: i, key: 'metric', timestamp: 0 })),
      predicate: ({ value }: MetricEntity) => value >= 5 && value <= 95,
    },
  ])('should remove outliers correctly ($label)', ({ label, makeData, predicate }) => {
    const result = removeOutliersFromMetricHistory(makeData());
    expect(result.every(predicate)).toBeTruthy();
  });
});
