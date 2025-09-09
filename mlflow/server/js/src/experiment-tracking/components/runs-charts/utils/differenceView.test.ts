import type { KeyValueEntity } from '../../../../common/types';
import type { RunsChartsDifferenceCardConfig } from '../runs-charts.types';
import { DifferenceCardConfigCompareGroup, RunsChartType } from '../runs-charts.types';
import {
  getDifferenceChartDisplayedValue,
  differenceView,
  isDifferent,
  getDifferenceViewDataGroups,
  DifferenceChartCellDirection,
} from './differenceView';

describe('getFixedPointValue correctly rounds numbers and parses strings', () => {
  test('should convert number to 2 decimal points', () => {
    expect(getDifferenceChartDisplayedValue(3.33333333)).toEqual('3.33');
  });

  test('should convert number to 5 decimal points', () => {
    expect(getDifferenceChartDisplayedValue(3.123456, 5)).toEqual('3.12346');
  });

  test('should not change strings', () => {
    expect(getDifferenceChartDisplayedValue('this is a string', 5)).toEqual('this is a string');
  });
});

describe('differenceView correctly displays text indicating a positive or negative difference', () => {
  test('should display positive text with +', () => {
    expect(differenceView(3.33333333, 1)).toEqual({ label: '+2.33', direction: DifferenceChartCellDirection.POSITIVE });
  });

  test('should display negative text with -', () => {
    expect(differenceView(1, 3.12345)).toEqual({ label: '-2.12', direction: DifferenceChartCellDirection.NEGATIVE });
  });

  test('should ignore strings', () => {
    expect(differenceView('hi', 3.12345)).toEqual(undefined);
  });
});

describe('isDifferent correctly indicates whether two values are different', () => {
  test('should indicate different types as different', () => {
    expect(isDifferent(3.33333333, 'hi')).toEqual(true);
    expect(isDifferent('hi', 12345)).toEqual(true);
  });

  test('should indicate different numbers as different', () => {
    expect(isDifferent(3.33333333, 1)).toEqual(true);
    expect(isDifferent(1, 3.12345)).toEqual(true);
  });

  test('should indicate different strings as different', () => {
    expect(isDifferent('hi', 'hello')).toEqual(true);
  });

  test('should indicate the same strings as not different', () => {
    expect(isDifferent('hi', 'hi')).toEqual(false);
  });

  test('should indicate the same numbers as not different', () => {
    expect(isDifferent(3.33333333, 3.33333333)).toEqual(false);
  });
});

describe('getDifferenceViewDataGroups correctly groups metrics, system metrics, and parameters', () => {
  const cardConfig: RunsChartsDifferenceCardConfig = {
    type: RunsChartType.DIFFERENCE,
    chartName: 'someChartName',
    showChangeFromBaseline: true,
    showDifferencesOnly: false,
    compareGroups: [
      DifferenceCardConfigCompareGroup.MODEL_METRICS,
      DifferenceCardConfigCompareGroup.SYSTEM_METRICS,
      DifferenceCardConfigCompareGroup.PARAMETERS,
    ],
    deleted: false,
    isGenerated: false,
    baselineColumnUuid: 'runUuid1',
  };

  const previewData = [
    {
      uuid: 'runUuid1',
      displayName: 'Run 1', // Add displayName property
      params: {
        algorithm: { key: 'algorithm', value: 'linear' },
      }, // Add params property
      metrics: {
        metric1: { key: 'metric1', value: 1, step: 1, timestamp: 0 },
        'system/metric': { key: 'system/metric', value: 1, step: 1, timestamp: 0 },
      },
      tags: {
        tag1: { key: 'tag1', value: 'val1' } as KeyValueEntity,
        'mlflow.user': { key: 'mlflow.user', value: 'user1' } as KeyValueEntity,
      },
      images: {},
    },
    {
      uuid: 'runUuid2',
      user_id: 'user1',
      displayName: 'Run 2', // Add displayName property
      params: {
        algorithm: { key: 'algorithm', value: 'logistic' },
      }, // Add params property
      metrics: {
        metric1: { key: 'metric1', value: 2, step: 1, timestamp: 0 },
        'system/metric': { key: 'system/metric', value: 2, step: 1, timestamp: 0 },
      },
      tags: {
        tag1: { key: 'tag1', value: 'val2' } as KeyValueEntity,
        'mlflow.user': { key: 'mlflow.user', value: 'user1' } as KeyValueEntity,
      },
      images: {},
    },
  ];

  test('should group metrics', () => {
    const result = getDifferenceViewDataGroups(previewData, cardConfig, 'headingColumn', null);
    expect(result.modelMetrics).toEqual([
      {
        headingColumn: 'metric1',
        runUuid1: 1,
        runUuid2: 2,
      },
    ]);
    expect(result.systemMetrics).toEqual([
      {
        headingColumn: 'system/metric',
        runUuid1: 1,
        runUuid2: 2,
      },
    ]);
    expect(result.parameters).toEqual([
      {
        headingColumn: 'algorithm',
        runUuid1: 'linear',
        runUuid2: 'logistic',
      },
    ]);
    expect(result.tags).toEqual([
      {
        headingColumn: 'tag1',
        runUuid1: 'val1',
        runUuid2: 'val2',
      },
    ]);
  });
});
