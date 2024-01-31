import { findAbsoluteTimestampRangeForRelativeRange, findChartStepsByTimestamp } from './findChartStepsByTimestamp';

const existingMetricsRangeA = [
  { key: '', step: 3, timestamp: new Date('2023-10-24 13:00:00').valueOf() },
  { key: '', step: 2, timestamp: new Date('2023-10-24 12:00:00').valueOf() },
  { key: '', step: -1, timestamp: new Date('2023-10-24 09:00:00').valueOf() },
  { key: '', step: 5, timestamp: new Date('2023-10-24 15:00:00').valueOf() },
];

const existingMetricsRangeB = [
  { key: '', step: 7, timestamp: new Date('2023-10-24 17:00:00').valueOf() },
  { key: '', step: 8, timestamp: new Date('2023-10-24 18:00:00').valueOf() },
  { key: '', step: 6, timestamp: new Date('2023-10-24 16:00:00').valueOf() },
  { key: '', step: 4, timestamp: new Date('2023-10-24 14:00:00').valueOf() },
];

const ranges = {
  '-1,5': {
    metricsHistory: existingMetricsRangeA as any,
  },
  '4,8': {
    metricsHistory: existingMetricsRangeB as any,
  },
};

describe('findChartStepsByTimestamp', () => {
  test('should find proper ranges for middle timestamps', () => {
    expect(findChartStepsByTimestamp(ranges, ['2023-10-24 13:30:00', '2023-10-24 13:45:00'])).toEqual([3, 4]);
  });

  test('should find proper ranges for lower boundary timestamp', () => {
    expect(findChartStepsByTimestamp(ranges, ['2023-10-24 09:01:00', '2023-10-24 13:45:00'])).toEqual([-1, 4]);
  });

  test('should find proper ranges for upper boundary timestamp', () => {
    expect(findChartStepsByTimestamp(ranges, ['2023-10-24 13:30:00', '2023-10-24 17:59:00'])).toEqual([3, 8]);
  });

  test('should find proper ranges when exceeding boundaries', () => {
    expect(findChartStepsByTimestamp(ranges, ['2022-10-23 10:00:00', '2024-12-25 18:00:00'])).toEqual([-1, 8]);
  });

  test('should return nothing when history is insufficient', () => {
    expect(
      findChartStepsByTimestamp({ DEFAULT: { metricsHistory: [] } }, ['2022-10-23 10:00:00', '2024-12-25 18:00:00']),
    ).toBeUndefined();
  });
});
import { findChartStepsByTimestampForRuns } from './findChartStepsByTimestamp';

describe('findChartStepsByTimestampForRuns', () => {
  const sampledMetrics = {
    runUuid1: {
      metricKey1: {
        DEFAULT: {
          metricsHistory: [
            { key: 'metricKey1', step: 3, timestamp: new Date('2023-10-24 13:00:00').valueOf() },
            { key: 'metricKey1', step: 2, timestamp: new Date('2023-10-24 12:00:00').valueOf() },
            { key: 'metricKey1', step: 1, timestamp: new Date('2023-10-24 11:00:00').valueOf() },
            { key: 'metricKey1', step: 5, timestamp: new Date('2023-10-24 15:00:00').valueOf() },
          ],
        },
      },
    },
    runUuid2: {
      metricKey1: {
        DEFAULT: {
          metricsHistory: [
            { key: 'metricKey1', step: 2, timestamp: new Date('2023-10-24 12:00:00').valueOf() },
            { key: 'metricKey1', step: 4, timestamp: new Date('2023-10-24 14:00:00').valueOf() },
            { key: 'metricKey1', step: 6, timestamp: new Date('2023-10-24 16:00:00').valueOf() },
            { key: 'metricKey1', step: 8, timestamp: new Date('2023-10-24 18:00:00').valueOf() },
          ],
        },
      },
    },
    runUuid3: {
      metricKey1: {
        DEFAULT: {
          metricsHistory: [
            { key: 'metricKey1', step: 1, timestamp: new Date('2023-10-26 12:00:00').valueOf() },
            { key: 'metricKey1', step: 33, timestamp: new Date('2023-10-26 14:00:00').valueOf() },
            { key: 'metricKey1', step: 66, timestamp: new Date('2023-10-26 16:00:00').valueOf() },
            { key: 'metricKey1', step: 99, timestamp: new Date('2023-10-26 18:00:00').valueOf() },
          ],
        },
      },
    },
  } as any;

  test('should find proper step ranges for two runs', () => {
    const runUuids = ['runUuid1', 'runUuid2'];
    const metricKey = 'metricKey1';
    const range: [string, string] = ['2023-10-24 12:00:00', '2023-10-24 15:00:00'];
    const result = findChartStepsByTimestampForRuns(sampledMetrics, runUuids, metricKey, range);
    expect(result).toEqual([2, 6]);
  });

  test('should find proper step ranges for a single run', () => {
    const runUuids = ['runUuid1'];
    const metricKey = 'metricKey1';
    const range: [string, string] = ['2023-10-24 12:00:00', '2023-10-24 15:00:00'];
    const result = findChartStepsByTimestampForRuns(sampledMetrics, runUuids, metricKey, range);
    expect(result).toEqual([2, 5]);
  });

  test('should skip range of run that is not included in the timestamp range', () => {
    const runUuids = ['runUuid1', 'runUuid2', 'runUuid3'];
    const metricKey = 'metricKey1';
    const range: [string, string] = ['2023-10-24 12:00:00', '2023-10-24 15:00:00'];
    const result = findChartStepsByTimestampForRuns(sampledMetrics, runUuids, metricKey, range);
    expect(result).toEqual([2, 6]);
  });

  test('should skip all ranges of run that is not included in the timestamp range', () => {
    const runUuids = ['runUuid1', 'runUuid2', 'runUuid3'];
    const metricKey = 'metricKey1';
    const range: [string, string] = ['2023-10-28 12:00:00', '2023-10-29 15:00:00'];
    const result = findChartStepsByTimestampForRuns(sampledMetrics, runUuids, metricKey, range);
    expect(result).toEqual(undefined);
  });
});

describe('findAbsoluteTimestampRangeForRelativeRange', () => {
  const currentlyVisibleMetrics = {
    runUuid1: {
      runUuid: 'runUuid1',
      metric1: {
        metricsHistory: [
          { key: '', step: 3, timestamp: new Date('2023-10-24 13:00:00').valueOf() },
          { key: '', step: 2, timestamp: new Date('2023-10-24 12:00:00').valueOf() },
        ],
      },
      metric2: {
        metricsHistory: [
          { key: '', step: 1, timestamp: new Date('2023-10-24 11:00:00').valueOf() },
          { key: '', step: 5, timestamp: new Date('2023-10-24 15:00:00').valueOf() },
        ],
      },
    },
    runUuid2: {
      runUuid: 'runUuid2',
      metric1: {
        metricsHistory: [
          { key: '', step: 2, timestamp: new Date('2023-10-26 13:00:00').valueOf() },
          { key: '', step: 4, timestamp: new Date('2023-10-26 12:00:00').valueOf() },
        ],
      },
      metric2: {
        metricsHistory: [
          { key: '', step: 6, timestamp: new Date('2023-10-26 11:00:00').valueOf() },
          { key: '', step: 8, timestamp: new Date('2023-10-26 15:00:00').valueOf() },
        ],
      },
    },
  } as any;

  test('should return the correct timestamp range for relative range', () => {
    const runUuids = ['runUuid1', 'runUuid2'];
    const relativeRange: [number, number] = [3600, 7200];
    const result = findAbsoluteTimestampRangeForRelativeRange(currentlyVisibleMetrics, runUuids, relativeRange);
    expect(result).toEqual([new Date('2023-10-24 12:00:00').valueOf(), new Date('2023-10-26 13:00:00').valueOf()]);
  });
});
