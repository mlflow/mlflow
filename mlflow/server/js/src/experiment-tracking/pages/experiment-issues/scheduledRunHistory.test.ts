import { describe, expect, test } from '@jest/globals';

import {
  MLFLOW_ISSUE_DETECTION_ERROR_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG,
} from '../../constants';
import type { RunEntity } from '../../types';
import {
  SCHEDULED_DETECTION_RUN_FILTER,
  scheduledRunDurationMs,
  scheduledRunOutcome,
  summarizeScheduledRuns,
  toScheduledRunRow,
  type ScheduledRunRow,
} from './scheduledRunHistory';

const START_MS = 1_800_000_000_000;

const run = ({
  status = 'FINISHED',
  startTime = START_MS,
  endTime = START_MS + 192_000,
  tags = {},
  runUuid = 'run-1',
}: {
  status?: string;
  startTime?: number;
  endTime?: number;
  tags?: Record<string, string>;
  runUuid?: string;
} = {}) =>
  ({
    info: { runUuid, status, startTime, endTime },
    data: { tags: Object.entries(tags).map(([key, value]) => ({ key, value })) },
  }) as unknown as RunEntity;

describe('SCHEDULED_DETECTION_RUN_FILTER', () => {
  test('matches on the trigger tag, which only a scheduled occurrence carries', () => {
    // Filtering on the run type instead would also return every manual detection run.
    expect(SCHEDULED_DETECTION_RUN_FILTER).toBe("tags.`mlflow.issueDetection.trigger` = 'schedule'");
  });
});

describe('scheduledRunOutcome', () => {
  test.each([
    ['FINISHED', 12, 'succeeded'],
    // A finished run that analyzed nothing fired over a window with no traces it could use.
    // Reading it as "Succeeded" would claim an analysis that never happened.
    ['FINISHED', 0, 'quiet'],
    ['FAILED', 4, 'failed'],
    // Somebody stopping an occurrence is not the same event as one breaking.
    ['KILLED', 4, 'cancelled'],
    ['RUNNING', 4, 'running'],
    ['SCHEDULED', undefined, 'running'],
  ])('%s over %s traces is %s', (status, traces, expected) => {
    expect(scheduledRunOutcome(status, traces as number | undefined)).toBe(expected);
  });

  test('a finished run with no trace count is not read as quiet', () => {
    // An absent count is "not recorded", which is not the same as zero, and calling it quiet
    // would tell someone their traces were never analyzed when they were.
    expect(scheduledRunOutcome('FINISHED', undefined)).toBe('succeeded');
  });

  test('a status this build does not know reads as still going', () => {
    // Terminal-looking is the dangerous guess: a row frozen as "Succeeded" never updates again.
    expect(scheduledRunOutcome('SOMETHING_NEW', 1)).toBe('running');
  });
});

describe('toScheduledRunRow', () => {
  test('reads what the occurrence recorded', () => {
    const row = toScheduledRunRow(
      run({
        tags: {
          total_traces: '312',
          total_cost_usd: '0.42',
          model: 'openai:/gpt-5-mini',
          [MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG]: String(START_MS - 604_800_000),
          [MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG]: String(START_MS),
        },
      }),
    );

    expect(row).toEqual({
      runId: 'run-1',
      createdAtMs: START_MS,
      endedAtMs: START_MS + 192_000,
      windowStartMs: START_MS - 604_800_000,
      windowEndMs: START_MS,
      outcome: 'succeeded',
      scannedTraces: 312,
      model: 'openai:/gpt-5-mini',
      costUsd: 0.42,
      error: undefined,
    });
  });

  test('a run still going has no end time', () => {
    // The proto renders an absent end time as 0 rather than omitting it, so zero has to read as
    // "not finished" -- taken literally it would date the run to 1970 and show a 56-year duration.
    const row = toScheduledRunRow(run({ status: 'RUNNING', endTime: 0 }));

    expect(row?.endedAtMs).toBeUndefined();
    expect(row?.outcome).toBe('running');
  });

  test('a failed run carries why', () => {
    const row = toScheduledRunRow(
      run({ status: 'FAILED', tags: { [MLFLOW_ISSUE_DETECTION_ERROR_TAG]: 'MlflowException: no API key' } }),
    );

    expect(row?.error).toBe('MlflowException: no API key');
  });

  test('an occurrence from before the window was recorded still reads', () => {
    const row = toScheduledRunRow(run({ tags: { total_traces: '5' } }));

    expect(row?.windowStartMs).toBeUndefined();
    expect(row?.scannedTraces).toBe(5);
  });

  test.each([
    // Tag values are free-form strings the server stores verbatim, so each of these is reachable
    // by hand -- and `Number('')` is 0, `Number(' 7 ')` is 7, both of which would read as counts.
    ['', undefined],
    ['   ', undefined],
    ['not a number', undefined],
    ['-3', undefined],
    ['1.5', undefined],
    ['1e3', 1000],
  ])('a trace count of %p reads as %p', (value, expected) => {
    expect(toScheduledRunRow(run({ tags: { total_traces: value } }))?.scannedTraces).toBe(expected);
  });

  test('a cost of Python None does not read as a number', () => {
    // `set_tag` stringifies whatever it is given, and the pipeline's cost is optional.
    expect(toScheduledRunRow(run({ tags: { total_cost_usd: 'None' } }))?.costUsd).toBeUndefined();
  });

  test('a run with no id is dropped rather than rendered without one', () => {
    // Built without the key rather than with `undefined`, which the helper's default would fill in.
    const idless = { info: { status: 'FINISHED', startTime: START_MS }, data: { tags: [] } } as unknown as RunEntity;

    expect(toScheduledRunRow(idless)).toBeUndefined();
  });

  test('a run with no start time is dropped', () => {
    expect(toScheduledRunRow(run({ startTime: 0 }))).toBeUndefined();
  });
});

describe('scheduledRunDurationMs', () => {
  const row = (overrides: Partial<ScheduledRunRow>): ScheduledRunRow => ({
    runId: 'run-1',
    createdAtMs: START_MS,
    outcome: 'succeeded',
    ...overrides,
  });

  test('measures a finished occurrence end to end', () => {
    expect(scheduledRunDurationMs(row({ endedAtMs: START_MS + 5_000 }), START_MS + 900_000)).toBe(5_000);
  });

  test('measures one still going against now', () => {
    expect(scheduledRunDurationMs(row({ outcome: 'running' }), START_MS + 7_000)).toBe(7_000);
  });

  test('never reads as negative', () => {
    // Clock skew between the server that stamped the run and this browser would otherwise show a
    // run that has not started yet as having run for minus two minutes.
    expect(scheduledRunDurationMs(row({ outcome: 'running' }), START_MS - 120_000)).toBe(0);
  });
});

describe('summarizeScheduledRuns', () => {
  const DAY = 24 * 60 * 60 * 1000;
  const occurrence = (daysAgo: number, overrides: Partial<ScheduledRunRow> = {}): ScheduledRunRow => ({
    runId: `run-${daysAgo}`,
    createdAtMs: START_MS - daysAgo * DAY,
    outcome: 'succeeded',
    scannedTraces: 100,
    costUsd: 0.5,
    ...overrides,
  });

  test('totals only the occurrences inside the window', () => {
    const summary = summarizeScheduledRuns([occurrence(1), occurrence(29), occurrence(31)], START_MS);

    expect(summary.runCount).toBe(2);
    expect(summary.tracesScanned).toBe(200);
    expect(summary.spendUsd).toBe(1);
  });

  test('only the occurrences that ran to completion count toward traces scanned', () => {
    // A failed, cancelled or running occurrence is tagged with the sample it set out to analyze.
    // Totalling those would report abandoned work as work that was done.
    const summary = summarizeScheduledRuns(
      [
        occurrence(1, { outcome: 'running', scannedTraces: 268 }),
        occurrence(2, { outcome: 'failed', scannedTraces: 184 }),
        occurrence(3, { outcome: 'cancelled', scannedTraces: 96 }),
        occurrence(4, { outcome: 'quiet', scannedTraces: 0 }),
        occurrence(5, { outcome: 'succeeded', scannedTraces: 312 }),
      ],
      START_MS,
    );

    expect(summary.tracesScanned).toBe(312);
    // Every one of them is still an occurrence that fired.
    expect(summary.runCount).toBe(5);
  });

  test('an occurrence missing a count or a cost does not poison the totals', () => {
    const summary = summarizeScheduledRuns(
      [occurrence(1, { scannedTraces: undefined, costUsd: undefined }), occurrence(2)],
      START_MS,
    );

    expect(summary.tracesScanned).toBe(100);
    expect(summary.spendUsd).toBe(0.5);
  });

  test('the last outcome is the newest one that has settled', () => {
    // A running occurrence has no outcome yet, so reporting it would leave "Last outcome"
    // reading Running for as long as one is in flight.
    const summary = summarizeScheduledRuns(
      [occurrence(0, { outcome: 'running' }), occurrence(7, { outcome: 'failed' }), occurrence(14)],
      START_MS,
    );

    expect(summary.lastOutcome).toBe('failed');
  });

  test('no settled occurrence leaves the last outcome unknown', () => {
    expect(summarizeScheduledRuns([occurrence(0, { outcome: 'running' })], START_MS).lastOutcome).toBeUndefined();
  });

  test('an empty history totals to nothing rather than to NaN', () => {
    expect(summarizeScheduledRuns([], START_MS)).toEqual({
      runCount: 0,
      tracesScanned: 0,
      spendUsd: 0,
      lastOutcome: undefined,
    });
  });
});
