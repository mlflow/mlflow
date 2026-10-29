import { describe, expect, test } from '@jest/globals';

import {
  MLFLOW_ISSUE_DETECTION_ERROR_TAG,
  MLFLOW_ISSUE_DETECTION_TRIGGER_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG,
} from '../../constants';
import type { RunEntity } from '../../types';
import {
  DETECTION_RUN_FILTER,
  detectionRunDurationMs,
  isDetectionRunSettled,
  toDetectionRunRow,
  type DetectionRunRow,
} from './detectionRunHistory';

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

describe('DETECTION_RUN_FILTER', () => {
  test('matches on the run type, which every detection run carries', () => {
    // Filtering on the trigger tag instead would find the scheduled runs and miss the ad-hoc
    // ones, and `!=` cannot express the complement: a run-tag inequality drops runs with no such
    // tag, which is exactly the ad-hoc case.
    expect(DETECTION_RUN_FILTER).toBe("tags.`mlflow.runType` = 'issue_detection'");
  });
});

describe('isDetectionRunSettled', () => {
  test.each([
    ['FINISHED', true],
    ['FAILED', true],
    ['KILLED', true],
    ['RUNNING', false],
    ['SCHEDULED', false],
    [undefined, false],
    // Terminal-looking is the dangerous guess: a row read as settled stops polling and never
    // updates again, and its issue count is published as final.
    ['SOMETHING_NEW', false],
  ])('%s is settled: %s', (status, expected) => {
    expect(isDetectionRunSettled(status as string | undefined)).toBe(expected);
  });
});

describe('toDetectionRunRow', () => {
  test('reads what a scheduled run recorded', () => {
    const row = toDetectionRunRow(
      run({
        tags: {
          [MLFLOW_ISSUE_DETECTION_TRIGGER_TAG]: 'schedule',
          // What was submitted, and what the sampler left to analyze. The row carries the second.
          total_traces: '640',
          total_traces_analyzed: '312',
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
      status: 'FINISHED',
      trigger: 'schedule',
      windowStartMs: START_MS - 604_800_000,
      windowEndMs: START_MS,
      analyzedTraces: 312,
      model: 'openai:/gpt-5-mini',
      costUsd: 0.42,
      error: undefined,
    });
  });

  test('a run with no trigger tag is ad hoc, and covers no window', () => {
    // Ad-hoc detection is given an explicit set of trace IDs rather than a stretch of time, so
    // the absence of both the tag and the bounds is what that run actually was.
    const row = toDetectionRunRow(run({ tags: { total_traces_analyzed: '40', model: 'openai:/gpt-5-mini' } }));

    expect(row?.trigger).toBe('adhoc');
    expect(row?.windowStartMs).toBeUndefined();
    expect(row?.windowEndMs).toBeUndefined();
    expect(row?.analyzedTraces).toBe(40);
  });

  test('a trigger value this build does not know reads as ad hoc', () => {
    // Only "schedule" means the dispatcher fired it. Anything else is something a caller asked
    // for, which is what "ad hoc" says.
    expect(toDetectionRunRow(run({ tags: { [MLFLOW_ISSUE_DETECTION_TRIGGER_TAG]: 'webhook' } }))?.trigger).toBe(
      'adhoc',
    );
  });

  test('the run status is carried through verbatim', () => {
    // Not normalized into a vocabulary of this page's own: the Evaluation runs table reads the
    // same field, and the two must not be able to describe one run differently.
    expect(toDetectionRunRow(run({ status: 'KILLED' }))?.status).toBe('KILLED');
  });

  test('a run still going has no end time', () => {
    // The proto renders an absent end time as 0 rather than omitting it, so zero has to read as
    // "not finished" -- taken literally it would date the run to 1970 and show a 56-year duration.
    const row = toDetectionRunRow(run({ status: 'RUNNING', endTime: 0 }));

    expect(row?.endedAtMs).toBeUndefined();
    expect(isDetectionRunSettled(row?.status)).toBe(false);
  });

  test('a failed run carries why', () => {
    const row = toDetectionRunRow(
      run({ status: 'FAILED', tags: { [MLFLOW_ISSUE_DETECTION_ERROR_TAG]: 'MlflowException: no API key' } }),
    );

    expect(row?.error).toBe('MlflowException: no API key');
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
    expect(toDetectionRunRow(run({ tags: { total_traces_analyzed: value } }))?.analyzedTraces).toBe(expected);
  });

  test('a cost of Python None does not read as a number', () => {
    // `set_tag` stringifies whatever it is given, and the pipeline's cost is optional.
    expect(toDetectionRunRow(run({ tags: { total_cost_usd: 'None' } }))?.costUsd).toBeUndefined();
  });

  test('a run with no id is dropped rather than rendered without one', () => {
    // Built without the key rather than with `undefined`, which the helper's default would fill in.
    const idless = { info: { status: 'FINISHED', startTime: START_MS }, data: { tags: [] } } as unknown as RunEntity;

    expect(toDetectionRunRow(idless)).toBeUndefined();
  });

  test('a run with no start time is dropped', () => {
    expect(toDetectionRunRow(run({ startTime: 0 }))).toBeUndefined();
  });
});

describe('detectionRunDurationMs', () => {
  const row = (overrides: Partial<DetectionRunRow>): DetectionRunRow => ({
    runId: 'run-1',
    createdAtMs: START_MS,
    status: 'FINISHED',
    trigger: 'schedule',
    ...overrides,
  });

  test('measures a finished run end to end', () => {
    expect(detectionRunDurationMs(row({ endedAtMs: START_MS + 5_000 }), START_MS + 900_000)).toBe(5_000);
  });

  test('measures one still going against now', () => {
    expect(detectionRunDurationMs(row({ status: 'RUNNING' }), START_MS + 7_000)).toBe(7_000);
  });

  test('never reads as negative', () => {
    // Clock skew between the server that stamped the run and this browser would otherwise show a
    // run that has not started yet as having run for minus two minutes.
    expect(detectionRunDurationMs(row({ status: 'RUNNING' }), START_MS - 120_000)).toBe(0);
  });
});
