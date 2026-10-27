/**
 * Reading a scheduled detection occurrence off the run it recorded.
 *
 * Every occurrence that fires creates one run, so the history of a schedule is a run search. The
 * run carries what the occurrence did as tags; this module turns those strings back into a row,
 * and is deliberately pure so the parsing is testable without a server.
 */

import type { RunEntity } from '../../types';
import {
  MLFLOW_ISSUE_DETECTION_ERROR_TAG,
  MLFLOW_ISSUE_DETECTION_TRIGGER_TAG,
  MLFLOW_ISSUE_DETECTION_TRIGGER_VALUE_SCHEDULE,
  MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG,
} from '../../constants';

/**
 * Finds exactly the scheduled occurrences.
 *
 * Only the scheduled job writes this tag, so equality is enough -- unlike filtering detection
 * runs by run type, which would also match every manual run.
 */
export const SCHEDULED_DETECTION_RUN_FILTER = `tags.\`${MLFLOW_ISSUE_DETECTION_TRIGGER_TAG}\` = '${MLFLOW_ISSUE_DETECTION_TRIGGER_VALUE_SCHEDULE}'`;

/** How far back the figures above the table reach. */
export const SUMMARY_WINDOW_DAYS = 30;

/**
 * What an occurrence turned out to be.
 *
 * `quiet` is a finished run that analyzed nothing: the schedule fired over a window holding no
 * traces it could use. It is a success in run terms, but reading it as "Succeeded" would suggest
 * an analysis that never happened.
 */
export type ScheduledRunOutcome = 'running' | 'succeeded' | 'quiet' | 'failed' | 'cancelled';

export interface ScheduledRunRow {
  runId: string;
  /** When the job created the run, which is when the occurrence began work. */
  createdAtMs: number;
  /** Absent while the occurrence is still going. */
  endedAtMs?: number;
  /** The trace window the occurrence covered. Absent on an occurrence from before it was recorded. */
  windowStartMs?: number;
  windowEndMs?: number;
  outcome: ScheduledRunOutcome;
  /** Traces sampled for analysis, which the sampler caps -- not the number the window held. */
  scannedTraces?: number;
  model?: string;
  costUsd?: number;
  /** Why it failed, when the run recorded a reason. */
  error?: string;
}

const readTag = (run: RunEntity, key: string): string | undefined =>
  run.data?.tags?.find((tag) => tag.key === key)?.value;

/**
 * A tag value as a whole count, or `undefined` for anything that is not one.
 *
 * Tag values are free-form strings the server stores verbatim, so every one of these is reachable
 * from a hand-written tag: `Number('')` is 0 and `Number(' 12 ')` is 12, which would both read as
 * real counts, and a `total_cost_usd` of Python `None` arrives as the string `"None"`.
 */
const parseCount = (value: string | undefined): number | undefined => {
  if (value === undefined || value.trim() === '') {
    return undefined;
  }
  const count = Number(value);
  return Number.isSafeInteger(count) && count >= 0 ? count : undefined;
};

const parseAmount = (value: string | undefined): number | undefined => {
  if (value === undefined || value.trim() === '') {
    return undefined;
  }
  const amount = Number(value);
  return Number.isFinite(amount) && amount >= 0 ? amount : undefined;
};

const parseTimestamp = (value: string | undefined): number | undefined => {
  const parsed = parseCount(value);
  // A zero-epoch bound carries no information and would render as 1970, so it reads as absent.
  return parsed === undefined || parsed === 0 ? undefined : parsed;
};

/**
 * An epoch-ms time the run entity carries, or `undefined` when it has none.
 *
 * A run still going has no end time, which the proto renders as `0` rather than omitting it, so
 * zero has to mean absent here as well.
 */
const liveTimestamp = (value: number | undefined): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined;

/**
 * Which outcome a run's status and trace count describe.
 *
 * `KILLED` is its own outcome rather than a failure: the Issues tab can cancel a detection run,
 * including a scheduled one, and somebody stopping an occurrence on purpose is not the same event
 * as one that broke.
 */
export const scheduledRunOutcome = (status: string | undefined, scannedTraces: number | undefined) => {
  switch (status) {
    case 'FAILED':
      return 'failed';
    case 'KILLED':
      return 'cancelled';
    case 'FINISHED':
      return scannedTraces === 0 ? 'quiet' : 'succeeded';
    default:
      // RUNNING, SCHEDULED, and anything a newer server introduces: work that has not landed.
      return 'running';
  }
};

/** The row one run describes, or `undefined` for a run with no usable identity or start time. */
export const toScheduledRunRow = (run: RunEntity): ScheduledRunRow | undefined => {
  const runId = run.info?.runUuid;
  const createdAtMs = liveTimestamp(run.info?.startTime);
  if (!runId || createdAtMs === undefined) {
    return undefined;
  }
  const scannedTraces = parseCount(readTag(run, 'total_traces'));
  return {
    runId,
    createdAtMs,
    endedAtMs: liveTimestamp(run.info?.endTime),
    windowStartMs: parseTimestamp(readTag(run, MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG)),
    windowEndMs: parseTimestamp(readTag(run, MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG)),
    outcome: scheduledRunOutcome(run.info?.status, scannedTraces),
    scannedTraces,
    model: readTag(run, 'model') || undefined,
    costUsd: parseAmount(readTag(run, 'total_cost_usd')),
    error: readTag(run, MLFLOW_ISSUE_DETECTION_ERROR_TAG) || undefined,
  };
};

/** How long an occurrence took, or how long it has been going when it has not finished. */
export const scheduledRunDurationMs = (row: ScheduledRunRow, nowMs: number): number =>
  Math.max(0, (row.endedAtMs ?? nowMs) - row.createdAtMs);

/** The outcomes that finished the analysis they started, which is what "scanned" can total. */
const COMPLETED_OUTCOMES: ScheduledRunOutcome[] = ['succeeded', 'quiet'];

export interface ScheduledRunsSummary {
  /** Occurrences inside the summary window, whatever became of them. */
  runCount: number;
  /**
   * Traces the occurrences that ran to completion scanned.
   *
   * A failed, cancelled or still-running occurrence is tagged with the size of the sample it set
   * out to analyze, which is not what it got through -- totalling those would report work that
   * was abandoned as work that was done.
   */
  tracesScanned: number;
  spendUsd: number;
  /** The newest occurrence that has finished, which is what "last outcome" means. */
  lastOutcome?: ScheduledRunOutcome;
}

/**
 * The figures above the table, over the last {@link SUMMARY_WINDOW_DAYS} days.
 *
 * Counts only what the rows loaded so far say, like the run count beside the table: a run search
 * returns no totals, so anything else here would be a guess presented as a fact.
 */
export const summarizeScheduledRuns = (
  rows: ScheduledRunRow[],
  nowMs: number,
  windowDays: number = SUMMARY_WINDOW_DAYS,
): ScheduledRunsSummary => {
  const since = nowMs - windowDays * 24 * 60 * 60 * 1000;
  const recent = rows.filter((row) => row.createdAtMs >= since);
  return {
    runCount: recent.length,
    tracesScanned: recent
      .filter((row) => COMPLETED_OUTCOMES.includes(row.outcome))
      .reduce((total, row) => total + (row.scannedTraces ?? 0), 0),
    spendUsd: recent.reduce((total, row) => total + (row.costUsd ?? 0), 0),
    // Ordered newest first by the search, so the first settled row is the latest one.
    lastOutcome: rows.find((row) => row.outcome !== 'running')?.outcome,
  };
};
