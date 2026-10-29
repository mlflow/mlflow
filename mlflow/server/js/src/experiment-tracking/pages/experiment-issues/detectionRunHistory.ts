/**
 * Reading an issue detection run off the run entity it recorded.
 *
 * Every detection run is an MLflow run -- the ones a schedule fired and the ones somebody started
 * from the Issues tab alike -- so the history is a run search. The run carries what it did as
 * tags; this module turns those strings back into a row, and is deliberately pure so the parsing
 * is testable without a server.
 */

import type { RunEntity } from '../../types';
import {
  MLFLOW_ISSUE_DETECTION_ERROR_TAG,
  MLFLOW_ISSUE_DETECTION_TRIGGER_TAG,
  MLFLOW_ISSUE_DETECTION_TRIGGER_VALUE_SCHEDULE,
  MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG,
  MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG,
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
} from '../../constants';

/**
 * Finds every detection run, however it was started.
 *
 * On the run type rather than on the trigger tag, because the trigger tag is only written by the
 * scheduled job: selecting on it would find the scheduled runs and miss the ad-hoc ones, and
 * `!=` cannot be used to express the complement -- a run-tag inequality drops runs that carry no
 * such tag, which is exactly the ad-hoc case.
 */
export const DETECTION_RUN_FILTER = `tags.\`${MLFLOW_RUN_TYPE_TAG}\` = '${MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION}'`;

/**
 * How the run was started.
 *
 * `adhoc` is the absence of the schedule marker rather than a marker of its own, so a detection
 * run from before the trigger tag existed reads as what it was: something a person asked for.
 */
export type DetectionRunTrigger = 'schedule' | 'adhoc';

export interface DetectionRunRow {
  runId: string;
  /** When the job created the run, which is when it began work. */
  createdAtMs: number;
  /** Absent while the run is still going. */
  endedAtMs?: number;
  /**
   * The run's own status, verbatim, so this page and the Evaluation runs table can never describe
   * the same run two ways. Empty for a run whose status could not be read.
   */
  status: string;
  trigger: DetectionRunTrigger;
  /** The trace window a scheduled run covered. An ad-hoc run has none: it is given trace IDs. */
  windowStartMs?: number;
  windowEndMs?: number;
  /**
   * How many traces the model was shown.
   *
   * The count the run recorded when it finished, not the number handed to it: the sampler caps
   * what it analyzes, so a run given 200 sessionless traces analyzes 100. Absent on a run still
   * going, and on a run from before that count was recorded.
   */
  analyzedTraces?: number;
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

/** Statuses that mean the run is over, so the figures beside it are final. */
const SETTLED_RUN_STATUSES = ['FINISHED', 'FAILED', 'KILLED'];

/**
 * Whether the run has stopped.
 *
 * Anything else -- `RUNNING`, `SCHEDULED`, or a status a newer server introduces -- is treated as
 * work that has not landed, so an unrecognized status leaves the page polling rather than
 * reporting a result the run has not produced.
 */
export const isDetectionRunSettled = (status: string | undefined): boolean =>
  SETTLED_RUN_STATUSES.includes(status ?? '');

/** The row one run describes, or `undefined` for a run with no usable identity or start time. */
export const toDetectionRunRow = (run: RunEntity): DetectionRunRow | undefined => {
  const runId = run.info?.runUuid;
  const createdAtMs = liveTimestamp(run.info?.startTime);
  if (!runId || createdAtMs === undefined) {
    return undefined;
  }
  return {
    runId,
    createdAtMs,
    endedAtMs: liveTimestamp(run.info?.endTime),
    status: run.info?.status ?? '',
    trigger:
      readTag(run, MLFLOW_ISSUE_DETECTION_TRIGGER_TAG) === MLFLOW_ISSUE_DETECTION_TRIGGER_VALUE_SCHEDULE
        ? 'schedule'
        : 'adhoc',
    windowStartMs: parseTimestamp(readTag(run, MLFLOW_ISSUE_DETECTION_WINDOW_START_TAG)),
    windowEndMs: parseTimestamp(readTag(run, MLFLOW_ISSUE_DETECTION_WINDOW_END_TAG)),
    analyzedTraces: parseCount(readTag(run, 'total_traces_analyzed')),
    model: readTag(run, 'model') || undefined,
    costUsd: parseAmount(readTag(run, 'total_cost_usd')),
    error: readTag(run, MLFLOW_ISSUE_DETECTION_ERROR_TAG) || undefined,
  };
};

/** How long a run took, or how long it has been going when it has not finished. */
export const detectionRunDurationMs = (row: DetectionRunRow, nowMs: number): number =>
  Math.max(0, (row.endedAtMs ?? nowMs) - row.createdAtMs);
