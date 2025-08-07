import { FINISHED_RUN_STATUSES } from '../constants';
import type { RunInfoEntity } from '../types';

/**
 * Type-safe check if a run status is considered "finished"
 * A finished run is one that has completed execution and cannot be modified.
 * This includes FINISHED (successful), FAILED (error), and KILLED (terminated) statuses.
 *
 * @param status - The run status to check
 * @returns true if the status indicates a finished run, false otherwise
 */
export const isFinishedRunStatus = (status: RunInfoEntity['status']): boolean => {
  return (FINISHED_RUN_STATUSES as readonly string[]).includes(status);
};

/**
 * Type-safe check if a run status is considered "active" (not finished)
 * Active runs are those that are still running or scheduled to run.
 *
 * @param status - The run status to check
 * @returns true if the status indicates an active run, false otherwise
 */
export const isActiveRunStatus = (status: RunInfoEntity['status']): boolean => {
  return !isFinishedRunStatus(status);
};

/**
 * Filter runs by finished/active status
 *
 * @param runs - Array of run info objects to filter
 * @param hideFinished - Whether to hide finished runs (show only active)
 * @returns Filtered array of runs
 */
export const filterRunsByStatus = (runs: RunInfoEntity[], hideFinished = false): RunInfoEntity[] => {
  if (!hideFinished) return runs;
  return runs.filter((run) => isActiveRunStatus(run.status));
};
