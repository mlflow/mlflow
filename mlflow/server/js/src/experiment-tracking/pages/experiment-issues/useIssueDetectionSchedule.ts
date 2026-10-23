import { useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { SERVER_FEATURE_KEYS, useFeatureEnabled } from '../../hooks/useServerInfo';
import {
  loadIssueDetectionSchedule,
  noIssueDetectionSchedule,
  saveIssueDetectionSchedule,
  type IssueDetectionSchedule,
  type IssueDetectionScheduleModel,
  type IssueDetectionScheduleSetting,
  type IssueDetectionScheduleUpdate,
  type ScheduleTraceFilter,
} from './issueDetectionSchedule';

export const ISSUE_DETECTION_SCHEDULE_QUERY_KEY = 'EXPERIMENT_ISSUE_DETECTION_SCHEDULE';

export type { IssueDetectionScheduleSetting } from './issueDetectionSchedule';

/**
 * Whether this server can run a schedule at all.
 *
 * Recurring detection needs the job system and its periodic tasks enabled behind a SQL backend
 * store, which `server-info` reports. Defaults to unavailable when the field is missing or the
 * read failed: the field ships with the scheduler itself, so a server that does not report it has
 * no scheduler either. Guessing the other way would offer a control that writes a schedule nothing
 * ever picks up, which fails silently -- worse than a disabled control that says why.
 */
export const useScheduledDetectionAvailable = (): boolean =>
  useFeatureEnabled(SERVER_FEATURE_KEYS.SCHEDULED_ISSUE_DETECTION, false);

export interface UseIssueDetectionScheduleResult {
  /** Current setting derived from the stored schedule (a paused schedule reads as 'off'). */
  setting: IssueDetectionScheduleSetting;
  /** Current trace filter clauses (persisted independently of pause) for prefilling the modal. */
  traceFilters: ScheduleTraceFilter[];
  /** The stored model selection, or undefined on a schedule that names none. */
  model: IssueDetectionScheduleModel | undefined;
  isLoading: boolean;
  isUpdating: boolean;
  error: Error | null;
  /**
   * Set when the stored schedule could not be read. Separate from `error` because the two call for
   * opposite handling: a failed save keeps the draft open to retry, while a failed read means the
   * values here are a fallback rather than what is stored, so editing has to be blocked -- saving
   * over them would replace a real schedule with this one's defaults.
   */
  loadError: Error | null;
  /** Retry the read, so a transient failure does not strand the control until a remount. */
  retryLoad: () => void;
  /**
   * Apply the desired cadence + clauses. Resolves once the writes succeed and rejects if any
   * fail, so the modal can await it -- keeping itself open on failure instead of closing and
   * silently discarding the user's draft.
   */
  setSetting: (update: IssueDetectionScheduleUpdate) => Promise<unknown>;
}

/**
 * Reads and mutates the experiment's recurring issue-detection schedule.
 *
 * The schedule lives in experiment tags that the server's dispatcher reads, so saving one is two
 * tag writes rather than a call to a schedule service.
 */
export const useIssueDetectionSchedule = ({
  experimentId,
}: {
  experimentId: string;
}): UseIssueDetectionScheduleResult => {
  const workspace = useActiveWorkspace();
  const queryClient = useQueryClient();
  const queryKey = [ISSUE_DETECTION_SCHEDULE_QUERY_KEY, workspace, experimentId];

  const {
    data: schedule,
    isLoading,
    error,
    refetch,
  } = useQuery<IssueDetectionSchedule, Error>({
    queryKey,
    queryFn: () => loadIssueDetectionSchedule(experimentId),
    enabled: Boolean(experimentId),
    refetchOnWindowFocus: false,
    retry: false,
  });

  const mutation = useMutation<IssueDetectionSchedule, Error, IssueDetectionScheduleUpdate>({
    // A read already in flight was issued before this write, so it answers with the schedule as it
    // was and would land after the cache update below, showing the cadence the user just replaced.
    onMutate: () => queryClient.cancelQueries({ queryKey }),
    // `schedule` is passed so turning the schedule off keeps the cadence it had, which is what
    // makes turning it back on restore the cadence rather than start from nothing.
    mutationFn: (update) => saveIssueDetectionSchedule(experimentId, update, schedule),
    onSuccess: (saved) => queryClient.setQueryData(queryKey, saved),
    // On failure, reconcile the cache with what actually persisted: the two tag writes are not
    // atomic, so a reopened modal should show the real stored state rather than the draft.
    onError: () => {
      queryClient.invalidateQueries(queryKey);
    },
  });

  const setSetting = useCallback((update: IssueDetectionScheduleUpdate) => mutation.mutateAsync(update), [mutation]);

  return {
    setting: schedule?.setting ?? noIssueDetectionSchedule().setting,
    traceFilters: schedule?.traceFilters ?? [],
    model: schedule?.model,
    isLoading,
    isUpdating: mutation.isLoading,
    error: error ?? (mutation.error instanceof Error ? mutation.error : null),
    loadError: error ?? null,
    retryLoad: () => {
      void refetch();
    },
    setSetting,
  };
};
