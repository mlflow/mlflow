import { useCallback, useEffect, useRef, useState } from 'react';
import { NotFoundError } from '@databricks/web-shared/errors';
import { Typography, useLegacyNotification } from '@databricks/design-system';
import { defineMessages, FormattedMessage, type MessageDescriptor, useIntl } from '@databricks/i18n';
import { useNavigate, useSearchParams } from '../../../../../common/utils/RoutingUtils';
import LocalStorageUtils from '../../../../../common/utils/LocalStorageUtils';
import Routes from '../../../../routes';
import { ExperimentPageTabName } from '../../../../constants';
import { JobStatus, isJobComplete, useFetchJobStatus } from '../../../run-page/hooks/useFetchJobStatus';
import { getPreservedQueryString } from '../../../../pages/experiment-page-tabs/side-nav/utils';
import { useActiveWorkspace } from '../../../../../workspaces/utils/WorkspaceUtils';

export interface SubmittedIssueDetectionJob {
  experimentId: string;
  workspace?: string | null;
  jobId: string;
  runId: string;
  traceCount: number;
  submittedAtMs: number;
  startTimeMs?: number;
}

type TrackedIssueDetectionJob = SubmittedIssueDetectionJob & { observedOnly?: boolean };

type StoredSubmittedIssueDetectionJob = Omit<SubmittedIssueDetectionJob, 'experimentId'>;

export const ISSUE_DETECTION_SUBMITTED_JOBS_STORAGE_KEY = 'mlflow.issueDetection.submittedJobs';
const ISSUE_DETECTION_TRACKED_EXPERIMENTS_STORAGE_KEY = 'mlflow.issueDetection.trackedExperiments';
const ISSUE_DETECTION_JOB_STORAGE_COMPONENT = 'IssueDetectionJobWatcher';
export const ISSUE_DETECTION_JOB_SUBMITTED_EVENT = 'mlflow.issueDetection.jobSubmitted';
export const ISSUE_DETECTION_JOB_OBSERVED_EVENT = 'mlflow.issueDetection.jobObserved';
export const ISSUE_DETECTION_JOB_COMPLETED_EVENT = 'mlflow.issueDetection.jobCompleted';
const ISSUE_DETECTION_JOB_TTL_MS = 60 * 60 * 1000;
const MAX_ERROR_MESSAGE_LENGTH = 500;

/* eslint-disable formatjs/no-multiple-plurals */
const COMPLETION_MESSAGES = defineMessages({
  total: {
    defaultMessage:
      'Found {issueCount, plural, =0 {no issues} one {1 issue} other {# issues}} across {traceCount, plural, one {1 trace} other {# traces}}.',
    description: 'Notification body shown when an issue detection job finishes successfully',
  },
  allNew: {
    defaultMessage:
      'Found {newCount, plural, one {1 new issue} other {# new issues}} across {traceCount, plural, one {1 trace} other {# traces}}.',
    description: 'Notification body shown when every issue an issue detection job found is newly reported',
  },
  newAndRefreshed: {
    defaultMessage:
      'Found {newCount, plural, =0 {no new issues} one {1 new issue} other {# new issues}} and refreshed {refreshedCount, plural, one {1 existing issue} other {# existing issues}} across {traceCount, plural, one {1 trace} other {# traces}}.',
    description:
      'Notification body shown when an issue detection job both filed new issues and re-found issues the experiment already carried',
  },
});
/* eslint-enable formatjs/no-multiple-plurals */

/**
 * The completion wording, and the counts it needs. An older server sends no `new_issue_count`,
 * so the total-only wording is kept for it rather than calling issues new with nothing behind it.
 */
const completionSummary = (
  issueCount: number,
  newCount: number | undefined,
  traceCount: number,
): { descriptor: MessageDescriptor; values: Record<string, number> } => {
  // Not just `undefined`: the count is read off a JSON payload through a type assertion, so
  // null, a numeric string and NaN reach here too, and none of them describe a count.
  if (typeof newCount !== 'number' || !Number.isFinite(newCount) || issueCount === 0) {
    return { descriptor: COMPLETION_MESSAGES.total, values: { issueCount, traceCount } };
  }
  // Held inside the total it describes: outside that range the server disagreed with itself,
  // and "5 new" beside a total of 3 is worse than the nearest coherent answer.
  const filedCount = Math.min(Math.max(newCount, 0), issueCount);
  const refreshedCount = issueCount - filedCount;
  return refreshedCount === 0
    ? { descriptor: COMPLETION_MESSAGES.allNew, values: { newCount: filedCount, traceCount } }
    : {
        descriptor: COMPLETION_MESSAGES.newAndRefreshed,
        values: { newCount: filedCount, refreshedCount, traceCount },
      };
};

const getSubmittedJobStore = (experimentId: string) =>
  LocalStorageUtils.getSessionScopedStoreForComponent(ISSUE_DETECTION_JOB_STORAGE_COMPONENT, experimentId);

const getTrackedExperimentsStore = () =>
  LocalStorageUtils.getSessionScopedStoreForComponent(ISSUE_DETECTION_JOB_STORAGE_COMPONENT, 'experiments');

const isStoredSubmittedIssueDetectionJob = (value: unknown): value is StoredSubmittedIssueDetectionJob => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<StoredSubmittedIssueDetectionJob>;
  return (
    typeof candidate.jobId === 'string' &&
    typeof candidate.runId === 'string' &&
    typeof candidate.traceCount === 'number' &&
    typeof candidate.submittedAtMs === 'number'
  );
};

const isSubmittedIssueDetectionJob = (value: unknown): value is SubmittedIssueDetectionJob => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<SubmittedIssueDetectionJob>;
  return typeof candidate.experimentId === 'string' && isStoredSubmittedIssueDetectionJob(candidate);
};

const appendSubmittedJob = <T extends { jobId: string }>(jobs: T[], job: T) => [
  ...jobs.filter((trackedJob) => trackedJob.jobId !== job.jobId),
  job,
];

const isSubmittedIssueDetectionJobExpired = (job: SubmittedIssueDetectionJob, nowMs = Date.now()) =>
  nowMs - job.submittedAtMs >= ISSUE_DETECTION_JOB_TTL_MS;

const getTrackedExperimentIds = () => {
  try {
    const serializedExperimentIds = getTrackedExperimentsStore().getItem(
      ISSUE_DETECTION_TRACKED_EXPERIMENTS_STORAGE_KEY,
    );
    if (!serializedExperimentIds) {
      return [];
    }

    const parsedExperimentIds: unknown = JSON.parse(serializedExperimentIds);
    return Array.isArray(parsedExperimentIds)
      ? parsedExperimentIds.filter((experimentId): experimentId is string => typeof experimentId === 'string')
      : [];
  } catch {
    return [];
  }
};

const setTrackedExperimentIds = (experimentIds: string[]) => {
  getTrackedExperimentsStore().setItem(
    ISSUE_DETECTION_TRACKED_EXPERIMENTS_STORAGE_KEY,
    JSON.stringify(Array.from(new Set(experimentIds))),
  );
};

const setSubmittedIssueDetectionJobsForExperiment = (experimentId: string, jobs: SubmittedIssueDetectionJob[]) => {
  const storedJobs = jobs.map<StoredSubmittedIssueDetectionJob>(
    ({ jobId, runId, traceCount, submittedAtMs, workspace }) => ({
      jobId,
      runId,
      traceCount,
      submittedAtMs,
      workspace,
    }),
  );
  getSubmittedJobStore(experimentId).setItem(ISSUE_DETECTION_SUBMITTED_JOBS_STORAGE_KEY, JSON.stringify(storedJobs));

  const trackedExperimentIds = getTrackedExperimentIds();
  setTrackedExperimentIds(
    storedJobs.length
      ? appendSubmittedJob(
          trackedExperimentIds.map((trackedExperimentId) => ({ jobId: trackedExperimentId })),
          { jobId: experimentId },
        ).map(({ jobId }) => jobId)
      : trackedExperimentIds.filter((trackedExperimentId) => trackedExperimentId !== experimentId),
  );
};

const getSubmittedIssueDetectionJobsForExperiment = (experimentId: string): SubmittedIssueDetectionJob[] => {
  try {
    const serializedJobs = getSubmittedJobStore(experimentId).getItem(ISSUE_DETECTION_SUBMITTED_JOBS_STORAGE_KEY);
    if (!serializedJobs) {
      return [];
    }

    const parsedJobs: unknown = JSON.parse(serializedJobs);
    if (!Array.isArray(parsedJobs)) {
      setSubmittedIssueDetectionJobsForExperiment(experimentId, []);
      return [];
    }

    const jobs = parsedJobs.filter(isStoredSubmittedIssueDetectionJob).map((job) => ({
      experimentId,
      ...job,
    }));
    const activeJobs = jobs.filter((job) => !isSubmittedIssueDetectionJobExpired(job));
    if (activeJobs.length !== parsedJobs.length) {
      setSubmittedIssueDetectionJobsForExperiment(experimentId, activeJobs);
    }
    return activeJobs;
  } catch {
    return [];
  }
};

export const getSubmittedIssueDetectionJobs = () =>
  getTrackedExperimentIds().flatMap(getSubmittedIssueDetectionJobsForExperiment);

export const getSubmittedIssueDetectionJob = () => getSubmittedIssueDetectionJobs()[0] ?? null;

const removeSubmittedIssueDetectionJob = ({
  experimentId,
  jobId,
}: Pick<SubmittedIssueDetectionJob, 'experimentId' | 'jobId'>) => {
  setSubmittedIssueDetectionJobsForExperiment(
    experimentId,
    getSubmittedIssueDetectionJobsForExperiment(experimentId).filter((job) => job.jobId !== jobId),
  );
};

export const clearSubmittedIssueDetectionJob = (jobId?: string) => {
  if (!jobId) {
    getTrackedExperimentIds().forEach((experimentId) => {
      setSubmittedIssueDetectionJobsForExperiment(experimentId, []);
    });
    setTrackedExperimentIds([]);
    return;
  }

  getSubmittedIssueDetectionJobs().forEach((job) => {
    if (job.jobId === jobId) {
      removeSubmittedIssueDetectionJob(job);
    }
  });
};

export const recordSubmittedIssueDetectionJob = (
  job: Omit<SubmittedIssueDetectionJob, 'submittedAtMs'> & Partial<Pick<SubmittedIssueDetectionJob, 'submittedAtMs'>>,
) => {
  const submittedJob = { ...job, submittedAtMs: job.submittedAtMs ?? Date.now() };
  setSubmittedIssueDetectionJobsForExperiment(
    submittedJob.experimentId,
    appendSubmittedJob(getSubmittedIssueDetectionJobsForExperiment(submittedJob.experimentId), submittedJob),
  );
  window.dispatchEvent(new CustomEvent(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, { detail: submittedJob }));
};

/**
 * Where "View issues" goes once a run has succeeded: the tab that lists what it found.
 *
 * The other two toasts link to the run history page instead. A run in flight and a run that broke
 * both have nothing for the Issues tab to show, while the history page has the row for this run --
 * its status while it works, and the reason underneath it if it failed.
 */
const getIssuesTabRoute = (experimentId: string) =>
  Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Issues);

const TrackedIssueDetectionJobNotification = ({
  job,
  onTerminalStatus,
  onStopTracking,
}: {
  job: TrackedIssueDetectionJob;
  onTerminalStatus: (
    job: SubmittedIssueDetectionJob,
    status: JobStatus,
    result: unknown,
    errorMessage?: string,
  ) => void;
  onStopTracking: (job: SubmittedIssueDetectionJob) => void;
}) => {
  const workspace = useActiveWorkspace();
  const sawActiveStatus = useRef(false);
  const { status, result, error, errorMessage } = useFetchJobStatus({
    jobId: job.jobId,
    enabled: job.workspace === undefined || job.workspace === workspace,
  });

  useEffect(() => {
    if (error instanceof NotFoundError) {
      onStopTracking(job);
    }
  }, [error, job, onStopTracking]);

  useEffect(() => {
    if (isSubmittedIssueDetectionJobExpired(job)) {
      onStopTracking(job);
      return;
    }

    const timeoutMs = Math.max(job.submittedAtMs + ISSUE_DETECTION_JOB_TTL_MS - Date.now(), 0);
    const timeoutId = window.setTimeout(() => onStopTracking(job), timeoutMs);
    return () => window.clearTimeout(timeoutId);
  }, [job, onStopTracking]);

  useEffect(() => {
    if (status === JobStatus.PENDING || status === JobStatus.RUNNING) {
      sawActiveStatus.current = true;
    } else if (status && isJobComplete(status)) {
      // A stale RUNNING run can point to an already-ended job; it must not re-announce on reload.
      if (job.observedOnly && !sawActiveStatus.current) onStopTracking(job);
      else onTerminalStatus(job, status, result, errorMessage);
    }
  }, [job, onTerminalStatus, onStopTracking, result, status, errorMessage]);

  return null;
};

export const IssueDetectionJobNotifications = () => {
  const intl = useIntl();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [notification, notificationContextHolder] = useLegacyNotification();

  const [trackedJobs, setTrackedJobs] = useState<TrackedIssueDetectionJob[]>(() => getSubmittedIssueDetectionJobs());
  const [startedNotificationJobs, setStartedNotificationJobs] = useState<SubmittedIssueDetectionJob[]>([]);
  const notifiedStartJobIdsRef = useRef<Set<string>>(new Set());
  const notifiedCompletionJobIdsRef = useRef<Set<string>>(new Set());
  const preservedQueryString = getPreservedQueryString(searchParams.toString());
  const withPreservedQueryString = useCallback(
    (route: string, workspace?: string | null) => {
      const query = new URLSearchParams(preservedQueryString);
      if (workspace === null) query.delete('workspace');
      else if (workspace !== undefined) query.set('workspace', workspace);
      const queryString = query.toString();
      return queryString ? `${route}?${queryString}` : route;
    },
    [preservedQueryString],
  );
  const stopTrackingJob = useCallback((trackedJob: SubmittedIssueDetectionJob) => {
    removeSubmittedIssueDetectionJob(trackedJob);
    setTrackedJobs((currentJobs) => currentJobs.filter((job) => job.jobId !== trackedJob.jobId));
  }, []);

  useEffect(() => {
    const handleJobEvent = (event: Event) => {
      const job = (event as CustomEvent<unknown>).detail;
      if (!isSubmittedIssueDetectionJob(job)) {
        return;
      }

      if (!notifiedCompletionJobIdsRef.current.has(job.jobId)) {
        setTrackedJobs((currentJobs) => {
          const existing = currentJobs.find((tracked) => tracked.jobId === job.jobId);
          // An observation adds the actual start time without resetting submission metadata or expiry.
          const trackedJob = existing
            ? {
                ...(event.type === ISSUE_DETECTION_JOB_OBSERVED_EVENT ? existing : job),
                startTimeMs: existing.startTimeMs ?? job.startTimeMs,
              }
            : event.type === ISSUE_DETECTION_JOB_OBSERVED_EVENT
              ? { ...job, observedOnly: true }
              : job;
          return appendSubmittedJob(currentJobs, trackedJob);
        });
      }
      if (event.type === ISSUE_DETECTION_JOB_SUBMITTED_EVENT) {
        setStartedNotificationJobs((currentJobs) => appendSubmittedJob(currentJobs, job));
      }
    };

    window.addEventListener(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, handleJobEvent);
    window.addEventListener(ISSUE_DETECTION_JOB_OBSERVED_EVENT, handleJobEvent);
    return () => {
      window.removeEventListener(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, handleJobEvent);
      window.removeEventListener(ISSUE_DETECTION_JOB_OBSERVED_EVENT, handleJobEvent);
    };
  }, []);

  useEffect(() => {
    startedNotificationJobs.forEach((startedNotificationJob) => {
      if (notifiedStartJobIdsRef.current.has(startedNotificationJob.jobId)) {
        return;
      }

      notifiedStartJobIdsRef.current.add(startedNotificationJob.jobId);
      const startedKey = `issue-detection-started-${startedNotificationJob.jobId}`;
      notification.info({
        key: startedKey,
        placement: 'topRight',
        duration: 8,
        message: (
          <FormattedMessage
            defaultMessage="Issue detection started"
            description="Notification title shown when an issue detection job is submitted"
          />
        ),
        description: (
          <div>
            <FormattedMessage
              defaultMessage="Analyzing {count, plural, one {1 trace} other {# traces}}. We'll notify you here when it completes."
              description="Notification body shown when an issue detection job is submitted"
              values={{ count: startedNotificationJob.traceCount }}
            />{' '}
            <Typography.Link
              componentId="mlflow.traces.issue-detection.started-toast.view-progress"
              onClick={() => {
                notification.close(startedKey);
                navigate(
                  withPreservedQueryString(
                    Routes.getIssueDetectionRunsRoute(startedNotificationJob.experimentId),
                    startedNotificationJob.workspace,
                  ),
                );
              }}
            >
              <FormattedMessage
                defaultMessage="View progress"
                description="Link to the issue detection run history from the job-started notification"
              />
            </Typography.Link>
          </div>
        ),
      });
    });
  }, [startedNotificationJobs, notification, navigate, withPreservedQueryString]);

  const handleTerminalStatus = useCallback(
    (trackedJob: SubmittedIssueDetectionJob, status: JobStatus, result: unknown, errorMessage?: string) => {
      const { experimentId, jobId, runId } = trackedJob;
      const reason = errorMessage?.replace(/\s+/g, ' ').trim();
      const shortReason =
        reason && reason.length > MAX_ERROR_MESSAGE_LENGTH
          ? `${reason.slice(0, MAX_ERROR_MESSAGE_LENGTH - 1)}…`
          : reason;
      const startTimeMs = trackedJob.startTimeMs;
      const started = typeof startTimeMs === 'number' && Number.isFinite(startTimeMs) && startTimeMs > 0 && (
        <div>
          <FormattedMessage
            defaultMessage="Started {startTime}."
            description="Start time shown in an issue detection job completion notification"
            values={{
              startTime: intl.formatDate(startTimeMs, {
                month: 'short',
                day: 'numeric',
                hour: 'numeric',
                minute: '2-digit',
              }),
            }}
          />
        </div>
      );
      if (!notifiedCompletionJobIdsRef.current.has(jobId)) {
        notifiedCompletionJobIdsRef.current.add(jobId);
        window.dispatchEvent(new CustomEvent(ISSUE_DETECTION_JOB_COMPLETED_EVENT, { detail: trackedJob }));
        const completedKey = `issue-detection-completed-${jobId}`;
        if (status === JobStatus.SUCCEEDED) {
          const jobResult =
            typeof result === 'object' && result !== null
              ? (result as { issues?: number; new_issue_count?: number; total_traces_analyzed?: number })
              : undefined;
          const issueCount = jobResult?.issues ?? 0;
          const summary = completionSummary(
            issueCount,
            jobResult?.new_issue_count,
            jobResult?.total_traces_analyzed ?? 0,
          );
          notification.success({
            key: completedKey,
            placement: 'topRight',
            duration: 12,
            message: (
              <FormattedMessage
                defaultMessage="Issue detection completed"
                description="Notification title shown when an issue detection job finishes successfully"
              />
            ),
            description: (
              <div>
                {started}
                <FormattedMessage {...summary.descriptor} values={summary.values} />{' '}
                <Typography.Link
                  componentId="mlflow.traces.issue-detection.completed-toast.view-results"
                  onClick={() => {
                    notification.close(completedKey);
                    navigate(withPreservedQueryString(getIssuesTabRoute(experimentId), trackedJob.workspace));
                  }}
                >
                  {issueCount <= 1 ? (
                    <FormattedMessage
                      defaultMessage="View details"
                      description="Link to the Issues tab from the completion notification"
                    />
                  ) : (
                    <FormattedMessage
                      defaultMessage="View issues"
                      description="Link to the detected issues from the completion notification"
                    />
                  )}
                </Typography.Link>
              </div>
            ),
          });
        } else if (status === JobStatus.FAILED || status === JobStatus.TIMEOUT) {
          notification.error({
            key: completedKey,
            placement: 'topRight',
            duration: 0,
            message: (
              <FormattedMessage
                defaultMessage="Issue detection failed"
                description="Notification title shown when an issue detection job fails"
              />
            ),
            description: (
              <div>
                {started}
                <div>
                  {shortReason ||
                    (status === JobStatus.TIMEOUT ? (
                      <FormattedMessage
                        defaultMessage="The run timed out."
                        description="Fallback reason in a timed out issue detection notification"
                      />
                    ) : (
                      <FormattedMessage
                        defaultMessage="The run ended with an error."
                        description="Fallback reason in a failed issue detection notification"
                      />
                    ))}
                </div>
                <Typography.Link
                  componentId="mlflow.traces.issue-detection.failed-toast.view-details"
                  onClick={() => {
                    notification.close(completedKey);
                    navigate(
                      withPreservedQueryString(Routes.getIssueDetectionRunsRoute(experimentId), trackedJob.workspace),
                    );
                  }}
                >
                  <FormattedMessage
                    defaultMessage="View details"
                    description="Link to the issue detection run history from the failure notification"
                  />
                </Typography.Link>
              </div>
            ),
          });
        }
      }
      stopTrackingJob(trackedJob);
    },
    [notification, navigate, stopTrackingJob, withPreservedQueryString, intl],
  );

  return (
    <>
      {notificationContextHolder}
      {trackedJobs.map((job) => (
        <TrackedIssueDetectionJobNotification
          key={job.jobId}
          job={job}
          onTerminalStatus={handleTerminalStatus}
          onStopTracking={stopTrackingJob}
        />
      ))}
    </>
  );
};
