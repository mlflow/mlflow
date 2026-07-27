import { useCallback, useEffect, useRef, useState } from 'react';
import { NotFoundError } from '@databricks/web-shared/errors';
import { Typography, useLegacyNotification } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { useNavigate, useSearchParams } from '../../../../../common/utils/RoutingUtils';
import LocalStorageUtils from '../../../../../common/utils/LocalStorageUtils';
import Routes from '../../../../routes';
import { RunPageTabName } from '../../../../constants';
import { JobStatus, isJobComplete, useFetchJobStatus } from '../../../run-page/hooks/useFetchJobStatus';
import { getPreservedQueryString } from '../../../../pages/experiment-page-tabs/side-nav/utils';

export interface SubmittedIssueDetectionJob {
  experimentId: string;
  jobId: string;
  runId: string;
  traceCount: number;
  submittedAtMs: number;
}

type StoredSubmittedIssueDetectionJob = Omit<SubmittedIssueDetectionJob, 'experimentId'>;

export const ISSUE_DETECTION_SUBMITTED_JOBS_STORAGE_KEY = 'mlflow.issueDetection.submittedJobs';
const ISSUE_DETECTION_TRACKED_EXPERIMENTS_STORAGE_KEY = 'mlflow.issueDetection.trackedExperiments';
const ISSUE_DETECTION_JOB_STORAGE_COMPONENT = 'IssueDetectionJobWatcher';
const ISSUE_DETECTION_JOB_SUBMITTED_EVENT = 'mlflow.issueDetection.jobSubmitted';
const ISSUE_DETECTION_JOB_TTL_MS = 60 * 60 * 1000;

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
  const storedJobs = jobs.map<StoredSubmittedIssueDetectionJob>(({ jobId, runId, traceCount, submittedAtMs }) => ({
    jobId,
    runId,
    traceCount,
    submittedAtMs,
  }));
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

const getIssueDetectionRunRoute = (experimentId: string, runId: string, issueCount?: number) => {
  if (issueCount === undefined || issueCount <= 1) {
    return Routes.getIssueDetectionRunDetailsRoute(experimentId, runId);
  }

  return Routes.getIssueDetectionRunDetailsTabRoute(experimentId, runId, RunPageTabName.ISSUES);
};

const TrackedIssueDetectionJobNotification = ({
  job,
  onTerminalStatus,
  onStopTracking,
}: {
  job: SubmittedIssueDetectionJob;
  onTerminalStatus: (job: SubmittedIssueDetectionJob, status: JobStatus, result: unknown) => void;
  onStopTracking: (job: SubmittedIssueDetectionJob) => void;
}) => {
  const { status, result, error } = useFetchJobStatus({
    jobId: job.jobId,
    enabled: true,
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
    if (status && isJobComplete(status)) {
      onTerminalStatus(job, status, result);
    }
  }, [job, onTerminalStatus, result, status]);

  return null;
};

export const IssueDetectionJobNotifications = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [notification, notificationContextHolder] = useLegacyNotification();

  const [trackedJobs, setTrackedJobs] = useState<SubmittedIssueDetectionJob[]>(() => getSubmittedIssueDetectionJobs());
  const [startedNotificationJobs, setStartedNotificationJobs] = useState<SubmittedIssueDetectionJob[]>([]);
  const notifiedStartJobIdsRef = useRef<Set<string>>(new Set());
  const notifiedCompletionJobIdsRef = useRef<Set<string>>(new Set());
  const preservedQueryString = getPreservedQueryString(searchParams.toString());
  const withPreservedQueryString = useCallback(
    (route: string) => (preservedQueryString ? `${route}${preservedQueryString}` : route),
    [preservedQueryString],
  );
  const stopTrackingJob = useCallback((trackedJob: SubmittedIssueDetectionJob) => {
    removeSubmittedIssueDetectionJob(trackedJob);
    setTrackedJobs((currentJobs) => currentJobs.filter((job) => job.jobId !== trackedJob.jobId));
  }, []);

  useEffect(() => {
    const handleSubmittedJob = (event: Event) => {
      const job = (event as CustomEvent<unknown>).detail;
      if (!isSubmittedIssueDetectionJob(job)) {
        return;
      }

      setTrackedJobs((currentJobs) => appendSubmittedJob(currentJobs, job));
      setStartedNotificationJobs((currentJobs) => appendSubmittedJob(currentJobs, job));
    };

    window.addEventListener(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, handleSubmittedJob);
    return () => {
      window.removeEventListener(ISSUE_DETECTION_JOB_SUBMITTED_EVENT, handleSubmittedJob);
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
                    Routes.getIssueDetectionRunDetailsRoute(
                      startedNotificationJob.experimentId,
                      startedNotificationJob.runId,
                    ),
                  ),
                );
              }}
            >
              <FormattedMessage
                defaultMessage="View progress"
                description="Link to the issue detection run page from the job-started notification"
              />
            </Typography.Link>
          </div>
        ),
      });
    });
  }, [startedNotificationJobs, notification, navigate, withPreservedQueryString]);

  const handleTerminalStatus = useCallback(
    (trackedJob: SubmittedIssueDetectionJob, status: JobStatus, result: unknown) => {
      const { experimentId, jobId, runId } = trackedJob;
      if (!notifiedCompletionJobIdsRef.current.has(jobId)) {
        notifiedCompletionJobIdsRef.current.add(jobId);
        const completedKey = `issue-detection-completed-${jobId}`;
        if (status === JobStatus.SUCCEEDED) {
          const jobResult =
            typeof result === 'object' && result !== null
              ? (result as { issues?: number; total_traces_analyzed?: number })
              : undefined;
          const issueCount = jobResult?.issues ?? 0;
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
                {/* eslint-disable-next-line formatjs/no-multiple-plurals */}
                <FormattedMessage
                  defaultMessage="Found {issueCount, plural, =0 {no issues} one {1 issue} other {# issues}} across {traceCount, plural, one {1 trace} other {# traces}}."
                  description="Notification body shown when an issue detection job finishes successfully"
                  values={{ issueCount, traceCount: jobResult?.total_traces_analyzed ?? 0 }}
                />{' '}
                <Typography.Link
                  componentId="mlflow.traces.issue-detection.completed-toast.view-results"
                  onClick={() => {
                    notification.close(completedKey);
                    navigate(withPreservedQueryString(getIssueDetectionRunRoute(experimentId, runId, issueCount)));
                  }}
                >
                  {issueCount <= 1 ? (
                    <FormattedMessage
                      defaultMessage="View details"
                      description="Link to the issue detection run overview from the completion notification"
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
              <Typography.Link
                componentId="mlflow.traces.issue-detection.failed-toast.view-details"
                onClick={() => {
                  notification.close(completedKey);
                  navigate(withPreservedQueryString(Routes.getIssueDetectionRunDetailsRoute(experimentId, runId)));
                }}
              >
                <FormattedMessage
                  defaultMessage="View details"
                  description="Link to the issue detection run page from the failure notification"
                />
              </Typography.Link>
            ),
          });
        }
      }
      stopTrackingJob(trackedJob);
    },
    [notification, navigate, stopTrackingJob, withPreservedQueryString],
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
