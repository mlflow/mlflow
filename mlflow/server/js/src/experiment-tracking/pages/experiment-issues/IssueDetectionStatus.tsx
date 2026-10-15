import { Alert, Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState } from 'react';
import { Link } from '../../../common/utils/RoutingUtils';
import { JobStatus, isJobComplete } from '../../components/run-page/hooks/useFetchJobStatus';
import { useCancelJob } from '../../components/run-page/hooks/useCancelJob';
import { ConfirmModal } from '../../components/modals/ConfirmModal';
import Routes from '../../routes';
import type { useLatestIssueDetection } from './useIssueDetection';

export const IssueDetectionStatus = ({
  detection,
  experimentId,
  canEdit,
}: {
  detection: ReturnType<typeof useLatestIssueDetection>;
  experimentId: string;
  canEdit: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const [confirmCancel, setConfirmCancel] = useState(false);
  const cancel = useCancelJob();
  const { run, job, jobId, error, status, result } = detection;
  if (error) {
    return (
      <Alert
        componentId="mlflow.experiment-issues.history-error"
        type="warning"
        closable={false}
        message={
          <FormattedMessage
            defaultMessage="Unable to load detection history"
            description="Issues detection history read failure"
          />
        }
        description={error.message}
      />
    );
  }
  if (!run) return null;
  const failed = status === JobStatus.FAILED || status === JobStatus.TIMEOUT;
  const canCancel = canEdit && Boolean(jobId && job.status) && !job.error && !isJobComplete(status);
  return (
    <>
      <Alert
        componentId="mlflow.experiment-issues.detection-status"
        type={failed ? 'error' : 'info'}
        closable={false}
        message={
          status === JobStatus.SUCCEEDED ? (
            <FormattedMessage
              defaultMessage="Issue detection completed"
              description="Issues completed detection status"
            />
          ) : failed ? (
            <FormattedMessage defaultMessage="Issue detection failed" description="Issues failed detection status" />
          ) : status === JobStatus.CANCELED ? (
            <FormattedMessage
              defaultMessage="Issue detection canceled"
              description="Issues canceled detection status"
            />
          ) : (
            <FormattedMessage defaultMessage="Detecting issues" description="Issues running detection status" />
          )
        }
        description={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, flexWrap: 'wrap' }}>
            {status === JobStatus.SUCCEEDED &&
              result?.issues !== undefined &&
              result.total_traces_analyzed !== undefined && (
                <FormattedMessage
                  defaultMessage="Found {count, plural, one {# issue} other {# issues}} across {traces}."
                  description="Issues latest detection sample outcome"
                  values={{
                    count: result.issues,
                    traces: (
                      <FormattedMessage
                        defaultMessage="{count, plural, one {# selected trace} other {# selected traces}}"
                        description="Issues detection selected trace count"
                        values={{ count: result.total_traces_analyzed }}
                      />
                    ),
                  }}
                />
              )}
            {!isJobComplete(status) && job.status_details?.stage}
            {failed && typeof job.result === 'string' && job.result}
            <Link
              componentId="mlflow.experiment-issues.detection-run"
              to={Routes.getIssueDetectionRunDetailsRoute(experimentId, run.info.run_id)}
            >
              <FormattedMessage defaultMessage="View detection run" description="Issues latest detection run link" />
            </Link>
            {canCancel && (
              <Button componentId="mlflow.experiment-issues.cancel" size="small" onClick={() => setConfirmCancel(true)}>
                <FormattedMessage defaultMessage="Cancel detection" description="Issues cancel detection action" />
              </Button>
            )}
          </div>
        }
      />
      {(job.error || cancel.error) && (
        <Alert
          componentId="mlflow.experiment-issues.job-error"
          type="warning"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Unable to update detection progress"
              description="Issues job status or cancellation error"
            />
          }
          description={
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, flexWrap: 'wrap' }}>
              {(cancel.error ?? job.error)?.message}
              {job.error && (
                <Button
                  componentId="mlflow.experiment-issues.retry-job-status"
                  size="small"
                  loading={job.isFetching}
                  onClick={() => job.refetch()}
                >
                  <FormattedMessage
                    defaultMessage="Retry status"
                    description="Issues detection job status retry action"
                  />
                </Button>
              )}
            </div>
          }
        />
      )}
      <ConfirmModal
        isOpen={confirmCancel && canCancel}
        onClose={() => setConfirmCancel(false)}
        handleSubmit={() =>
          cancel.cancelJobAsync({ jobId: jobId ?? '', runUuid: run.info.run_id }).catch(() => undefined)
        }
        title={
          <FormattedMessage
            defaultMessage="Cancel issue detection?"
            description="Issues cancel detection confirmation title"
          />
        }
        helpText={
          <FormattedMessage
            defaultMessage="Issues already found will remain available."
            description="Issues cancel detection confirmation explanation"
          />
        }
        confirmButtonText={
          <FormattedMessage
            defaultMessage="Cancel detection"
            description="Issues cancel detection confirmation button"
          />
        }
      />
    </>
  );
};
