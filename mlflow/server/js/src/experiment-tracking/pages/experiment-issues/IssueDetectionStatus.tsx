import { Alert, Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { JobStatus, isJobComplete } from '../../components/run-page/hooks/useFetchJobStatus';
import Routes from '../../routes';
import type { useLatestIssueDetection } from './useIssueDetection';

export const IssueDetectionStatus = ({
  detection,
  experimentId,
}: {
  detection: ReturnType<typeof useLatestIssueDetection>;
  experimentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { run, job, error, status } = detection;
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
  if (!run || status === JobStatus.SUCCEEDED) return null;
  const failed = status === JobStatus.FAILED || status === JobStatus.TIMEOUT;
  return (
    <>
      {failed && (
        <Alert
          componentId="mlflow.experiment-issues.detection-status"
          type="error"
          closable={false}
          message={
            <FormattedMessage defaultMessage="Issue detection failed" description="Issues failed detection status" />
          }
          description={
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, flexWrap: 'wrap' }}>
              {typeof job.result === 'string' && job.result}
              <Link
                componentId="mlflow.experiment-issues.detection-run"
                to={Routes.getIssueDetectionRunDetailsRoute(experimentId, run.info.run_id)}
              >
                <FormattedMessage defaultMessage="View detection run" description="Issues latest detection run link" />
              </Link>
            </div>
          }
        />
      )}
      {job.error && !isJobComplete(status) && (
        <Alert
          componentId="mlflow.experiment-issues.job-error"
          type="warning"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Unable to update detection progress"
              description="Issues job status error"
            />
          }
          description={
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, flexWrap: 'wrap' }}>
              {job.error.message}
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
            </div>
          }
        />
      )}
    </>
  );
};
