import type { ReactNode } from 'react';
import { FormattedMessage } from 'react-intl';
import {
  Button,
  CheckCircleIcon,
  ParagraphSkeleton,
  XCircleIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { GenAIMarkdownRenderer } from '../../../../shared/web-shared/genai-markdown-renderer/GenAIMarkdownRenderer';
import { useNavigate, useParams } from '../../../../common/utils/RoutingUtils';
import { RunPageTabName } from '../../../constants';
import Routes from '../../../routes';
import { useSearchIssuesQuery } from '../hooks/useSearchIssuesQuery';
import { useFetchIssueJobStatus, IssueJobStatus, isJobComplete } from '../hooks/useFetchIssueJobStatus';

export interface IssueDetectionProgressProps {
  /** Callback when cancel button is clicked */
  onCancel?: () => void;
  /** Whether the cancel operation is in progress */
  isCancelling?: boolean;
  /** Job ID for fetching issue detection job status */
  jobId?: string;
}

export const IssueDetectionProgress = ({ onCancel, isCancelling, jobId }: IssueDetectionProgressProps) => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const { experimentId, runUuid } = useParams<{ experimentId: string; runUuid: string }>();

  const handleViewTraces = () => {
    if (experimentId && runUuid) {
      navigate(Routes.getIssueDetectionRunDetailsTabRoute(experimentId, runUuid, RunPageTabName.TRACES));
    }
  };

  const handleViewIssues = () => {
    if (experimentId && runUuid) {
      navigate(Routes.getIssueDetectionRunDetailsTabRoute(experimentId, runUuid, RunPageTabName.ISSUES));
    }
  };

  const {
    status: jobStatus,
    totalTraces,
    result,
    isLoading,
    error,
  } = useFetchIssueJobStatus({
    jobId,
    enabled: !!jobId,
  });

  const isJobSucceeded = jobStatus === IssueJobStatus.SUCCEEDED;
  const isJobFailed = jobStatus === IssueJobStatus.FAILED || jobStatus === IssueJobStatus.TIMEOUT || !!error;
  const isJobCanceled = jobStatus === IssueJobStatus.CANCELED;
  const jobComplete = isJobComplete(jobStatus) || !!error;

  const { issues } = useSearchIssuesQuery({
    experimentId: experimentId ?? '',
    sourceRunId: runUuid ?? '',
    enabled: !!experimentId && !!runUuid,
    pollingEnabled: !jobComplete,
  });

  const identifiedIssues = issues.length;

  if (!jobId) {
    return null;
  }

  if (isLoading) {
    return (
      <div css={{ marginBottom: theme.spacing.lg }}>
        <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="Detection progress" description="Issue detection progress > Title" />
        </Typography.Title>
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: theme.spacing.md,
          }}
        >
          <ParagraphSkeleton />
        </div>
      </div>
    );
  }

  return (
    <div css={{ marginBottom: theme.spacing.lg }}>
      <div
        css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.sm }}
      >
        <Typography.Title level={4} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="Detection progress" description="Issue detection progress > Title" />
        </Typography.Title>
        {onCancel && !jobComplete && (
          <Button componentId="mlflow.traces.issue-detection.cancel-button" onClick={onCancel} loading={isCancelling}>
            <FormattedMessage defaultMessage="Cancel" description="Issue detection progress > Cancel button" />
          </Button>
        )}
      </div>

      <div
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          padding: theme.spacing.md,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.xs }}>
          {isJobSucceeded ? (
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />
          ) : isJobFailed || isJobCanceled ? (
            <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />
          ) : (
            <div
              css={{
                width: 16,
                height: 16,
                borderRadius: '50%',
                border: `2px solid ${theme.colors.border}`,
                borderTopColor: theme.colors.actionPrimaryBackgroundDefault,
                animation: 'spin 1s linear infinite',
                '@keyframes spin': {
                  '0%': { transform: 'rotate(0deg)' },
                  '100%': { transform: 'rotate(360deg)' },
                },
              }}
            />
          )}
          <Typography.Text>
            {isJobSucceeded ? (
              <FormattedMessage
                defaultMessage="Issue detection completed"
                description="Issue detection progress > Step label when completed"
              />
            ) : isJobFailed ? (
              <FormattedMessage
                defaultMessage="Issue detection failed"
                description="Issue detection progress > Step label when failed"
              />
            ) : isJobCanceled ? (
              <FormattedMessage
                defaultMessage="Issue detection canceled"
                description="Issue detection progress > Step label when canceled"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Identifying issues from traces..."
                description="Issue detection progress > Step label"
              />
            )}
          </Typography.Text>
        </div>
        <div css={{ marginLeft: 24 }}>
          <Typography.Hint>
            {jobComplete ? (
              <FormattedMessage
                defaultMessage="Scanned <tracesLink>{totalTraces} traces</tracesLink>, <issuesLink>{identifiedIssues} issues</issuesLink> found"
                description="Issue detection progress > Progress summary when completed"
                values={{
                  totalTraces,
                  identifiedIssues,
                  tracesLink: (chunks: ReactNode) => (
                    <Typography.Link
                      componentId="mlflow.traces.issue-detection.view-traces-link"
                      onClick={handleViewTraces}
                    >
                      {chunks}
                    </Typography.Link>
                  ),
                  issuesLink: (chunks: ReactNode) => (
                    <Typography.Link
                      componentId="mlflow.traces.issue-detection.view-issues-link"
                      onClick={handleViewIssues}
                    >
                      {chunks}
                    </Typography.Link>
                  ),
                }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Scanning <tracesLink>{totalTraces} traces</tracesLink>, <issuesLink>{identifiedIssues} issues</issuesLink> identified so far"
                description="Issue detection progress > Progress summary while running"
                values={{
                  totalTraces,
                  identifiedIssues,
                  tracesLink: (chunks: ReactNode) => (
                    <Typography.Link
                      componentId="mlflow.traces.issue-detection.view-traces-link"
                      onClick={handleViewTraces}
                    >
                      {chunks}
                    </Typography.Link>
                  ),
                  issuesLink: (chunks: ReactNode) => (
                    <Typography.Link
                      componentId="mlflow.traces.issue-detection.view-issues-link"
                      onClick={handleViewIssues}
                    >
                      {chunks}
                    </Typography.Link>
                  ),
                }}
              />
            )}
          </Typography.Hint>
        </div>
      </div>

      {isJobSucceeded && result?.summary && (
        <>
          <Typography.Title level={4} css={{ marginTop: theme.spacing.lg, marginBottom: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Issue detection summary" description="Issue detection summary > Title" />
          </Typography.Title>
          <div
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              padding: theme.spacing.md,
            }}
          >
            <GenAIMarkdownRenderer compact>{result.summary}</GenAIMarkdownRenderer>
          </div>
        </>
      )}
    </div>
  );
};
