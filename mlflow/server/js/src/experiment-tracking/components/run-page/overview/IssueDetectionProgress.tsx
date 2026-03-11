import type { ReactNode } from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, CheckCircleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useNavigate, useParams } from '../../../../common/utils/RoutingUtils';
import { RunPageTabName } from '../../../constants';
import Routes from '../../../routes';

export interface IssueDetectionProgressProps {
  /** Callback when cancel button is clicked */
  onCancel?: () => void;
  /** Whether the cancel operation is in progress */
  isCancelling?: boolean;
}

const ProgressBar = ({ percent }: { percent: number }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        height: 6,
        borderRadius: 3,
        backgroundColor: theme.colors.grey200,
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          height: '100%',
          width: `${percent}%`,
          backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
          transition: 'width 0.3s ease',
        }}
      />
    </div>
  );
};

export const IssueDetectionProgress = ({ onCancel, isCancelling }: IssueDetectionProgressProps) => {
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

  // TODO: Replace with actual data from backend
  const totalTraces = 25;
  const scannedTraces = 20;
  const identifiedIssues = 8;
  const isComplete = false;
  const progress = totalTraces > 0 ? (scannedTraces / totalTraces) * 100 : 0;

  // TODO: Don't render if detection is complete (once real data is available)
  // if (isComplete) {
  //   return null;
  // }

  return (
    <div css={{ marginBottom: theme.spacing.lg }}>
      <div
        css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.sm }}
      >
        <Typography.Title level={4} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="Detection progress" description="Issue detection progress > Title" />
        </Typography.Title>
        {onCancel && (
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
          {isComplete ? (
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />
          ) : (
            <div
              css={{
                width: 16,
                height: 16,
                borderRadius: '50%',
                border: `2px solid ${theme.colors.border}`,
              }}
            />
          )}
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Identifying issues from traces..."
              description="Issue detection progress > Step label"
            />
          </Typography.Text>
        </div>
        <div css={{ marginLeft: 24 }}>
          {!isComplete && (
            <div css={{ marginBottom: theme.spacing.xs }}>
              <ProgressBar percent={progress} />
            </div>
          )}
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="{scannedTraces} of <tracesLink>{totalTraces} traces</tracesLink> scanned, <issuesLink>{identifiedIssues} issues</issuesLink> identified so far"
              description="Issue detection progress > Progress summary"
              values={{
                scannedTraces,
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
          </Typography.Hint>
        </div>
      </div>
    </div>
  );
};
