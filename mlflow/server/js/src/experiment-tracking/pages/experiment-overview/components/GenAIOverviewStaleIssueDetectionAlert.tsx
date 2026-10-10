import { useState } from 'react';
import { Alert, SparkleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { FormattedMessage, useIntl } from 'react-intl';
import type { GenAIOverviewIssueDetectionPhase } from '../hooks/useGenAIOverviewIssueDetectionProgress';
import type { GenAIOverviewStaleIssueDetectionState } from '../hooks/useGenAIOverviewStaleIssueDetectionState';

export interface GenAIOverviewStaleIssueDetectionAlertProps {
  experimentId: string;
  state: GenAIOverviewStaleIssueDetectionState;
  issueDetectionPhase: GenAIOverviewIssueDetectionPhase;
  canStartDetection: boolean;
  onDetectNewIssues: () => void;
}

export const GenAIOverviewStaleIssueDetectionAlert = ({
  experimentId,
  state,
  issueDetectionPhase,
  canStartDetection,
  onDetectNewIssues,
}: GenAIOverviewStaleIssueDetectionAlertProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [dismissedRunUuid, setDismissedRunUuid] = useLocalStorage({
    key: `mlflow.experiment.${experimentId}.genaiOverview.staleIssueDetectionDismissedRunUuid`,
    version: 0,
    initialValue: '',
  });
  const [detectionStartedForRunUuid, setDetectionStartedForRunUuid] = useState<string>();
  const isDetectionInProgress = issueDetectionPhase === 'starting' || issueDetectionPhase === 'running';

  if (state.status !== 'ready' || dismissedRunUuid === state.latestRunUuid) return null;
  const wasDetectionStartedFromThisAlert = isDetectionInProgress && detectionStartedForRunUuid === state.latestRunUuid;
  if (isDetectionInProgress && !wasDetectionStartedFromThisAlert) return null;

  const actionLabel =
    issueDetectionPhase === 'starting' ? (
      <FormattedMessage
        defaultMessage="Starting..."
        description="Analyze overview stale issue detection action while a run is starting"
      />
    ) : issueDetectionPhase === 'running' ? (
      <FormattedMessage
        defaultMessage="Detecting issues..."
        description="Analyze overview stale issue detection action while a run is running"
      />
    ) : (
      <FormattedMessage
        defaultMessage="Detect new issues"
        description="Analyze overview action to rerun stale issue detection"
      />
    );

  return (
    <div
      css={{
        padding: `0 ${theme.spacing.lg}px ${theme.spacing.md}px`,
        '@media (max-width: 800px)': {
          padding: `0 ${theme.spacing.md}px ${theme.spacing.md}px`,
        },
      }}
    >
      <Alert
        componentId="mlflow.genai-overview.stale-issue-detection"
        type="info"
        size="small"
        message={
          <Typography.Text>
            <FormattedMessage
              defaultMessage="It's been <bold>{days} days</bold> and <bold>{traceCount, plural, one {# new trace} other {# new traces}}</bold> since your last issue detection."
              description="Analyze overview reminder that new traces have arrived since issue detection last ran"
              values={{
                days: state.elapsedDays,
                traceCount: state.newTraceCount,
                bold: (chunks) => <Typography.Text bold>{chunks}</Typography.Text>,
              }}
            />
          </Typography.Text>
        }
        closable
        closeIconLabel={intl.formatMessage({
          defaultMessage: 'Dismiss stale issue detection reminder',
          description: 'Accessible label for dismissing stale issue detection guidance on the overview',
        })}
        onClose={() => setDismissedRunUuid(state.latestRunUuid)}
        actions={[
          {
            componentId: 'mlflow.genai-overview.stale-issue-detection.detect',
            icon: <SparkleIcon color="ai" />,
            disabled: isDetectionInProgress || !canStartDetection,
            onClick: () => {
              setDetectionStartedForRunUuid(state.latestRunUuid);
              onDetectNewIssues();
            },
            children: actionLabel,
          },
        ]}
      />
    </div>
  );
};
