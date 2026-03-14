import { useCallback, useState } from 'react';
import { Notification, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { Link } from '../../../../../../common/utils/RoutingUtils';
import Routes from '../../../../../routes';

const NOTIFICATION_DURATION_MS = 10000;

export const useIssueDetectionNotification = (experimentId?: string) => {
  const { theme } = useDesignSystemTheme();
  const [isOpen, setIsOpen] = useState(false);
  const [runId, setRunId] = useState<string | undefined>(undefined);

  const runPageLink = experimentId && runId ? Routes.getIssueDetectionRunDetailsRoute(experimentId, runId) : undefined;

  const showIssueDetectionNotification = useCallback((newRunId?: string) => {
    setRunId(newRunId);
    setIsOpen(true);
  }, []);

  const notificationContextHolder = (
    <Notification.Provider>
      <Notification.Root
        componentId="mlflow.traces.issue-detection-notification"
        open={isOpen}
        onOpenChange={setIsOpen}
        duration={NOTIFICATION_DURATION_MS}
      >
        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm, paddingLeft: theme.spacing.sm }}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, flex: 1 }}>
            <Notification.Title>
              <FormattedMessage
                defaultMessage="Issue detection job triggered"
                description="Notification message when issue detection job is started"
              />
            </Notification.Title>
            <Notification.Description>
              {runPageLink && (
                <Link componentId="mlflow.traces.issue_detection_notification_link" to={runPageLink}>
                  <FormattedMessage
                    defaultMessage="View status"
                    description="Link to view issue detection job status"
                  />
                </Link>
              )}
            </Notification.Description>
          </div>
          <Notification.Close componentId="mlflow.traces.issue-detection-notification.close" />
        </div>
      </Notification.Root>
      <Notification.Viewport css={{ position: 'fixed', top: theme.spacing.lg, right: theme.spacing.lg }} />
    </Notification.Provider>
  );

  return {
    showIssueDetectionNotification,
    notificationContextHolder,
  };
};
