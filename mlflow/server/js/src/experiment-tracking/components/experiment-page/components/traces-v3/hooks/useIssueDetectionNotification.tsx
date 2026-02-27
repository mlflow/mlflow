import { useCallback, useState, useEffect, useRef } from 'react';
import { keyframes } from '@emotion/react';
import { Notification, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { Link } from '../../../../../../common/utils/RoutingUtils';
import Routes from '../../../../../routes';
import { ExperimentPageTabName } from '../../../../../constants';

const NOTIFICATION_DURATION_MS = 10000;

const shrinkProgressAnimation = keyframes`
  from {
    width: 100%;
  }
  to {
    width: 0%;
  }
`;

export const useIssueDetectionNotification = (experimentId?: string) => {
  const { theme } = useDesignSystemTheme();
  const [isOpen, setIsOpen] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const evaluationRunsLink = experimentId
    ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns)
    : undefined;

  const showIssueDetectionNotification = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    setIsOpen(true);
    timerRef.current = setTimeout(() => {
      setIsOpen(false);
    }, NOTIFICATION_DURATION_MS);
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const notificationContextHolder = (
    <Notification.Provider>
      <Notification.Root
        componentId="mlflow.traces.issue-detection-notification"
        open={isOpen}
        onOpenChange={setIsOpen}
      >
        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, flex: 1 }}>
            <Notification.Title>
              <FormattedMessage
                defaultMessage="Issue detection job triggered"
                description="Notification message when issue detection job is started"
              />
            </Notification.Title>
            <Notification.Description>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                {evaluationRunsLink && (
                  <Link to={evaluationRunsLink}>
                    <FormattedMessage
                      defaultMessage="View status"
                      description="Link to view issue detection job status"
                    />
                  </Link>
                )}
                <div
                  css={{
                    height: 4,
                    backgroundColor: theme.colors.backgroundSecondary,
                    borderRadius: 2,
                    overflow: 'hidden',
                    marginTop: theme.spacing.xs,
                  }}
                >
                  {isOpen && (
                    <div
                      css={{
                        height: '100%',
                        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                        borderRadius: 2,
                        animation: `${shrinkProgressAnimation} ${NOTIFICATION_DURATION_MS}ms linear forwards`,
                      }}
                    />
                  )}
                </div>
              </div>
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
