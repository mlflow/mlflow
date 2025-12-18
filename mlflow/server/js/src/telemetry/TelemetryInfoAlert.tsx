import {
  TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_KEY,
  TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_VERSION,
} from '../telemetry/utils';
import { useLocalStorage } from '../shared/web-shared/hooks';
import { Alert, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { Link } from '../common/utils/RoutingUtils';

export const TelemetryInfoAlert = () => {
  const { theme } = useDesignSystemTheme();

  const [isTelemetryAlertDismissed, setIsTelemetryAlertDismissed] = useLocalStorage({
    key: TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_KEY,
    version: TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_VERSION,
    initialValue: false,
  });

  if (isTelemetryAlertDismissed) {
    return null;
  }

  return (
    <Alert
      componentId="mlflow.home.telemetry-alert"
      message="Information about UI telemetry"
      type="info"
      onClose={() => setIsTelemetryAlertDismissed(true)}
      description={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <span>
            <FormattedMessage
              defaultMessage="MLflow collects usage data to improve the product. To confirm your preferences, please visit the settings page in the navigation sidebar. To learn more about what data is collected, please visit the <documentation>documentation</documentation>."
              description="Telemetry alert description"
              values={{
                documentation: (chunks: any) => (
                  <Link
                    to="https://mlflow.org/docs/latest/community/usage-tracking.html"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {chunks}
                  </Link>
                ),
              }}
            />
          </span>
        </div>
      }
    />
  );
};
