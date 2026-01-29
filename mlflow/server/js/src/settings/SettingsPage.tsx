import { Button, Spinner, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useLocalStorage } from '../shared/web-shared/hooks';
import { TELEMETRY_ENABLED_STORAGE_KEY, TELEMETRY_ENABLED_STORAGE_VERSION } from '../telemetry/utils';
import { telemetryClient } from '../telemetry';
import { useCallback, useState } from 'react';
import { getAjaxUrl } from '../common/utils/FetchUtils';

const SettingsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isCleaningDemo, setIsCleaningDemo] = useState(false);
  const [cleanupMessage, setCleanupMessage] = useState<string | null>(null);

  const [isTelemetryEnabled, setIsTelemetryEnabled] = useLocalStorage({
    key: TELEMETRY_ENABLED_STORAGE_KEY,
    version: TELEMETRY_ENABLED_STORAGE_VERSION,
    initialValue: true,
  });

  const handleTelemetryToggle = useCallback(
    (checked: boolean) => {
      setIsTelemetryEnabled(checked);
      if (checked) {
        telemetryClient.start();
      } else {
        telemetryClient.shutdown();
      }
    },
    [setIsTelemetryEnabled],
  );

  const handleClearAllDemoData = useCallback(async () => {
    setIsCleaningDemo(true);
    setCleanupMessage(null);
    try {
      const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/demo/delete'), {
        method: 'POST',
      });
      const data = await response.json();
      const featuresDeleted = data.features_deleted ?? [];
      if (featuresDeleted.length === 0) {
        setCleanupMessage(
          intl.formatMessage({
            defaultMessage: 'No demo data found to clear',
            description: 'Demo cleanup message when no data exists',
          }),
        );
      } else {
        setCleanupMessage(
          intl.formatMessage(
            {
              defaultMessage: 'Cleared {count} {count, plural, one {demo} other {demos}}',
              description: 'Demo cleanup success message showing number of demo types cleaned',
            },
            { count: featuresDeleted.length },
          ),
        );
      }
    } catch (error) {
      setCleanupMessage(
        intl.formatMessage({
          defaultMessage: 'Failed to clear demo data',
          description: 'Demo cleanup error message',
        }),
      );
    } finally {
      setIsCleaningDemo(false);
    }
  }, [intl]);

  return (
    <div css={{ padding: theme.spacing.md }}>
      <Typography.Title level={2}>
        <FormattedMessage defaultMessage="Settings" description="Settings page title" />
      </Typography.Title>

      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', maxWidth: 600 }}>
        <div css={{ display: 'flex', flexDirection: 'column', marginRight: theme.spacing.lg }}>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Enable telemetry" description="Enable telemetry settings title" />
          </Typography.Title>
          <Typography.Text>
            <FormattedMessage
              defaultMessage="This setting enables UI telemetry data collection. Learn more about what types of data are collected in our {documentation}."
              description="Enable telemetry settings description"
              values={{
                documentation: (
                  <Typography.Link
                    componentId="mlflow.settings.telemetry.documentation-link"
                    href="https://mlflow.org/docs/latest/community/usage-tracking.html"
                    openInNewTab
                  >
                    <FormattedMessage defaultMessage="documentation" description="Documentation link text" />
                  </Typography.Link>
                ),
              }}
            />
          </Typography.Text>
        </div>
        <Switch
          componentId="mlflow.settings.telemetry.toggle-switch"
          checked={isTelemetryEnabled}
          onChange={handleTelemetryToggle}
          label=" "
          activeLabel={intl.formatMessage({ defaultMessage: 'On', description: 'Telemetry enabled label' })}
          inactiveLabel={intl.formatMessage({ defaultMessage: 'Off', description: 'Telemetry disabled label' })}
        />
      </div>

      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          maxWidth: 600,
          marginTop: theme.spacing.lg,
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', marginRight: theme.spacing.lg }}>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Demo data" description="Demo data settings title" />
          </Typography.Title>
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Clear all demo data generated from the home page. This removes demo experiments, traces, evaluations, and prompts."
              description="Demo data settings description"
            />
          </Typography.Text>
          {cleanupMessage && (
            <Typography.Text css={{ marginTop: theme.spacing.xs }} color="secondary">
              {cleanupMessage}
            </Typography.Text>
          )}
        </div>
        <Button
          componentId="mlflow.settings.demo.clear-all-button"
          onClick={handleClearAllDemoData}
          disabled={isCleaningDemo}
        >
          {isCleaningDemo ? (
            <Spinner size="small" />
          ) : (
            <FormattedMessage defaultMessage="Clear all demo data" description="Clear demo data button" />
          )}
        </Button>
      </div>
    </div>
  );
};

export default SettingsPage;
