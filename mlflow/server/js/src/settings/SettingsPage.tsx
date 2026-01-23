import { Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useLocalStorage } from '../shared/web-shared/hooks';
import { TELEMETRY_ENABLED_STORAGE_KEY, TELEMETRY_ENABLED_STORAGE_VERSION } from '../telemetry/utils';
import { telemetryClient } from '../telemetry';
import { useCallback } from 'react';
import { useAssistant } from '../assistant/AssistantContext';

const SettingsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { isAssistantEnabled, setAssistantEnabled, isLocalServer, closePanel } = useAssistant();

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

  const handleAssistantToggle = useCallback(
    (checked: boolean) => {
      setAssistantEnabled(checked);
      if (!checked) {
        closePanel();
      }
    },
    [setAssistantEnabled, closePanel],
  );

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
          disabledLabel=" "
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
            <FormattedMessage defaultMessage="Enable Assistant" description="Enable assistant settings title" />
          </Typography.Title>
          <Typography.Text>
            {isLocalServer ? (
              <FormattedMessage
                defaultMessage="This setting enables the AI Assistant feature in the sidebar. Learn more about what the Assistant can do in our {documentation}."
                description="Enable assistant settings description"
                values={{
                  documentation: (
                    <Typography.Link
                      componentId="mlflow.settings.assistant.documentation-link"
                      href="https://mlflow.org/docs/latest/assistant/index.html"
                      openInNewTab
                    >
                      <FormattedMessage defaultMessage="documentation" description="Documentation link text" />
                    </Typography.Link>
                  ),
                }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="The Assistant feature is currently only available when running MLflow locally. Support for remote servers is coming soon."
                description="Enable assistant settings description when not on local server"
              />
            )}
          </Typography.Text>
        </div>
        <Switch
          componentId="mlflow.settings.assistant.toggle-switch"
          checked={isLocalServer && isAssistantEnabled}
          onChange={handleAssistantToggle}
          disabled={!isLocalServer}
          label=" "
          activeLabel={intl.formatMessage({ defaultMessage: 'On', description: 'Assistant enabled label' })}
          inactiveLabel={intl.formatMessage({ defaultMessage: 'Off', description: 'Assistant disabled label' })}
          disabledLabel=" "
        />
      </div>
    </div>
  );
};

export default SettingsPage;
