import { Button, Modal, Spinner, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useLocalStorage } from '../shared/web-shared/hooks';
import { TELEMETRY_ENABLED_STORAGE_KEY, TELEMETRY_ENABLED_STORAGE_VERSION } from '../telemetry/utils';
import { telemetryClient } from '../telemetry';
import { useCallback, useState } from 'react';
import { fetchEndpointRaw, HTTPMethods } from '../common/utils/FetchUtils';
import { useDarkThemeContext } from '../common/contexts/DarkThemeContext';

const SettingsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isCleaningDemo, setIsCleaningDemo] = useState(false);
  const [isConfirmModalOpen, setIsConfirmModalOpen] = useState(false);
  const { setIsDarkTheme } = useDarkThemeContext();
  const isDarkTheme = theme.isDarkMode;

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

  const handleThemeToggle = useCallback(
    (checked: boolean) => {
      setIsDarkTheme(checked);
    },
    [setIsDarkTheme],
  );

  const handleClearAllDemoData = useCallback(async () => {
    setIsCleaningDemo(true);
    try {
      await fetchEndpointRaw({
        relativeUrl: 'ajax-api/3.0/mlflow/demo/delete',
        method: HTTPMethods.POST,
      });
    } catch (error) {
      console.error('Failed to clear demo data:', error);
    } finally {
      setIsCleaningDemo(false);
    }
  }, []);

  return (
    <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      <Typography.Title level={2} withoutMargins>
        <FormattedMessage defaultMessage="Settings" description="Settings page title" />
      </Typography.Title>

      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', maxWidth: 600 }}>
        <div css={{ display: 'flex', flexDirection: 'column', marginRight: theme.spacing.lg }}>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Theme preference" description="Theme settings title" />
          </Typography.Title>
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Select your theme preference between light and dark."
              description="Description for the theme setting in the settings page"
            />
          </Typography.Text>
        </div>
        <Switch
          componentId="mlflow.settings.theme.toggle-switch"
          checked={isDarkTheme}
          onChange={handleThemeToggle}
          label={
            isDarkTheme
              ? intl.formatMessage({ defaultMessage: 'Dark', description: 'Dark theme label' })
              : intl.formatMessage({ defaultMessage: 'Light', description: 'Light theme label' })
          }
          activeLabel={intl.formatMessage({ defaultMessage: 'Dark', description: 'Dark theme label' })}
          inactiveLabel={intl.formatMessage({ defaultMessage: 'Light', description: 'Light theme label' })}
        />
      </div>

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
          label={
            isTelemetryEnabled
              ? intl.formatMessage({ defaultMessage: 'On', description: 'Telemetry enabled label' })
              : intl.formatMessage({ defaultMessage: 'Off', description: 'Telemetry disabled label' })
          }
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
        </div>
        <Button
          componentId="mlflow.settings.demo.clear-all-button"
          onClick={() => setIsConfirmModalOpen(true)}
          disabled={isCleaningDemo}
        >
          {isCleaningDemo ? (
            <Spinner size="small" />
          ) : (
            <FormattedMessage defaultMessage="Clear all demo data" description="Clear demo data button" />
          )}
        </Button>
      </div>

      <Modal
        componentId="mlflow.settings.demo.confirm-modal"
        title={intl.formatMessage({
          defaultMessage: 'Clear demo data',
          description: 'Demo data deletion confirmation modal title',
        })}
        visible={isConfirmModalOpen}
        onCancel={() => setIsConfirmModalOpen(false)}
        onOk={async () => {
          setIsConfirmModalOpen(false);
          await handleClearAllDemoData();
        }}
        okText={intl.formatMessage({
          defaultMessage: 'Clear',
          description: 'Demo data deletion confirm button',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Demo data deletion cancel button',
        })}
        okButtonProps={{ danger: true }}
      >
        <Typography.Text>
          <FormattedMessage
            defaultMessage="This will delete the demo experiment and all associated traces, evaluations, and prompts. You can regenerate demo data from the home page, but any manual changes you made to the demo data will be lost."
            description="Demo data deletion confirmation message"
          />
        </Typography.Text>
      </Modal>
    </div>
  );
};

export default SettingsPage;
