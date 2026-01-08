import { Button, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useLocalStorage } from '../shared/web-shared/hooks';
import { TELEMETRY_ENABLED_STORAGE_KEY, TELEMETRY_ENABLED_STORAGE_VERSION } from '../telemetry/utils';
import { telemetryClient } from '../telemetry';
import { useCallback } from 'react';

const EXPERIMENT_TYPE_POSITION_STORAGE_KEY = 'mlflow.sidebar.experimentTypeSelectorPosition';
const EXPERIMENT_TYPE_POSITION_STORAGE_VERSION = 1;
const WORKFLOW_SELECTOR_GUIDANCE_STORAGE_KEY = 'mlflow.sidebar.workflowSelectorGuidanceShown';
const WORKFLOW_SELECTOR_GUIDANCE_STORAGE_VERSION = 1;

const SettingsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [isTelemetryEnabled, setIsTelemetryEnabled] = useLocalStorage({
    key: TELEMETRY_ENABLED_STORAGE_KEY,
    version: TELEMETRY_ENABLED_STORAGE_VERSION,
    initialValue: true,
  });

  const [isExperimentTypeSelectorAtTop, setIsExperimentTypeSelectorAtTop] = useLocalStorage({
    key: EXPERIMENT_TYPE_POSITION_STORAGE_KEY,
    version: EXPERIMENT_TYPE_POSITION_STORAGE_VERSION,
    initialValue: false,
  });

  const [hasSeenGuidance, setHasSeenGuidance] = useLocalStorage({
    key: WORKFLOW_SELECTOR_GUIDANCE_STORAGE_KEY,
    version: WORKFLOW_SELECTOR_GUIDANCE_STORAGE_VERSION,
    initialValue: false,
  });

  const handleResetGuidance = useCallback(() => {
    setHasSeenGuidance(false);
  }, [setHasSeenGuidance]);

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
            <FormattedMessage
              defaultMessage="Workflow type selector position"
              description="Workflow type selector position settings title"
            />
          </Typography.Title>
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Controls whether the workflow type selector (GenAI / Machine Learning) appears at the top or bottom of the sidebar."
              description="Workflow type selector position settings description"
            />
          </Typography.Text>
        </div>
        <Switch
          componentId="mlflow.settings.experiment-type-position.toggle-switch"
          checked={isExperimentTypeSelectorAtTop}
          onChange={setIsExperimentTypeSelectorAtTop}
          label=" "
          activeLabel={intl.formatMessage({ defaultMessage: 'Top', description: 'Workflow selector at top' })}
          inactiveLabel={intl.formatMessage({
            defaultMessage: 'Bottom',
            description: 'Workflow selector at bottom',
          })}
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
            <FormattedMessage
              defaultMessage="Reset workflow selector guidance"
              description="Reset workflow selector guidance settings title"
            />
          </Typography.Title>
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Show the workflow type selector guidance popover again on your next page navigation."
              description="Reset workflow selector guidance settings description"
            />
          </Typography.Text>
        </div>
        <Button
          componentId="mlflow.settings.workflow-guidance.reset-button"
          onClick={handleResetGuidance}
          disabled={!hasSeenGuidance}
        >
          <FormattedMessage defaultMessage="Reset" description="Reset guidance button" />
        </Button>
      </div>
    </div>
  );
};

export default SettingsPage;
