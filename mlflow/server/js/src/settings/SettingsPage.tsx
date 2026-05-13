import { Button, Card, Modal, Spinner, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { TELEMETRY_ENABLED_STORAGE_KEY, TELEMETRY_ENABLED_STORAGE_VERSION } from '../telemetry/utils';
import { telemetryClient } from '../telemetry';
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { fetchEndpointRaw, HTTPMethods } from '../common/utils/FetchUtils';
import { useLocation, useNavigate, useParams } from '../common/utils/RoutingUtils';
import { useDarkThemeContext } from '../common/contexts/DarkThemeContext';
import { ApiKeysPageInner } from '../gateway/pages/ApiKeysPage';
import Routes from '../experiment-tracking/routes';
import WebhooksSettings from './WebhooksSettings';
import {
  isSettingsPathSegment,
  SETTINGS_RETURN_TO_PARAM,
  SETTINGS_SECTION_GENERAL,
  SETTINGS_SECTION_LLM_CONNECTIONS,
  type SettingsPathSegment,
} from './settingsSectionConstants';

type SettingsSectionHeaderProps = {
  title: ReactNode;
  subtitle: ReactNode;
};

const SettingsSectionHeader = ({ title, subtitle }: SettingsSectionHeaderProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, width: '100%' }}>
      <Typography.Title level={3} withoutMargins>
        {title}
      </Typography.Title>
      <Typography.Text color="secondary">{subtitle}</Typography.Text>
    </div>
  );
};

type SettingsRowProps = {
  children: ReactNode;
  trailing: ReactNode;
  isFirst?: boolean;
};

const SettingsRow = ({ children, trailing, isFirst }: SettingsRowProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'flex-start',
        gap: theme.spacing.md,
        padding: theme.spacing.md,
        borderTop: isFirst ? undefined : `1px solid ${theme.colors.borderDecorative}`,
      }}
    >
      <div
        css={{
          flex: 1,
          minWidth: 0,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          textAlign: 'left',
        }}
      >
        {children}
      </div>
      <div css={{ flexShrink: 0, paddingTop: theme.spacing.xs }}>{trailing}</div>
    </div>
  );
};

const SettingsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const location = useLocation();
  const { section: sectionParam } = useParams();
  const [isCleaningDemo, setIsCleaningDemo] = useState(false);
  const [isConfirmModalOpen, setIsConfirmModalOpen] = useState(false);
  const { setIsDarkTheme } = useDarkThemeContext();
  const isDarkTheme = theme.isDarkMode;

  const activeSection: SettingsPathSegment = useMemo(() => {
    if (sectionParam && isSettingsPathSegment(sectionParam)) {
      return sectionParam;
    }
    return 'general';
  }, [sectionParam]);

  useEffect(() => {
    if (sectionParam && !isSettingsPathSegment(sectionParam)) {
      const returnTo = new URLSearchParams(location.search).get(SETTINGS_RETURN_TO_PARAM);
      const target = Routes.getSettingsSectionRoute(SETTINGS_SECTION_GENERAL);
      navigate(returnTo ? `${target}?${SETTINGS_RETURN_TO_PARAM}=${encodeURIComponent(returnTo)}` : target, {
        replace: true,
        state: location.state,
      });
    }
  }, [sectionParam, navigate, location.search, location.state]);

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
    } finally {
      setIsCleaningDemo(false);
    }
  }, []);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        minHeight: '100%',
        padding: theme.spacing.md,
        gap: theme.spacing.lg,
        textAlign: 'left',
      }}
    >
      <div
        css={{
          width: '100%',
          maxWidth: theme.responsive.breakpoints.lg,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.lg,
          textAlign: 'left',
        }}
      >
        {activeSection === 'general' && (
          <>
            <SettingsSectionHeader
              title={
                <FormattedMessage defaultMessage="General" description="Settings content section title: general" />
              }
              subtitle={
                <FormattedMessage
                  defaultMessage="Appearance, product feedback, telemetry, and workspace demo data."
                  description="Settings content section subtitle for general preferences including demo data"
                />
              }
            />
            <Card componentId="mlflow.settings.general.preferences-card" css={{ padding: 0, overflow: 'hidden' }}>
              <SettingsRow
                isFirst
                trailing={
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
                }
              >
                <Typography.Title level={4} withoutMargins>
                  <FormattedMessage defaultMessage="Theme preference" description="Theme settings title" />
                </Typography.Title>
                <Typography.Text>
                  <FormattedMessage
                    defaultMessage="Select your theme preference between light and dark."
                    description="Description for the theme setting in the settings page"
                  />
                </Typography.Text>
              </SettingsRow>
              <SettingsRow
                trailing={
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
                    inactiveLabel={intl.formatMessage({
                      defaultMessage: 'Off',
                      description: 'Telemetry disabled label',
                    })}
                  />
                }
              >
                <Typography.Title level={4} withoutMargins>
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
              </SettingsRow>
              <SettingsRow
                trailing={
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
                }
              >
                <Typography.Title level={4} withoutMargins>
                  <FormattedMessage defaultMessage="Demo data" description="Demo data settings title" />
                </Typography.Title>
                <Typography.Text>
                  <FormattedMessage
                    defaultMessage="Clear all demo data generated from the home page. This removes demo experiments, traces, evaluations, and prompts."
                    description="Demo data settings description"
                  />
                </Typography.Text>
              </SettingsRow>
            </Card>
          </>
        )}

        {activeSection === SETTINGS_SECTION_LLM_CONNECTIONS && (
          <>
            <SettingsSectionHeader
              title={
                <FormattedMessage
                  defaultMessage="LLM Connections"
                  description="Settings content section title: LLM connections"
                />
              }
              subtitle={
                <FormattedMessage
                  defaultMessage="Create and manage API keys for authenticating to external LLM providers."
                  description="Settings content section subtitle for LLM connections (API keys)"
                />
              }
            />
            <ApiKeysPageInner />
          </>
        )}

        {activeSection === 'webhooks' && (
          <>
            <SettingsSectionHeader
              title={
                <FormattedMessage defaultMessage="Webhooks" description="Settings content section title: webhooks" />
              }
              subtitle={
                <FormattedMessage
                  defaultMessage="Receive HTTP notifications when events occur in MLflow."
                  description="Settings content section subtitle for webhooks"
                />
              }
            />
            <WebhooksSettings showTitle={false} showDescription={false} />
          </>
        )}
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
