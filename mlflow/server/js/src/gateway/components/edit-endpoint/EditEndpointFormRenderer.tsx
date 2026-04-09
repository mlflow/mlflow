import { Link, useSearchParams } from '../../../common/utils/RoutingUtils';
import {
  Alert,
  Breadcrumb,
  Button,
  InfoFillIcon,
  Spinner,
  Switch,
  Tabs,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { UseFormReturn } from 'react-hook-form';
import { Controller } from 'react-hook-form';
import { useMemo } from 'react';
import GatewayRoutes from '../../routes';
import { GatewayLabel } from '../../../common/components/GatewayNewTag';
import { LongFormSummary } from '../../../common/components/long-form/LongFormSummary';
import type { EditEndpointFormData } from '../../hooks/useEditEndpointForm';
import { TrafficSplitConfigurator } from './TrafficSplitConfigurator';
import { FallbackModelsConfigurator } from './FallbackModelsConfigurator';
import { StarterCodeCard } from './StarterCodeCard';
import { EditableEndpointName } from './EditableEndpointName';
import { GatewayUsageSection } from './GatewayUsageSection';
import type { Endpoint, EndpointModelMapping } from '../../types';
import { TracesV3Logs } from '../../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3Logs';
import { MonitoringConfigProvider } from '../../../experiment-tracking/hooks/useMonitoringConfig';
import { useMonitoringFiltersTimeRange } from '../../../experiment-tracking/hooks/useMonitoringFilters';
import { TracesV3DateSelector } from '../../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3DateSelector';

/**
 * Returns the provider string to pass to StarterCodeCard.
 * If all models share the same passthrough API (treating openai and azure as equivalent),
 * returns the first model's provider so the passthrough tab is shown.
 * Otherwise returns undefined so only the MLflow Chat Completions tab is shown.
 */
export const getStarterCodeProvider = (modelMappings: EndpointModelMapping[]): string | undefined => {
  const normalized = new Set(
    modelMappings.map((m) => {
      const p = m.model_definition?.provider;
      return p === 'azure' ? 'openai' : p;
    }),
  );
  return normalized.size <= 1 ? modelMappings[0]?.model_definition?.provider : undefined;
};

const LogsTabContent = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const timeRange = useMonitoringFiltersTimeRange();

  return (
    <>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: theme.spacing.sm,
        }}
      >
        <TracesV3DateSelector excludeOptions={['ALL']} />
        <Link
          componentId="mlflow.gateway.edit_endpoint.traces_link"
          to={`/experiments/${experimentId}/traces`}
          css={{
            color: theme.colors.actionPrimaryBackgroundDefault,
            textDecoration: 'none',
            '&:hover': { textDecoration: 'underline' },
          }}
        >
          <FormattedMessage
            defaultMessage="Open full trace viewer"
            description="Link to open the full trace viewer for the endpoint's experiment"
          />
        </Link>
      </div>
      <TracesV3Logs experimentIds={[experimentId]} disableActions timeRange={timeRange} />
    </>
  );
};

export interface EditEndpointFormRendererProps {
  form: UseFormReturn<EditEndpointFormData>;
  isLoadingEndpoint: boolean;
  isSubmitting: boolean;
  loadError: Error | null;
  mutationError: Error | null;
  errorMessage: string | null;
  endpoint: Endpoint | undefined;
  existingEndpoints: Endpoint[] | undefined;
  isFormComplete: boolean;
  hasChanges: boolean;
  onSubmit: (values: EditEndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameUpdate: (newName: string) => Promise<void>;
}

export const EditEndpointFormRenderer = ({
  form,
  isLoadingEndpoint,
  isSubmitting,
  loadError,
  mutationError,
  errorMessage,
  endpoint,
  existingEndpoints,
  isFormComplete,
  hasChanges,
  onSubmit,
  onCancel,
  onNameUpdate,
}: EditEndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [searchParams, setSearchParams] = useSearchParams();
  const VALID_TABS = ['overview', 'usage', 'traces'] as const;
  const tabParam = searchParams.get('tab');
  // Support legacy ?tab=configuration URLs
  const normalizedTab = tabParam === 'configuration' ? 'overview' : tabParam;
  const activeTab = VALID_TABS.includes(normalizedTab as (typeof VALID_TABS)[number])
    ? (normalizedTab as string)
    : 'overview';

  const trafficSplitModels = form.watch('trafficSplitModels');
  const fallbackModels = form.watch('fallbackModels');
  const experimentId = form.watch('experimentId');

  // Don't disable tabs that were requested via URL query param
  const isUsageTabDisabled = !experimentId && activeTab !== 'usage';
  const isTracesTabDisabled = !experimentId && activeTab !== 'traces';

  const tooltipLinkUrlBuilder = useMemo(() => {
    if (!endpoint) return undefined;
    return (_experimentId: string, timestampMs: number, timeIntervalSeconds: number) =>
      GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id, {
        tab: 'traces',
        startTime: new Date(timestampMs).toISOString(),
        endTime: new Date(timestampMs + timeIntervalSeconds * 1000).toISOString(),
      });
  }, [endpoint]);

  const totalWeight = trafficSplitModels.reduce((sum, m) => sum + m.weight, 0);
  const isValidTotal = Math.abs(totalWeight - 100) < 0.01;

  const uniqueSecretNames = useMemo(
    () => [
      ...new Set(endpoint?.model_mappings.map((m) => m.model_definition?.secret_name).filter(Boolean) as string[]),
    ],
    [endpoint?.model_mappings],
  );

  if (isLoadingEndpoint) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading endpoint..." description="Loading message for endpoint" />
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.edit-endpoint.error"
            type="error"
            message={loadError.message ?? 'Endpoint not found'}
          />
        </div>
      </div>
    );
  }

  return (
    <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
      <div css={{ padding: theme.spacing.md }}>
        <Breadcrumb includeTrailingCaret>
          <Breadcrumb.Item>
            <Link
              componentId="mlflow.gateway.edit_endpoint.breadcrumb_gateway_link"
              to={GatewayRoutes.gatewayPageRoute}
            >
              <GatewayLabel />
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>
            <Link
              componentId="mlflow.gateway.edit_endpoint.breadcrumb_endpoints_link"
              to={GatewayRoutes.gatewayPageRoute}
            >
              <FormattedMessage defaultMessage="Endpoints" description="Breadcrumb link to endpoints list" />
            </Link>
          </Breadcrumb.Item>
        </Breadcrumb>
        <div
          css={{ marginTop: theme.spacing.sm, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
        >
          <EditableEndpointName
            endpoint={endpoint}
            existingEndpoints={existingEndpoints}
            onNameUpdate={onNameUpdate}
            isSubmitting={isSubmitting}
          />
        </div>
      </div>

      {mutationError && (
        <div css={{ padding: `0 ${theme.spacing.md}px` }}>
          <Alert
            componentId="mlflow.gateway.edit-endpoint.mutation-error"
            closable={false}
            message={errorMessage}
            type="error"
            css={{ marginBottom: theme.spacing.md }}
          />
        </div>
      )}

      <Tabs.Root
        componentId="mlflow.gateway.endpoint.tabs"
        valueHasNoPii
        value={activeTab}
        onValueChange={(value) => {
          setSearchParams(
            (params) => {
              params.set('tab', value);
              return params;
            },
            { replace: true },
          );
        }}
        css={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
      >
        <div css={{ paddingLeft: theme.spacing.md, paddingRight: theme.spacing.md }}>
          <Tabs.List>
            <Tabs.Trigger value="overview">
              <FormattedMessage defaultMessage="Overview" description="Tab label for endpoint overview" />
            </Tabs.Trigger>
            {isUsageTabDisabled ? (
              <Tooltip
                componentId="mlflow.gateway.endpoint.usage-tab-tooltip"
                content={intl.formatMessage({
                  defaultMessage: 'Enable Usage Tracking in the Overview tab to view usage metrics',
                  description:
                    'Tooltip shown on disabled Usage tab explaining that usage tracking must be enabled first',
                })}
              >
                <Tabs.Trigger value="usage" disabled>
                  <FormattedMessage defaultMessage="Usage" description="Tab label for endpoint usage metrics" />
                </Tabs.Trigger>
              </Tooltip>
            ) : (
              <Tabs.Trigger value="usage">
                <FormattedMessage defaultMessage="Usage" description="Tab label for endpoint usage metrics" />
              </Tabs.Trigger>
            )}
            {isTracesTabDisabled ? (
              <Tooltip
                componentId="mlflow.gateway.endpoint.traces-tab-tooltip"
                content={intl.formatMessage({
                  defaultMessage: 'Enable Usage Tracking in the Overview tab to view logs',
                  description:
                    'Tooltip shown on disabled Logs tab explaining that usage tracking must be enabled first',
                })}
              >
                <Tabs.Trigger value="traces" disabled>
                  <FormattedMessage defaultMessage="Logs" description="Tab label for endpoint logs" />
                </Tabs.Trigger>
              </Tooltip>
            ) : (
              <Tabs.Trigger value="traces">
                <FormattedMessage defaultMessage="Logs" description="Tab label for endpoint logs" />
              </Tabs.Trigger>
            )}
          </Tabs.List>
        </div>

        <div
          css={{
            flex: 1,
            display: 'flex',
            gap: theme.spacing.md,
            padding: `${theme.spacing.md}px`,
            overflow: 'auto',
            backgroundColor: theme.colors.backgroundPrimary,
          }}
        >
          <div css={{ flex: 1 }}>
            <Tabs.Content value="overview">
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
                {/* Unified Model card */}
                <div
                  css={{
                    padding: theme.spacing.md,
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: theme.borders.borderRadiusMd,
                    backgroundColor: theme.colors.backgroundSecondary,
                  }}
                >
                  <Typography.Title level={3} css={{ margin: 0 }}>
                    <FormattedMessage
                      defaultMessage="Model"
                      description="Gateway > Endpoint details > Section title for model configuration card"
                    />
                  </Typography.Title>

                  {/* Primary sub-section */}
                  <div
                    css={{
                      marginTop: theme.spacing.md,
                      padding: theme.spacing.md,
                      border: `1px solid ${theme.colors.border}`,
                      borderRadius: theme.borders.borderRadiusMd,
                      backgroundColor: theme.colors.backgroundPrimary,
                    }}
                  >
                    <Typography.Title level={4} css={{ margin: 0 }}>
                      <FormattedMessage
                        defaultMessage="Primary Model"
                        description="Gateway > Endpoint details > Sub-section title for primary traffic split models"
                      />
                    </Typography.Title>

                    <div css={{ marginTop: theme.spacing.md }}>
                      <Controller
                        control={form.control}
                        name="trafficSplitModels"
                        render={({ field }) => (
                          <TrafficSplitConfigurator
                            value={field.value}
                            onChange={field.onChange}
                            componentId="mlflow.gateway.edit-endpoint.traffic-split"
                          />
                        )}
                      />
                    </div>
                  </div>

                  <Controller
                    control={form.control}
                    name="fallbackModels"
                    render={({ field }) => (
                      <FallbackModelsConfigurator
                        value={field.value}
                        onChange={field.onChange}
                        componentId="mlflow.gateway.edit-endpoint.fallback"
                      />
                    )}
                  />
                </div>

                {endpoint && (
                  <StarterCodeCard
                    endpointName={endpoint.name}
                    provider={getStarterCodeProvider(endpoint.model_mappings)}
                  />
                )}
              </div>
            </Tabs.Content>

            <Tabs.Content value="usage">
              {experimentId && (
                <GatewayUsageSection experimentId={experimentId} tooltipLinkUrlBuilder={tooltipLinkUrlBuilder} />
              )}
            </Tabs.Content>

            <Tabs.Content value="traces" css={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              {experimentId && (
                <MonitoringConfigProvider>
                  <LogsTabContent experimentId={experimentId} />
                </MonitoringConfigProvider>
              )}
            </Tabs.Content>
          </div>

          {activeTab === 'overview' && endpoint && (
            <div
              css={{
                width: 280,
                flexShrink: 0,
                position: 'sticky',
                top: 0,
                alignSelf: 'flex-start',
              }}
            >
              <LongFormSummary
                title={intl.formatMessage({
                  defaultMessage: 'Gateway Endpoint Details',
                  description: 'Sidebar title for endpoint details panel',
                })}
              >
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    <Typography.Text bold color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                      <FormattedMessage defaultMessage="Date created" description="Label for endpoint creation date" />
                    </Typography.Text>
                    <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
                      {intl.formatDate(new Date(endpoint.created_at), {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                        timeZoneName: 'short',
                      })}
                    </Typography.Text>
                  </div>

                  {uniqueSecretNames.length > 0 && (
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                      <Typography.Text bold color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                        <FormattedMessage
                          defaultMessage="{count, plural, =1 {API key} other {API keys}}"
                          description="Label for endpoint API key(s)"
                          values={{ count: uniqueSecretNames.length }}
                        />
                      </Typography.Text>
                      {uniqueSecretNames.map((secretName) => (
                        <Link
                          key={secretName}
                          componentId="mlflow.gateway.edit-endpoint.api-key-link"
                          to={GatewayRoutes.apiKeysPageRoute}
                          css={{
                            fontSize: theme.typography.fontSizeSm,
                            color: theme.colors.actionPrimaryBackgroundDefault,
                            textDecoration: 'none',
                            '&:hover': { textDecoration: 'underline' },
                          }}
                        >
                          {secretName}
                        </Link>
                      ))}
                    </div>
                  )}

                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                      <Typography.Text bold color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                        <FormattedMessage
                          defaultMessage="Usage tracking"
                          description="Label for usage tracking toggle in sidebar"
                        />
                      </Typography.Text>
                      <Tooltip
                        componentId="mlflow.gateway.edit-endpoint.usage-tracking-info"
                        content={intl.formatMessage({
                          defaultMessage:
                            'When enabled, all requests to this endpoint will be logged as traces. This allows you to monitor usage, debug issues, and analyze performance.',
                          description: 'Tooltip explaining what usage tracking does',
                        })}
                      >
                        <InfoFillIcon
                          css={{ width: 14, height: 14, color: theme.colors.textSecondary, cursor: 'help' }}
                        />
                      </Tooltip>
                    </div>
                    <Controller
                      control={form.control}
                      name="usageTracking"
                      render={({ field }) => (
                        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                          <Switch
                            componentId="mlflow.gateway.edit-endpoint.usage-tracking.toggle"
                            checked={field.value}
                            onChange={(checked) => field.onChange(checked)}
                            aria-label="Enable usage tracking"
                          />
                          <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
                            {field.value ? (
                              <FormattedMessage defaultMessage="On" description="Usage tracking enabled state" />
                            ) : (
                              <FormattedMessage defaultMessage="Off" description="Usage tracking disabled state" />
                            )}
                          </Typography.Text>
                        </div>
                      )}
                    />
                  </div>
                </div>
              </LongFormSummary>
            </div>
          )}
        </div>
      </Tabs.Root>

      {hasChanges && activeTab === 'overview' && (
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-end',
            gap: theme.spacing.sm,
            padding: theme.spacing.md,
            borderTop: `1px solid ${theme.colors.border}`,
            flexShrink: 0,
          }}
        >
          <Button componentId="mlflow.gateway.edit-endpoint.cancel" onClick={onCancel}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
          </Button>
          <Tooltip
            componentId="mlflow.gateway.edit-endpoint.save-tooltip"
            content={
              !isFormComplete && trafficSplitModels.length > 0 && !isValidTotal
                ? intl.formatMessage({
                    defaultMessage: 'Traffic split percentages must total 100%',
                    description: 'Tooltip shown when save button is disabled due to invalid traffic split total',
                  })
                : !isFormComplete
                  ? intl.formatMessage({
                      defaultMessage: 'Please configure at least one model in traffic split',
                      description: 'Tooltip shown when save button is disabled due to incomplete form',
                    })
                  : undefined
            }
          >
            <Button
              componentId="mlflow.gateway.edit-endpoint.save"
              type="primary"
              onClick={form.handleSubmit(onSubmit)}
              loading={isSubmitting}
              disabled={!isFormComplete}
            >
              <FormattedMessage defaultMessage="Save changes" description="Save changes button" />
            </Button>
          </Tooltip>
        </div>
      )}
    </div>
  );
};
