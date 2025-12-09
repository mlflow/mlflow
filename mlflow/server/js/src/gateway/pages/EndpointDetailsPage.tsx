import { useParams, Link, useNavigate } from '../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import {
  Alert,
  Breadcrumb,
  Button,
  Card,
  PencilIcon,
  Spinner,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useQuery } from '../../common/utils/reactQueryHooks';
import React, { useCallback, useMemo } from 'react';
import { GatewayApi } from '../api';
import GatewayRoutes from '../routes';
import { formatProviderName, formatAuthMethodName, formatSecretFieldName } from '../utils/providerUtils';
import { parseAuthConfig, parseMaskedValues, isSingleMaskedValue } from '../utils/secretUtils';
import { TimeAgo } from '../../shared/web-shared/browse/TimeAgo';
import { LongFormLayout, LongFormSummary } from '../../common/components/long-form';
import type { EndpointModelMapping, ModelDefinition, Model, SecretInfo } from '../types';
import { formatTokens, formatCost } from '../utils/formatters';

const useEndpointQuery = (endpointId: string) => {
  return useQuery(['gateway_endpoint', endpointId], {
    queryFn: () => GatewayApi.getEndpoint(endpointId),
    retry: false,
    enabled: Boolean(endpointId),
  });
};

const useModelsMetadataQuery = (provider: string | undefined) => {
  return useQuery(['gateway_models', provider], {
    queryFn: () => GatewayApi.listModels(provider),
    enabled: Boolean(provider),
  });
};

const EndpointDetailsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const { endpointId } = useParams<{ endpointId: string }>();

  const { data, error, isLoading } = useEndpointQuery(endpointId ?? '');
  const endpoint = data?.endpoint;

  // Get the primary model mapping and its model definition
  const primaryMapping = endpoint?.model_mappings?.[0];
  const primaryModelDef = primaryMapping?.model_definition;
  const { data: modelsData } = useModelsMetadataQuery(primaryModelDef?.provider);

  const handleEdit = useCallback(() => {
    navigate(GatewayRoutes.getEditEndpointRoute(endpointId ?? ''));
  }, [navigate, endpointId]);

  if (isLoading) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading endpoint..." description="Loading message for endpoint" />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (error || !endpoint) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.endpoint-details.error"
            type="error"
            message={(error as Error | null)?.message ?? 'Endpoint not found'}
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  const hasModels = endpoint.model_mappings && endpoint.model_mappings.length > 0;

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <div css={{ padding: theme.spacing.md }}>
        <Breadcrumb includeTrailingCaret>
          <Breadcrumb.Item>
            <Link to={GatewayRoutes.gatewayPageRoute}>
              <FormattedMessage defaultMessage="Gateway" description="Breadcrumb link to gateway page" />
            </Link>
          </Breadcrumb.Item>
        </Breadcrumb>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginTop: theme.spacing.sm,
          }}
        >
          <Typography.Title level={2}>{endpoint.name ?? endpoint.endpoint_id}</Typography.Title>
          <Button
            componentId="mlflow.gateway.endpoint-details.edit"
            type="tertiary"
            icon={<PencilIcon />}
            onClick={handleEdit}
          >
            <FormattedMessage defaultMessage="Edit" description="Edit endpoint button" />
          </Button>
        </div>
      </div>

      <LongFormLayout
        sidebar={
          <LongFormSummary
            title={intl.formatMessage({
              defaultMessage: 'About this endpoint',
              description: 'Sidebar title',
            })}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              <div>
                <Typography.Text color="secondary">
                  <FormattedMessage defaultMessage="Created" description="Created at label" />
                </Typography.Text>
                <div css={{ marginTop: theme.spacing.xs }}>
                  <TimeAgo date={new Date(endpoint.created_at)} />
                </div>
              </div>

              <div>
                <Typography.Text color="secondary">
                  <FormattedMessage defaultMessage="Last modified" description="Last modified label" />
                </Typography.Text>
                <div css={{ marginTop: theme.spacing.xs }}>
                  <TimeAgo date={new Date(endpoint.last_updated_at)} />
                </div>
              </div>

              {endpoint.created_by && (
                <div>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Created by" description="Created by label" />
                  </Typography.Text>
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <Typography.Text>{endpoint.created_by}</Typography.Text>
                  </div>
                </div>
              )}
            </div>
          </LongFormSummary>
        }
      >
        {/* Active Configuration */}
        <Card componentId="mlflow.gateway.endpoint-details.config-card">
          <div css={{ padding: theme.spacing.md }}>
            <Typography.Title level={3}>
              <FormattedMessage defaultMessage="Active configuration" description="Section title for active config" />
            </Typography.Title>

            {hasModels ? (
              <div
                css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, marginTop: theme.spacing.lg }}
              >
                {/* Provider */}
                <div>
                  <Typography.Text bold color="secondary" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                    <FormattedMessage defaultMessage="Provider" description="Provider label" />
                  </Typography.Text>
                  <div
                    css={{
                      padding: theme.spacing.md,
                      border: `1px solid ${theme.colors.borderDecorative}`,
                      borderRadius: theme.general.borderRadiusBase,
                      backgroundColor: theme.colors.backgroundSecondary,
                    }}
                  >
                    <Tag componentId="mlflow.gateway.endpoint-details.provider">
                      {formatProviderName(primaryModelDef?.provider ?? '')}
                    </Tag>
                  </div>
                </div>

                {/* Models */}
                <div>
                  <Typography.Text bold color="secondary" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                    <FormattedMessage defaultMessage="Models" description="Models section label" />
                  </Typography.Text>
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                    {endpoint.model_mappings.map((mapping: EndpointModelMapping) => (
                      <ModelCard
                        key={mapping.mapping_id}
                        modelDefinition={mapping.model_definition}
                        modelMetadata={modelsData?.models?.find(
                          (m: Model) => m.model === mapping.model_definition?.model_name,
                        )}
                      />
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="No models configured for this endpoint"
                  description="Message when no models configured"
                />
              </Typography.Text>
            )}
          </div>
        </Card>
      </LongFormLayout>
    </ScrollablePageWrapper>
  );
};

/** Helper component to display model card with metadata */
const ModelCard = ({
  modelDefinition,
  modelMetadata,
}: {
  modelDefinition: ModelDefinition | undefined;
  modelMetadata: Model | undefined;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Fetch secret for this model definition
  const { data: secretData } = useQuery(['gateway_secret', modelDefinition?.secret_id], {
    queryFn: () => GatewayApi.getSecret(modelDefinition!.secret_id),
    enabled: Boolean(modelDefinition?.secret_id),
  });

  // Memoize capabilities array
  const capabilities = useMemo(() => {
    const caps: string[] = [];
    if (modelMetadata?.supports_function_calling) caps.push('Tools');
    if (modelMetadata?.supports_reasoning) caps.push('Reasoning');
    if (modelMetadata?.supports_prompt_caching) caps.push('Caching');
    return caps;
  }, [
    modelMetadata?.supports_function_calling,
    modelMetadata?.supports_reasoning,
    modelMetadata?.supports_prompt_caching,
  ]);

  // Memoize formatted values
  const contextWindow = useMemo(
    () => formatTokens(modelMetadata?.max_input_tokens ?? null),
    [modelMetadata?.max_input_tokens],
  );
  const inputCost = useMemo(
    () => formatCost(modelMetadata?.input_cost_per_token ?? null),
    [modelMetadata?.input_cost_per_token],
  );
  const outputCost = useMemo(
    () => formatCost(modelMetadata?.output_cost_per_token ?? null),
    [modelMetadata?.output_cost_per_token],
  );

  // Parse auth config and masked values using shared utils
  const authConfig = useMemo(() => parseAuthConfig(secretData?.secret), [secretData?.secret]);
  const maskedValues = useMemo(() => parseMaskedValues(secretData?.secret), [secretData?.secret]);

  if (!modelDefinition) {
    return null;
  }

  const secret = secretData?.secret;

  return (
    <div
      css={{
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      {/* Model definition name as title */}
      <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
        {modelDefinition.name}
      </Typography.Title>

      {/* Properties grid */}
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr',
          gap: `${theme.spacing.xs}px ${theme.spacing.md}px`,
          alignItems: 'baseline',
        }}
      >
        {/* Model name */}
        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="Model:" description="Model name label" />
        </Typography.Text>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flexWrap: 'wrap' }}>
          <Typography.Text css={{ fontFamily: 'monospace' }}>{modelDefinition.model_name}</Typography.Text>
          {/* Capabilities */}
          {capabilities.length > 0 && (
            <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap' }}>
              {capabilities.map((cap) => (
                <Tag key={cap} color="turquoise" componentId={`mlflow.gateway.endpoint-details.capability.${cap}`}>
                  {cap}
                </Tag>
              ))}
            </div>
          )}
        </div>

        {/* Model specs - context and cost */}
        {modelMetadata && (contextWindow !== '-' || inputCost !== '-' || outputCost !== '-') && (
          <>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Specs:" description="Model specs label" />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              {[
                contextWindow !== '-' &&
                  intl.formatMessage(
                    { defaultMessage: 'Context: {tokens}', description: 'Context window size' },
                    { tokens: contextWindow },
                  ),
                inputCost !== '-' &&
                  intl.formatMessage(
                    { defaultMessage: 'Input: {cost}', description: 'Input cost' },
                    { cost: inputCost },
                  ),
                outputCost !== '-' &&
                  intl.formatMessage(
                    { defaultMessage: 'Output: {cost}', description: 'Output cost' },
                    { cost: outputCost },
                  ),
              ]
                .filter(Boolean)
                .join(' • ')}
            </Typography.Text>
          </>
        )}

        {/* API Key name */}
        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="API Key:" description="API key name label" />
        </Typography.Text>
        {secret ? (
          <Typography.Text bold>{secret.secret_name}</Typography.Text>
        ) : (
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Loading..." description="Loading secret" />
          </Typography.Text>
        )}

        {/* Auth type - only show if auth_mode is set in auth_config (indicates multi-auth provider) */}
        {authConfig?.['auth_mode'] && (
          <>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Auth Type:" description="Auth type label" />
            </Typography.Text>
            <Typography.Text>{formatAuthMethodName(String(authConfig['auth_mode']))}</Typography.Text>
          </>
        )}

        {/* Masked keys */}
        {maskedValues && maskedValues.length > 0 && (
          <>
            <Typography.Text color="secondary">
              {isSingleMaskedValue(maskedValues) ? (
                <FormattedMessage defaultMessage="Masked Key:" description="Masked API key label (singular)" />
              ) : (
                <FormattedMessage defaultMessage="Masked Keys:" description="Masked API keys section label" />
              )}
            </Typography.Text>
            <div css={{ display: 'flex', flexDirection: 'column' }}>
              {maskedValues.map(([key, value], index) =>
                key === '' ? (
                  // Single value without key label
                  <Typography.Text
                    key={index}
                    css={{
                      fontFamily: 'monospace',
                      fontSize: theme.typography.fontSizeSm,
                      backgroundColor: theme.colors.tagDefault,
                      padding: `2px ${theme.spacing.xs}px`,
                      borderRadius: theme.general.borderRadiusBase,
                      width: 'fit-content',
                    }}
                  >
                    {value}
                  </Typography.Text>
                ) : (
                  // Multiple values with key labels
                  <div key={key} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    <Typography.Text color="secondary">{formatSecretFieldName(key)}:</Typography.Text>
                    <Typography.Text
                      css={{
                        fontFamily: 'monospace',
                        fontSize: theme.typography.fontSizeSm,
                        backgroundColor: theme.colors.tagDefault,
                        padding: `2px ${theme.spacing.xs}px`,
                        borderRadius: theme.general.borderRadiusBase,
                      }}
                    >
                      {value}
                    </Typography.Text>
                  </div>
                ),
              )}
            </div>
          </>
        )}

        {/* Auth Config section - display non-encrypted configuration */}
        <AuthConfigDisplay secret={secret} />
      </div>
    </div>
  );
};

/** Helper component to display auth config from secret */
const AuthConfigDisplay = ({ secret }: { secret: SecretInfo | undefined }) => {
  const { theme } = useDesignSystemTheme();

  // Parse auth config using shared utils
  const authConfig = useMemo(() => parseAuthConfig(secret), [secret]);

  // Filter out auth_mode since it's already displayed in the Auth Type row
  const filteredEntries = authConfig ? Object.entries(authConfig).filter(([key]) => key !== 'auth_mode') : [];

  if (filteredEntries.length === 0) return null;

  return (
    <>
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Config:" description="Auth config label" />
      </Typography.Text>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        {filteredEntries.map(([key, value]) => (
          <div key={key}>
            <Typography.Text color="secondary">{formatSecretFieldName(key)}: </Typography.Text>
            <Typography.Text css={{ fontFamily: 'monospace' }}>{String(value)}</Typography.Text>
          </div>
        ))}
      </div>
    </>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EndpointDetailsPage);
