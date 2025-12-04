import { useParams, Link, useNavigate } from '../../common/utils/RoutingUtils';
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
import GatewayRoutes from '../routes';
import { formatProviderName } from '../utils/providerUtils';
import { timestampToDate } from '../utils/dateUtils';
import { TimeAgo } from '../../shared/web-shared/browse/TimeAgo';
import { useEndpointQuery } from '../hooks/useEndpointQuery';
import { useModelsQuery } from '../hooks/useModelsQuery';
import { useSecretQuery } from '../hooks/useSecretQuery';
import type { EndpointModelMapping, ModelDefinition, Model } from '../types';

const EndpointDetailsPage = () => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const { endpointId } = useParams<{ endpointId: string }>();

  const { data, error, isLoading } = useEndpointQuery(endpointId ?? '');
  const endpoint = data?.endpoint;

  // Get the primary model mapping and its model definition
  const primaryMapping = endpoint?.model_mappings?.[0];
  const primaryModelDef = primaryMapping?.model_definition;
  const { data: modelsData } = useModelsQuery({ provider: primaryModelDef?.provider });

  const handleEdit = () => {
    // TODO: Navigate to edit page or open edit modal
    navigate(GatewayRoutes.getEditEndpointRoute(endpointId ?? ''));
  };

  if (isLoading) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading endpoint..." description="Loading message for endpoint" />
        </div>
      </div>
    );
  }

  if (error || !endpoint) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.endpoint-details.error"
            type="error"
            message={(error as Error | null)?.message ?? 'Endpoint not found'}
          />
        </div>
      </div>
    );
  }

  const hasModels = endpoint.model_mappings && endpoint.model_mappings.length > 0;

  return (
    <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
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

      <div
        css={{
          flex: 1,
          display: 'flex',
          gap: theme.spacing.lg,
          padding: `0 ${theme.spacing.md}px ${theme.spacing.md}px`,
          overflow: 'auto',
        }}
      >
        {/* Main content */}
        <div css={{ flex: 2, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
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
                          modelMetadata={modelsData?.find(
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
        </div>

        {/* Sidebar */}
        <div css={{ flex: 1, maxWidth: 300 }}>
          <Card componentId="mlflow.gateway.endpoint-details.about-card">
            <div css={{ padding: theme.spacing.md }}>
              <Typography.Title level={4} css={{ marginBottom: theme.spacing.md }}>
                <FormattedMessage defaultMessage="About this endpoint" description="Sidebar title" />
              </Typography.Title>

              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                <div>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Created" description="Created at label" />
                  </Typography.Text>
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <TimeAgo date={timestampToDate(endpoint.created_at)} />
                  </div>
                </div>

                <div>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Last modified" description="Last modified label" />
                  </Typography.Text>
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <TimeAgo date={timestampToDate(endpoint.last_updated_at)} />
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
            </div>
          </Card>
        </div>
      </div>
    </div>
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
  const { data: secretData } = useSecretQuery(modelDefinition?.secret_id);

  const formatTokens = (tokens: number | null) => {
    if (tokens === null || tokens === undefined) return null;
    if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
    if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(0)}K`;
    return tokens.toString();
  };

  const formatCost = (cost: number | null) => {
    if (cost === null || cost === undefined) return null;
    if (cost === 0) return 'Free';
    const perMillion = cost * 1_000_000;
    if (perMillion < 0.01) return `$${perMillion.toFixed(4)}/1M`;
    return `$${perMillion.toFixed(2)}/1M`;
  };

  const capabilities: string[] = [];
  if (modelMetadata?.supports_function_calling) capabilities.push('Tools');
  if (modelMetadata?.supports_reasoning) capabilities.push('Reasoning');
  if (modelMetadata?.supports_prompt_caching) capabilities.push('Caching');

  const contextWindow = formatTokens(modelMetadata?.max_input_tokens ?? null);
  const inputCost = formatCost(modelMetadata?.input_cost_per_token ?? null);
  const outputCost = formatCost(modelMetadata?.output_cost_per_token ?? null);

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
      <div css={{ marginBottom: theme.spacing.sm }}>
        <Link
          to={GatewayRoutes.getModelDefinitionDetailsRoute(modelDefinition.model_definition_id)}
          css={{
            color: theme.colors.actionPrimaryBackgroundDefault,
            textDecoration: 'none',
            fontSize: theme.typography.fontSizeMd,
            fontWeight: theme.typography.typographyBoldFontWeight,
            '&:hover': { textDecoration: 'underline' },
          }}
        >
          {modelDefinition.name}
        </Link>
      </div>

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
        {modelMetadata && (contextWindow || inputCost || outputCost) && (
          <>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Specs:" description="Model specs label" />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              {[
                contextWindow &&
                  intl.formatMessage(
                    { defaultMessage: 'Context: {tokens}', description: 'Context window size' },
                    { tokens: contextWindow },
                  ),
                inputCost &&
                  intl.formatMessage(
                    { defaultMessage: 'Input: {cost}', description: 'Input cost' },
                    { cost: inputCost },
                  ),
                outputCost &&
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

        {/* API Key section */}
        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="API Key:" description="API key label" />
        </Typography.Text>
        {secret ? (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Typography.Text bold>{secret.secret_name}</Typography.Text>
              <Typography.Text
                css={{
                  fontFamily: 'monospace',
                  fontSize: theme.typography.fontSizeSm,
                  backgroundColor: theme.colors.tagDefault,
                  padding: `2px ${theme.spacing.xs}px`,
                  borderRadius: theme.general.borderRadiusBase,
                }}
              >
                {secret.masked_value}
              </Typography.Text>
            </div>
          </div>
        ) : (
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Loading..." description="Loading secret" />
          </Typography.Text>
        )}
      </div>
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EndpointDetailsPage);
