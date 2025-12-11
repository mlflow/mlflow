import {
  Accordion,
  Alert,
  Breadcrumb,
  Button,
  Card,
  ChevronRightIcon,
  importantify,
  PencilIcon,
  Spinner,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCallback, useMemo, useState } from 'react';
import { useParams, Link, useNavigate } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import GatewayRoutes from '../routes';
import { formatProviderName, formatAuthMethodName, formatCredentialFieldName } from '../utils/providerUtils';
import { parseAuthConfig } from '../utils/secretUtils';
import { timestampToDate } from '../utils/dateUtils';
import { formatTokens, formatCost } from '../utils/formatters';
import { TimeAgo } from '../../shared/web-shared/browse/TimeAgo';
import { useEndpointQuery } from '../hooks/useEndpointQuery';
import { useModelsQuery } from '../hooks/useModelsQuery';
import { useSecretQuery } from '../hooks/useSecretQuery';
import { useBindingsQuery } from '../hooks/useBindingsQuery';
import type { EndpointModelMapping, ModelDefinition, Model, SecretInfo, EndpointBinding, ResourceType } from '../types';
import { MaskedValueDisplay } from '../components/secrets/MaskedValueDisplay';

const EndpointDetailsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const { endpointId } = useParams<{ endpointId: string }>();

  const { data, error, isLoading } = useEndpointQuery(endpointId ?? '');
  const endpoint = data?.endpoint;

  // Get the primary model mapping and its model definition (memoized)
  const primaryMapping = useMemo(() => endpoint?.model_mappings?.[0], [endpoint?.model_mappings]);
  const primaryModelDef = useMemo(() => primaryMapping?.model_definition, [primaryMapping?.model_definition]);
  const { data: modelsData } = useModelsQuery({ provider: primaryModelDef?.provider });

  // Get bindings for this endpoint (memoized)
  const { data: allBindings } = useBindingsQuery();
  const endpointBindings = useMemo(
    () => allBindings?.filter((b) => b.endpoint_id === endpointId) ?? [],
    [allBindings, endpointId],
  );

  const handleEdit = useCallback(() => {
    navigate(GatewayRoutes.getEditEndpointRoute(endpointId ?? ''));
  }, [navigate, endpointId]);

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
              <FormattedMessage defaultMessage="AI Gateway" description="Breadcrumb link to gateway page" />
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>
            <Link to={GatewayRoutes.gatewayPageRoute}>
              <FormattedMessage defaultMessage="Endpoints" description="Breadcrumb link to endpoints list" />
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
          <Button componentId="mlflow.gateway.endpoint-details.edit" icon={<PencilIcon />} onClick={handleEdit}>
            <FormattedMessage
              defaultMessage="Edit"
              description="Gateway > Endpoint details page > Edit endpoint button"
            />
          </Button>
        </div>
        <div
          css={{
            marginTop: theme.spacing.md,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        />
      </div>

      <div
        css={{
          flex: 1,
          display: 'flex',
          gap: theme.spacing.md,
          padding: `0 ${theme.spacing.md}px ${theme.spacing.md}px`,
          overflow: 'auto',
        }}
      >
        {/* Main content */}
        <div css={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {/* Active Configuration */}
          <Card componentId="mlflow.gateway.endpoint-details.config-card" css={{ width: '100%' }}>
            <div css={{ padding: theme.spacing.md, width: '100%' }}>
              <Typography.Title level={3}>
                <FormattedMessage defaultMessage="Active configuration" description="Section title for active config" />
              </Typography.Title>

              {hasModels ? (
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.md,
                    marginTop: theme.spacing.lg,
                    width: '100%',
                  }}
                >
                  {/* Provider */}
                  <div css={{ width: '100%' }}>
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
                  <div css={{ width: '100%' }}>
                    <Typography.Text bold color="secondary" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                      <FormattedMessage defaultMessage="Models" description="Models section label" />
                    </Typography.Text>
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, width: '100%' }}>
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
        <div css={{ width: 300, flexShrink: 0 }}>
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

                {/* Connected resources - grouped by type */}
                <ConnectedResourcesSection bindings={endpointBindings} />
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

  // Parse auth config using shared utils
  const authConfig = useMemo(() => parseAuthConfig(secretData?.secret), [secretData?.secret]);

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

        {/* API Key Name */}
        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="API Key Name:" description="API key name label" />
        </Typography.Text>
        {secret ? (
          <Typography.Text bold>{secret.secret_name}</Typography.Text>
        ) : (
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Loading..." description="Loading secret" />
          </Typography.Text>
        )}

        {/* Auth Type - only show if auth_mode is set in auth_config (indicates multi-auth provider) */}
        {authConfig?.['auth_mode'] && (
          <>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Auth Type:" description="Auth type label" />
            </Typography.Text>
            <Typography.Text>{formatAuthMethodName(String(authConfig['auth_mode']))}</Typography.Text>
          </>
        )}

        {/* Masked Key */}
        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="Masked Key:" description="Masked API key label" />
        </Typography.Text>
        {secret ? (
          <MaskedValueDisplay maskedValue={secret.masked_value} compact />
        ) : (
          <Typography.Text color="secondary">—</Typography.Text>
        )}

        {/* Config - display non-encrypted configuration */}
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

  // Filter out auth_mode since it's already shown separately as "Auth Type"
  if (!authConfig) return null;
  const configEntries = Object.entries(authConfig).filter(([key]) => key !== 'auth_mode');
  if (configEntries.length === 0) return null;

  return (
    <>
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Config:" description="Auth config label" />
      </Typography.Text>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        {configEntries.map(([key, value]) => (
          <div key={key}>
            <Typography.Text color="secondary">{formatCredentialFieldName(key)}: </Typography.Text>
            <Typography.Text css={{ fontFamily: 'monospace' }}>{String(value)}</Typography.Text>
          </div>
        ))}
      </div>
    </>
  );
};

/** Connected resources section with collapsible accordion */
const ConnectedResourcesSection = ({ bindings }: { bindings: EndpointBinding[] }) => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const intl = useIntl();

  // Get unique resource types for accordion sections
  const resourceTypes = useMemo(() => Array.from(new Set(bindings.map((b) => b.resource_type))), [bindings]);

  // Track collapsed sections (inverted logic) - new resource types are expanded by default
  const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set());

  // Derive expanded sections: all resource types except those explicitly collapsed
  const expandedSections = useMemo(
    () => resourceTypes.filter((rt) => !collapsedSections.has(rt)),
    [resourceTypes, collapsedSections],
  );

  const formatResourceTypePlural = (type: string) => {
    switch (type) {
      case 'scorer_job':
        return intl.formatMessage({ defaultMessage: 'Scorer jobs', description: 'Scorer jobs resource type plural' });
      default:
        return type;
    }
  };

  // Custom expand icon for accordion
  const getExpandIcon = useCallback(
    ({ isActive }: { isActive?: boolean }) => (
      <div
        css={importantify({
          width: theme.general.heightBase / 2,
          transform: isActive ? 'rotate(90deg)' : undefined,
          transition: 'transform 0.2s',
        })}
      >
        <ChevronRightIcon
          css={{
            svg: { width: theme.general.heightBase / 2, height: theme.general.heightBase / 2 },
          }}
        />
      </div>
    ),
    [theme],
  );

  // Accordion styles
  const accordionStyles = useMemo(() => {
    const clsPrefix = getPrefixedClassName('collapse');
    const classItem = `.${clsPrefix}-item`;
    const classHeader = `.${clsPrefix}-header`;
    const classContentBox = `.${clsPrefix}-content-box`;

    return {
      border: 'none',
      backgroundColor: 'transparent',
      [`& > ${classItem}`]: {
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
        marginBottom: theme.spacing.xs,
        overflow: 'hidden',
      },
      [`& > ${classItem} > ${classHeader}`]: {
        paddingLeft: theme.spacing.sm,
        paddingTop: theme.spacing.xs,
        paddingBottom: theme.spacing.xs,
        display: 'flex',
        alignItems: 'center',
        backgroundColor: theme.colors.backgroundSecondary,
      },
      [classContentBox]: {
        padding: 0,
      },
    };
  }, [theme, getPrefixedClassName]);

  // Group bindings by resource type
  const bindingsByType = useMemo(() => {
    const groups = new Map<ResourceType, EndpointBinding[]>();
    bindings.forEach((binding) => {
      if (!groups.has(binding.resource_type)) {
        groups.set(binding.resource_type, []);
      }
      groups.get(binding.resource_type)!.push(binding);
    });
    return groups;
  }, [bindings]);

  const handleAccordionChange = useCallback(
    (keys: string | string[]) => {
      const expandedKeys = new Set(Array.isArray(keys) ? keys : [keys]);
      // Track which sections are now collapsed (not in the expanded keys)
      setCollapsedSections(new Set(resourceTypes.filter((rt) => !expandedKeys.has(rt))));
    },
    [resourceTypes],
  );

  return (
    <div>
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Connected resources" description="Connected resources label" />
      </Typography.Text>
      <div css={{ marginTop: theme.spacing.xs }}>
        {bindings.length === 0 ? (
          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, fontStyle: 'italic' }}>
            <FormattedMessage
              defaultMessage="No resources are using this endpoint"
              description="Empty state for connected resources"
            />
          </Typography.Text>
        ) : (
          <Accordion
            componentId="mlflow.gateway.endpoint-details.bindings-accordion"
            activeKey={expandedSections}
            onChange={handleAccordionChange}
            dangerouslyAppendEmotionCSS={accordionStyles}
            dangerouslySetAntdProps={{
              expandIconPosition: 'left',
              expandIcon: getExpandIcon,
            }}
          >
            {Array.from(bindingsByType.entries()).map(([resourceType, typeBindings]) => (
              <Accordion.Panel
                key={resourceType}
                header={
                  <span
                    css={{
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      fontSize: theme.typography.fontSizeSm,
                    }}
                  >
                    {formatResourceTypePlural(resourceType)} ({typeBindings.length})
                  </span>
                }
              >
                <div
                  css={{
                    maxHeight: 8 * 28, // ~8 items before scrolling
                    overflowY: 'auto',
                  }}
                >
                  {typeBindings.map((binding) => (
                    <div
                      key={binding.binding_id}
                      css={{
                        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                        '&:last-child': { borderBottom: 'none' },
                      }}
                    >
                      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm, fontFamily: 'monospace' }}>
                        {binding.resource_id}
                      </Typography.Text>
                    </div>
                  ))}
                </div>
              </Accordion.Panel>
            ))}
          </Accordion>
        )}
      </div>
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EndpointDetailsPage);
