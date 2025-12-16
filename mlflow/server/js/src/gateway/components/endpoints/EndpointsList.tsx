import {
  Accordion,
  Button,
  ChainIcon,
  ChevronRightIcon,
  Drawer,
  Empty,
  importantify,
  Input,
  LinkIcon,
  PencilIcon,
  SearchIcon,
  Spacer,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import { useBindingsQuery } from '../../hooks/useBindingsQuery';
import { formatProviderName } from '../../utils/providerUtils';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { EndpointsFilterButton, type EndpointsFilter } from './EndpointsFilterButton';
import { DeleteEndpointModal } from './DeleteEndpointModal';
import GatewayRoutes from '../../routes';
import type { Endpoint, EndpointBinding, ResourceType } from '../../types';
import { useCallback, useEffect, useMemo, useState } from 'react';

interface EndpointsListProps {
  onEndpointDeleted?: () => void;
}

export const EndpointsList = ({ onEndpointDeleted }: EndpointsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const { data: endpoints, isLoading, refetch } = useEndpointsQuery();
  const { data: bindings } = useBindingsQuery();
  const [searchFilter, setSearchFilter] = useState('');
  const [filter, setFilter] = useState<EndpointsFilter>({ providers: [] });
  const [bindingsDrawerEndpoint, setBindingsDrawerEndpoint] = useState<{
    endpointId: string;
    endpointName: string;
    bindings: EndpointBinding[];
  } | null>(null);
  const [deleteModalEndpoint, setDeleteModalEndpoint] = useState<Endpoint | null>(null);

  // Group bindings by endpoint_id for quick lookup
  const bindingsByEndpoint = useMemo(() => {
    const map = new Map<string, EndpointBinding[]>();
    bindings?.forEach((binding) => {
      const existing = map.get(binding.endpoint_id) ?? [];
      map.set(binding.endpoint_id, [...existing, binding]);
    });
    return map;
  }, [bindings]);

  // Get all unique providers from all endpoints' model mappings
  const availableProviders = useMemo(() => {
    if (!endpoints) return [];
    const providers = new Set<string>();
    endpoints.forEach((endpoint) => {
      endpoint.model_mappings?.forEach((mapping) => {
        if (mapping.model_definition?.provider) {
          providers.add(mapping.model_definition.provider);
        }
      });
    });
    return Array.from(providers);
  }, [endpoints]);

  const filteredEndpoints = useMemo(() => {
    if (!endpoints) return [];
    let filtered = endpoints;

    // Apply search filter
    if (searchFilter.trim()) {
      const lowerFilter = searchFilter.toLowerCase();
      filtered = filtered.filter((endpoint) =>
        (endpoint.name ?? endpoint.endpoint_id).toLowerCase().includes(lowerFilter),
      );
    }

    // Apply provider filter - show endpoints that have at least one model with a matching provider
    if (filter.providers.length > 0) {
      filtered = filtered.filter((endpoint) =>
        endpoint.model_mappings?.some(
          (mapping) =>
            mapping.model_definition?.provider && filter.providers.includes(mapping.model_definition.provider),
        ),
      );
    }

    return filtered;
  }, [endpoints, searchFilter, filter]);

  const handleDeleteClick = (endpoint: Endpoint) => {
    setDeleteModalEndpoint(endpoint);
  };

  const handleDeleteSuccess = () => {
    setDeleteModalEndpoint(null);
    refetch();
    onEndpointDeleted?.();
  };

  if (isLoading || !endpoints) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
          minHeight: 200,
        }}
      >
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading endpoints..." description="Loading message for endpoints list" />
      </div>
    );
  }

  if (!endpoints?.length) {
    return (
      <Empty
        image={<ChainIcon />}
        title={formatMessage({
          defaultMessage: 'No endpoints created yet',
          description: 'Empty state title for endpoints list',
        })}
        description={
          <FormattedMessage
            defaultMessage="Create an endpoint with models and API keys to securely connect MLflow features to your preferred GenAI providers."
            description="Empty state message for endpoints list explaining the feature"
          />
        }
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Input
          componentId="mlflow.gateway.endpoints-list.search"
          prefix={<SearchIcon />}
          placeholder={formatMessage({
            defaultMessage: 'Search Endpoints',
            description: 'Placeholder for endpoint search filter',
          })}
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          allowClear
          css={{ maxWidth: 300 }}
        />
        <EndpointsFilterButton availableProviders={availableProviders} filter={filter} onFilterChange={setFilter} />
      </div>

      {filteredEndpoints.length === 0 ? (
        <Empty
          image={<SearchIcon />}
          description={
            <FormattedMessage
              defaultMessage="No endpoints match your filter"
              description="Empty state message when filter returns no results"
            />
          }
        />
      ) : (
        <Table
          scrollable
          css={{
            border: `1px solid ${theme.colors.borderDecorative}`,
            borderRadius: theme.general.borderRadiusBase,
          }}
        >
          <TableRow isHeader>
            <TableHeader componentId="mlflow.gateway.endpoints-list.name-header" css={{ flex: 2 }}>
              <FormattedMessage defaultMessage="Name" description="Endpoint name column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.endpoints-list.provider-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Provider" description="Provider column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.endpoints-list.models-header" css={{ flex: 2 }}>
              <FormattedMessage defaultMessage="Models" description="Models column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.endpoints-list.bindings-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Connected resources" description="Connected resources column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.endpoints-list.modified-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Last modified" description="Last modified column header" />
            </TableHeader>
            <TableHeader
              componentId="mlflow.gateway.endpoints-list.actions-header"
              css={{ flex: 0, minWidth: 96, maxWidth: 96 }}
            />
          </TableRow>
          {filteredEndpoints.map((endpoint) => (
            <TableRow key={endpoint.endpoint_id}>
              <TableCell css={{ flex: 2 }}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <ChainIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
                  <Link
                    to={GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id)}
                    css={{
                      color: theme.colors.actionPrimaryBackgroundDefault,
                      textDecoration: 'none',
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      '&:hover': {
                        textDecoration: 'underline',
                      },
                    }}
                  >
                    {endpoint.name ?? endpoint.endpoint_id}
                  </Link>
                </div>
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <ProviderCell modelMappings={endpoint.model_mappings} />
              </TableCell>
              <TableCell css={{ flex: 2 }}>
                <ModelsCell modelMappings={endpoint.model_mappings} />
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <BindingsCell
                  bindings={bindingsByEndpoint.get(endpoint.endpoint_id) ?? []}
                  onViewBindings={() =>
                    setBindingsDrawerEndpoint({
                      endpointId: endpoint.endpoint_id,
                      endpointName: endpoint.name ?? endpoint.endpoint_id,
                      bindings: bindingsByEndpoint.get(endpoint.endpoint_id) ?? [],
                    })
                  }
                />
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <TimeAgo date={timestampToDate(endpoint.last_updated_at)} />
              </TableCell>
              <TableCell css={{ flex: 0, minWidth: 96, maxWidth: 96 }}>
                <div css={{ display: 'flex', gap: theme.spacing.xs }}>
                  <Link to={GatewayRoutes.getEditEndpointRoute(endpoint.endpoint_id)}>
                    <Button
                      componentId="mlflow.gateway.endpoints-list.edit-button"
                      type="primary"
                      icon={<PencilIcon />}
                      aria-label={formatMessage({
                        defaultMessage: 'Edit endpoint',
                        description: 'Gateway > Endpoints list > Edit endpoint button aria label',
                      })}
                    />
                  </Link>
                  <Button
                    componentId="mlflow.gateway.endpoints-list.delete-button"
                    type="primary"
                    icon={<TrashIcon />}
                    aria-label={formatMessage({
                      defaultMessage: 'Delete endpoint',
                      description: 'Gateway > Endpoints list > Delete endpoint button aria label',
                    })}
                    onClick={() => handleDeleteClick(endpoint)}
                  />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </Table>
      )}

      {/* Bindings drawer */}
      <EndpointBindingsDrawer
        open={bindingsDrawerEndpoint !== null}
        endpointName={bindingsDrawerEndpoint?.endpointName ?? ''}
        bindings={bindingsDrawerEndpoint?.bindings ?? []}
        onClose={() => setBindingsDrawerEndpoint(null)}
      />

      {/* Delete confirmation modal */}
      <DeleteEndpointModal
        open={deleteModalEndpoint !== null}
        endpoint={deleteModalEndpoint}
        bindings={deleteModalEndpoint ? bindingsByEndpoint.get(deleteModalEndpoint.endpoint_id) ?? [] : []}
        onClose={() => setDeleteModalEndpoint(null)}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};

const EndpointBindingsDrawer = ({
  open,
  endpointName,
  bindings,
  onClose,
}: {
  open: boolean;
  endpointName: string;
  bindings: EndpointBinding[];
  onClose: () => void;
}) => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const intl = useIntl();
  const [expandedSections, setExpandedSections] = useState<string[]>([]);

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
        marginBottom: theme.spacing.sm,
        overflow: 'hidden',
      },
      [`& > ${classItem} > ${classHeader}`]: {
        paddingLeft: theme.spacing.sm,
        paddingTop: theme.spacing.sm,
        paddingBottom: theme.spacing.sm,
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

  // Initialize expanded sections when drawer opens
  useEffect(() => {
    if (open && bindings.length > 0) {
      const types = Array.from(new Set(bindings.map((b) => b.resource_type)));
      setExpandedSections(types);
    }
  }, [open, bindings]);

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      onClose();
      setExpandedSections([]);
    }
  };

  const handleAccordionChange = (keys: string | string[]) => {
    setExpandedSections(Array.isArray(keys) ? keys : [keys]);
  };

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId="mlflow.gateway.endpoint-bindings.drawer"
        width={480}
        title={
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="Connected resources ({count})"
              description="Title for endpoint bindings drawer"
              values={{ count: bindings.length }}
            />
          </Typography.Title>
        }
      >
        <Spacer size="md" />
        {bindings.length === 0 ? (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No resources connected to this endpoint"
                description="Empty state when no resources connected"
              />
            }
          />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Resources using endpoint: {name}"
                description="Subtitle showing endpoint name"
                values={{ name: endpointName }}
              />
            </Typography.Text>

            <Accordion
              componentId="mlflow.gateway.endpoint-bindings.accordion"
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
                    <span css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
                      {formatResourceTypePlural(resourceType)} ({typeBindings.length})
                    </span>
                  }
                >
                  <div
                    css={{
                      maxHeight: 8 * 52, // ~8 items before scrolling
                      overflowY: 'auto',
                    }}
                  >
                    {typeBindings.map((binding) => (
                      <div
                        key={binding.binding_id}
                        css={{
                          padding: theme.spacing.sm,
                          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                          '&:last-child': { borderBottom: 'none' },
                        }}
                      >
                        <Typography.Text css={{ fontFamily: 'monospace' }}>{binding.resource_id}</Typography.Text>
                        <div css={{ marginTop: theme.spacing.xs / 2 }}>
                          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                            <FormattedMessage
                              defaultMessage="Created {date}"
                              description="When binding was created"
                              values={{
                                date: intl.formatDate(timestampToDate(binding.created_at), {
                                  year: 'numeric',
                                  month: 'short',
                                  day: 'numeric',
                                }),
                              }}
                            />
                          </Typography.Text>
                        </div>
                      </div>
                    ))}
                  </div>
                </Accordion.Panel>
              ))}
            </Accordion>
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};

const ProviderCell = ({ modelMappings }: { modelMappings: Endpoint['model_mappings'] }) => {
  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryProvider = modelMappings[0]?.model_definition?.provider;
  if (!primaryProvider) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  return <Tag componentId="mlflow.gateway.endpoints-list.provider-tag">{formatProviderName(primaryProvider)}</Tag>;
};

const ModelsCell = ({ modelMappings }: { modelMappings: Endpoint['model_mappings'] }) => {
  const { theme } = useDesignSystemTheme();

  if (!modelMappings || modelMappings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  const primaryMapping = modelMappings[0];
  const primaryModelDef = primaryMapping.model_definition;
  const additionalMappings = modelMappings.slice(1);
  const additionalCount = additionalMappings.length;

  const tooltipContent =
    additionalCount > 0 ? additionalMappings.map((m) => m.model_definition?.model_name ?? '-').join(', ') : undefined;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: theme.spacing.xs / 2 }}>
      {/* Provider's model name */}
      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
        {primaryModelDef?.model_name ?? '-'}
      </Typography.Text>
      {additionalCount > 0 && (
        <Tooltip componentId="mlflow.gateway.endpoints-list.models-more-tooltip" content={tooltipContent}>
          <button
            type="button"
            css={{
              background: 'none',
              border: 'none',
              padding: 0,
              margin: 0,
              textAlign: 'left',
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
              cursor: 'default',
              '&:hover': { textDecoration: 'underline' },
            }}
          >
            +{additionalCount} more
          </button>
        </Tooltip>
      )}
    </div>
  );
};

const BindingsCell = ({ bindings, onViewBindings }: { bindings: EndpointBinding[]; onViewBindings: () => void }) => {
  const { theme } = useDesignSystemTheme();

  if (!bindings || bindings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  return (
    <button
      type="button"
      onClick={onViewBindings}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        background: 'none',
        border: 'none',
        padding: 0,
        cursor: 'pointer',
        color: theme.colors.actionPrimaryBackgroundDefault,
        '&:hover': {
          textDecoration: 'underline',
        },
      }}
    >
      <LinkIcon css={{ color: theme.colors.textSecondary, fontSize: 14 }} />
      <Typography.Text css={{ color: 'inherit' }}>
        {bindings.length} {bindings.length === 1 ? 'resource' : 'resources'}
      </Typography.Text>
    </button>
  );
};
