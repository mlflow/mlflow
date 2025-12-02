import { useMemo, useState } from 'react';
import {
  Button,
  Empty,
  Input,
  LayerIcon,
  PencilIcon,
  SearchIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDebounce } from 'use-debounce';
import { useModelDefinitionsQuery } from '../../hooks/useModelDefinitionsQuery';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import { formatProviderName } from '../../utils/providerUtils';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { ModelDefinitionsFilterButton, type ModelDefinitionsFilter } from './ModelDefinitionsFilterButton';
import type { ModelDefinition, Endpoint } from '../../types';

interface ModelDefinitionsListProps {
  onModelDefinitionClick?: (modelDefinition: ModelDefinition) => void;
  onEditClick?: (modelDefinition: ModelDefinition) => void;
  onDeleteClick?: (modelDefinition: ModelDefinition, endpoints: Endpoint[]) => void;
  onEndpointsClick?: (modelDefinition: ModelDefinition, endpoints: Endpoint[]) => void;
}

export const ModelDefinitionsList = ({
  onModelDefinitionClick,
  onEditClick,
  onDeleteClick,
  onEndpointsClick,
}: ModelDefinitionsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const { data: modelDefinitions, isLoading: isLoadingModelDefinitions } = useModelDefinitionsQuery();
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();
  const [searchFilter, setSearchFilter] = useState('');
  const [debouncedSearchFilter] = useDebounce(searchFilter, 250);
  const [filter, setFilter] = useState<ModelDefinitionsFilter>({ providers: [] });

  // Compute how many endpoints use each model definition
  const modelDefinitionEndpointMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    if (!endpoints) return map;

    endpoints.forEach((endpoint: Endpoint) => {
      endpoint.model_mappings?.forEach((mapping) => {
        const modelDefId = mapping.model_definition_id;
        if (modelDefId) {
          if (!map.has(modelDefId)) {
            map.set(modelDefId, new Set());
          }
          map.get(modelDefId)!.add(endpoint.endpoint_id);
        }
      });
    });

    return map;
  }, [endpoints]);

  // Get all unique providers for the filter dropdown
  const availableProviders = useMemo(() => {
    if (!modelDefinitions) return [];
    const providers = new Set<string>();
    modelDefinitions.forEach((md) => {
      if (md.provider) {
        providers.add(md.provider);
      }
    });
    return Array.from(providers);
  }, [modelDefinitions]);

  const filteredModelDefinitions = useMemo(() => {
    if (!modelDefinitions) return [];
    let filtered = modelDefinitions;

    // Apply search filter (using debounced value for performance)
    if (debouncedSearchFilter.trim()) {
      const lowerFilter = debouncedSearchFilter.toLowerCase();
      filtered = filtered.filter(
        (md) => md.name.toLowerCase().includes(lowerFilter) || md.model_name.toLowerCase().includes(lowerFilter),
      );
    }

    // Apply provider filter
    if (filter.providers.length > 0) {
      filtered = filtered.filter((md) => md.provider && filter.providers.includes(md.provider));
    }

    return filtered;
  }, [modelDefinitions, debouncedSearchFilter, filter]);

  const isLoading = isLoadingModelDefinitions || isLoadingEndpoints;

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading models..." description="Loading message for models list" />
      </div>
    );
  }

  if (!modelDefinitions?.length) {
    return (
      <Empty
        image={<LayerIcon />}
        title={formatMessage({
          defaultMessage: 'No models created yet',
          description: 'Empty state title for models list',
        })}
        description={
          <FormattedMessage
            defaultMessage="Models are created when you set up an endpoint. They define the connection between your API key and a specific provider model."
            description="Empty state message for models list"
          />
        }
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Input
          componentId="mlflow.gateway.model-definitions-list.search"
          prefix={<SearchIcon />}
          placeholder={formatMessage({
            defaultMessage: 'Search models',
            description: 'Placeholder for model search filter',
          })}
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          allowClear
          css={{ maxWidth: 300 }}
        />
        <ModelDefinitionsFilterButton
          availableProviders={availableProviders}
          filter={filter}
          onFilterChange={setFilter}
        />
      </div>

      {filteredModelDefinitions.length === 0 ? (
        <Empty
          image={<SearchIcon />}
          description={
            <FormattedMessage
              defaultMessage="No models match your search"
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
            <TableHeader componentId="mlflow.gateway.model-definitions-list.name-header" css={{ flex: 2 }}>
              <FormattedMessage defaultMessage="Name" description="Model name column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.model-definitions-list.model-header" css={{ flex: 2 }}>
              <FormattedMessage defaultMessage="Provider model" description="Provider model column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.model-definitions-list.api-key-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="API Key" description="API key column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.model-definitions-list.endpoints-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Endpoints" description="Endpoints column header" />
            </TableHeader>
            <TableHeader componentId="mlflow.gateway.model-definitions-list.updated-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Last updated" description="Last updated column header" />
            </TableHeader>
            <TableHeader
              componentId="mlflow.gateway.model-definitions-list.actions-header"
              css={{ flex: 0, minWidth: 96, maxWidth: 96 }}
            />
          </TableRow>
          {filteredModelDefinitions.map((modelDefinition) => {
            const endpointCount = modelDefinitionEndpointMap.get(modelDefinition.model_definition_id)?.size ?? 0;

            return (
              <TableRow key={modelDefinition.model_definition_id}>
                <TableCell css={{ flex: 2 }}>
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    <LayerIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={() => onModelDefinitionClick?.(modelDefinition)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          onModelDefinitionClick?.(modelDefinition);
                        }
                      }}
                      css={{
                        color: theme.colors.actionPrimaryBackgroundDefault,
                        fontWeight: theme.typography.typographyBoldFontWeight,
                        cursor: 'pointer',
                        '&:hover': {
                          textDecoration: 'underline',
                        },
                      }}
                    >
                      {modelDefinition.name}
                    </span>
                  </div>
                </TableCell>
                <TableCell css={{ flex: 2 }}>
                  <div css={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Typography.Text css={{ fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }}>
                      {modelDefinition.model_name}
                    </Typography.Text>
                    <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                      {formatProviderName(modelDefinition.provider)}
                    </Typography.Text>
                  </div>
                </TableCell>
                <TableCell css={{ flex: 1 }}>
                  <Typography.Text>{modelDefinition.secret_name}</Typography.Text>
                </TableCell>
                <TableCell css={{ flex: 1 }}>
                  {endpointCount > 0 ? (
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={() => {
                        const endpointIds = modelDefinitionEndpointMap.get(modelDefinition.model_definition_id);
                        const modelDefEndpoints = endpoints?.filter((e) => endpointIds?.has(e.endpoint_id)) ?? [];
                        onEndpointsClick?.(modelDefinition, modelDefEndpoints);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          const endpointIds = modelDefinitionEndpointMap.get(modelDefinition.model_definition_id);
                          const modelDefEndpoints = endpoints?.filter((ep) => endpointIds?.has(ep.endpoint_id)) ?? [];
                          onEndpointsClick?.(modelDefinition, modelDefEndpoints);
                        }
                      }}
                      css={{
                        color: theme.colors.actionPrimaryBackgroundDefault,
                        cursor: 'pointer',
                        '&:hover': {
                          textDecoration: 'underline',
                        },
                      }}
                    >
                      {endpointCount}
                    </span>
                  ) : (
                    <Typography.Text color="secondary">0</Typography.Text>
                  )}
                </TableCell>
                <TableCell css={{ flex: 1 }}>
                  <TimeAgo date={timestampToDate(modelDefinition.last_updated_at)} />
                </TableCell>
                <TableCell css={{ flex: 0, minWidth: 96, maxWidth: 96 }}>
                  <div css={{ display: 'flex', gap: theme.spacing.xs }}>
                    <Button
                      componentId="mlflow.gateway.model-definitions-list.edit-button"
                      type="tertiary"
                      icon={<PencilIcon />}
                      aria-label={formatMessage({
                        defaultMessage: 'Edit model',
                        description: 'Edit model button aria label',
                      })}
                      onClick={() => onEditClick?.(modelDefinition)}
                    />
                    <Button
                      componentId="mlflow.gateway.model-definitions-list.delete-button"
                      type="tertiary"
                      icon={<TrashIcon />}
                      aria-label={formatMessage({
                        defaultMessage: 'Delete model',
                        description: 'Delete model button aria label',
                      })}
                      onClick={() => {
                        const endpointIds = modelDefinitionEndpointMap.get(modelDefinition.model_definition_id);
                        const modelDefinitionEndpoints =
                          endpoints?.filter((e) => endpointIds?.has(e.endpoint_id)) ?? [];
                        onDeleteClick?.(modelDefinition, modelDefinitionEndpoints);
                      }}
                    />
                  </div>
                </TableCell>
              </TableRow>
            );
          })}
        </Table>
      )}
    </div>
  );
};
