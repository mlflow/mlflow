import {
  Button,
  Empty,
  Input,
  KeyIcon,
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
import { useSecretsQuery } from '../../hooks/useSecretsQuery';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import { useBindingsQuery } from '../../hooks/useBindingsQuery';
import { formatProviderName } from '../../utils/providerUtils';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { ApiKeysFilterButton, type ApiKeysFilter } from './ApiKeysFilterButton';
import { ApiKeysColumnsButton, ApiKeysColumn, DEFAULT_VISIBLE_COLUMNS } from './ApiKeysColumnsButton';
import type { Secret, Endpoint, EndpointBinding } from '../../types';
import { useMemo, useState } from 'react';

interface ApiKeysListProps {
  onKeyClick?: (secret: Secret) => void;
  onEditClick?: (secret: Secret) => void;
  onDeleteClick?: (secret: Secret, endpoints: Endpoint[], bindingCount: number) => void;
  onEndpointsClick?: (secret: Secret, endpoints: Endpoint[]) => void;
  onBindingsClick?: (secret: Secret, bindings: EndpointBinding[]) => void;
}

export const ApiKeysList = ({
  onKeyClick,
  onEditClick,
  onDeleteClick,
  onEndpointsClick,
  onBindingsClick,
}: ApiKeysListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const { data: secrets, isLoading: isLoadingSecrets } = useSecretsQuery();
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();
  const { data: bindings, isLoading: isLoadingBindings } = useBindingsQuery();
  const [searchFilter, setSearchFilter] = useState('');
  const [filter, setFilter] = useState<ApiKeysFilter>({ providers: [] });
  const [visibleColumns, setVisibleColumns] = useState<ApiKeysColumn[]>(DEFAULT_VISIBLE_COLUMNS);

  // Compute how many endpoints use each secret
  const secretEndpointMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    if (!endpoints) return map;

    endpoints.forEach((endpoint: Endpoint) => {
      endpoint.model_mappings?.forEach((mapping) => {
        const secretId = mapping.model_definition?.secret_id;
        if (secretId) {
          if (!map.has(secretId)) {
            map.set(secretId, new Set());
          }
          map.get(secretId)!.add(endpoint.endpoint_id);
        }
      });
    });

    return map;
  }, [endpoints]);

  // Create a map from endpoint_id to secret_ids for binding lookups
  const endpointToSecretsMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    secretEndpointMap.forEach((endpointIds, secretId) => {
      endpointIds.forEach((endpointId) => {
        if (!map.has(endpointId)) {
          map.set(endpointId, new Set());
        }
        map.get(endpointId)!.add(secretId);
      });
    });
    return map;
  }, [secretEndpointMap]);

  // Compute how many bindings exist for endpoints that use each secret
  const secretBindingMap = useMemo(() => {
    const map = new Map<string, number>();
    if (!bindings || !endpointToSecretsMap.size) return map;

    // Count bindings per secret
    bindings.forEach((binding: EndpointBinding) => {
      const secretIds = endpointToSecretsMap.get(binding.endpoint_id);
      if (secretIds) {
        secretIds.forEach((secretId) => {
          const current = map.get(secretId) ?? 0;
          map.set(secretId, current + 1);
        });
      }
    });

    return map;
  }, [bindings, endpointToSecretsMap]);

  // Get all bindings for a specific secret
  const getBindingsForSecret = (secretId: string): EndpointBinding[] => {
    if (!bindings) return [];
    const endpointIds = secretEndpointMap.get(secretId);
    if (!endpointIds) return [];
    return bindings.filter((binding) => endpointIds.has(binding.endpoint_id));
  };

  // Get all unique providers for the filter dropdown
  const availableProviders = useMemo(() => {
    if (!secrets) return [];
    const providers = new Set<string>();
    secrets.forEach((secret) => {
      if (secret.provider) {
        providers.add(secret.provider);
      }
    });
    return Array.from(providers);
  }, [secrets]);

  const filteredSecrets = useMemo(() => {
    if (!secrets) return [];
    let filtered = secrets;

    // Apply search filter
    if (searchFilter.trim()) {
      const lowerFilter = searchFilter.toLowerCase();
      filtered = filtered.filter((secret) => secret.secret_name.toLowerCase().includes(lowerFilter));
    }

    // Apply provider filter
    if (filter.providers.length > 0) {
      filtered = filtered.filter((secret) => secret.provider && filter.providers.includes(secret.provider));
    }

    return filtered;
  }, [secrets, searchFilter, filter]);

  const isLoading = isLoadingSecrets || isLoadingEndpoints || isLoadingBindings;

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading API keys..." description="Loading message for API keys list" />
      </div>
    );
  }

  if (!secrets?.length) {
    return (
      <Empty
        image={<KeyIcon />}
        title={formatMessage({
          defaultMessage: 'No API keys created yet',
          description: 'Empty state title for API keys list',
        })}
        description={
          <FormattedMessage
            defaultMessage="API keys store your credentials for connecting to GenAI providers. Create a key to get started."
            description="Empty state message for API keys list"
          />
        }
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Input
          componentId="mlflow.gateway.api-keys-list.search"
          prefix={<SearchIcon />}
          placeholder={formatMessage({
            defaultMessage: 'Search keys',
            description: 'Placeholder for API key search filter',
          })}
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          allowClear
          css={{ maxWidth: 300 }}
        />
        <ApiKeysFilterButton availableProviders={availableProviders} filter={filter} onFilterChange={setFilter} />
        <ApiKeysColumnsButton visibleColumns={visibleColumns} onColumnsChange={setVisibleColumns} />
      </div>

      {filteredSecrets.length === 0 ? (
        <Empty
          image={<SearchIcon />}
          description={
            <FormattedMessage
              defaultMessage="No API keys match your search"
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
            <TableHeader componentId="mlflow.gateway.api-keys-list.name-header" css={{ flex: 2 }}>
              <FormattedMessage defaultMessage="Key name" description="API key name column header" />
            </TableHeader>
            {visibleColumns.includes(ApiKeysColumn.PROVIDER) && (
              <TableHeader componentId="mlflow.gateway.api-keys-list.provider-header" css={{ flex: 1 }}>
                <FormattedMessage defaultMessage="Provider" description="Provider column header" />
              </TableHeader>
            )}
            {visibleColumns.includes(ApiKeysColumn.ENDPOINTS) && (
              <TableHeader componentId="mlflow.gateway.api-keys-list.endpoints-header" css={{ flex: 1 }}>
                <FormattedMessage defaultMessage="Endpoints" description="Endpoints using this key column header" />
              </TableHeader>
            )}
            {visibleColumns.includes(ApiKeysColumn.USED_BY) && (
              <TableHeader componentId="mlflow.gateway.api-keys-list.used-by-header" css={{ flex: 1 }}>
                <FormattedMessage defaultMessage="Used by" description="Used by column header" />
              </TableHeader>
            )}
            {visibleColumns.includes(ApiKeysColumn.LAST_UPDATED) && (
              <TableHeader componentId="mlflow.gateway.api-keys-list.updated-header" css={{ flex: 1 }}>
                <FormattedMessage defaultMessage="Last updated" description="Last updated column header" />
              </TableHeader>
            )}
            {visibleColumns.includes(ApiKeysColumn.CREATED) && (
              <TableHeader componentId="mlflow.gateway.api-keys-list.created-header" css={{ flex: 1 }}>
                <FormattedMessage defaultMessage="Created" description="Created column header" />
              </TableHeader>
            )}
            <TableHeader
              componentId="mlflow.gateway.api-keys-list.actions-header"
              css={{ flex: 0, minWidth: 96, maxWidth: 96 }}
            />
          </TableRow>
          {filteredSecrets.map((secret) => {
            const endpointCount = secretEndpointMap.get(secret.secret_id)?.size ?? 0;
            const bindingCount = secretBindingMap.get(secret.secret_id) ?? 0;

            return (
              <TableRow key={secret.secret_id}>
                <TableCell css={{ flex: 2 }}>
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    <KeyIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={() => onKeyClick?.(secret)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          onKeyClick?.(secret);
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
                      {secret.secret_name}
                    </span>
                  </div>
                </TableCell>
                {visibleColumns.includes(ApiKeysColumn.PROVIDER) && (
                  <TableCell css={{ flex: 1 }}>
                    {secret.provider ? (
                      <Typography.Text>{formatProviderName(secret.provider)}</Typography.Text>
                    ) : (
                      <Typography.Text color="secondary">-</Typography.Text>
                    )}
                  </TableCell>
                )}
                {visibleColumns.includes(ApiKeysColumn.ENDPOINTS) && (
                  <TableCell css={{ flex: 1 }}>
                    {endpointCount > 0 ? (
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={() => {
                          const endpointIds = secretEndpointMap.get(secret.secret_id);
                          const secretEndpoints = endpoints?.filter((e) => endpointIds?.has(e.endpoint_id)) ?? [];
                          onEndpointsClick?.(secret, secretEndpoints);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            const endpointIds = secretEndpointMap.get(secret.secret_id);
                            const secretEndpoints = endpoints?.filter((ep) => endpointIds?.has(ep.endpoint_id)) ?? [];
                            onEndpointsClick?.(secret, secretEndpoints);
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
                      <Typography.Text color="secondary">{endpointCount}</Typography.Text>
                    )}
                  </TableCell>
                )}
                {visibleColumns.includes(ApiKeysColumn.USED_BY) && (
                  <TableCell css={{ flex: 1 }}>
                    {bindingCount > 0 ? (
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={() => {
                          const secretBindings = getBindingsForSecret(secret.secret_id);
                          onBindingsClick?.(secret, secretBindings);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            const secretBindings = getBindingsForSecret(secret.secret_id);
                            onBindingsClick?.(secret, secretBindings);
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
                        {bindingCount}
                      </span>
                    ) : (
                      <Typography.Text color="secondary">{bindingCount}</Typography.Text>
                    )}
                  </TableCell>
                )}
                {visibleColumns.includes(ApiKeysColumn.LAST_UPDATED) && (
                  <TableCell css={{ flex: 1 }}>
                    <TimeAgo date={timestampToDate(secret.last_updated_at)} />
                  </TableCell>
                )}
                {visibleColumns.includes(ApiKeysColumn.CREATED) && (
                  <TableCell css={{ flex: 1 }}>
                    <TimeAgo date={timestampToDate(secret.created_at)} />
                  </TableCell>
                )}
                <TableCell css={{ flex: 0, minWidth: 96, maxWidth: 96 }}>
                  <div css={{ display: 'flex', gap: theme.spacing.xs }}>
                    <Button
                      componentId="mlflow.gateway.api-keys-list.edit-button"
                      type="tertiary"
                      icon={<PencilIcon />}
                      aria-label={formatMessage({
                        defaultMessage: 'Edit API key',
                        description: 'Edit API key button aria label',
                      })}
                      onClick={() => onEditClick?.(secret)}
                    />
                    <Button
                      componentId="mlflow.gateway.api-keys-list.delete-button"
                      type="tertiary"
                      icon={<TrashIcon />}
                      aria-label={formatMessage({
                        defaultMessage: 'Delete API key',
                        description: 'Delete API key button aria label',
                      })}
                      onClick={() => {
                        const endpointIds = secretEndpointMap.get(secret.secret_id);
                        const secretEndpoints = endpoints?.filter((e) => endpointIds?.has(e.endpoint_id)) ?? [];
                        onDeleteClick?.(secret, secretEndpoints, bindingCount);
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
