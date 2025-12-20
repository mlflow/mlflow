import {
  Button,
  Empty,
  Input,
  KeyIcon,
  LinkIcon,
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
import { useApiKeysListData } from '../../hooks/useApiKeysListData';
import { formatProviderName } from '../../utils/providerUtils';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { ApiKeysFilterButton, type ApiKeysFilter } from './ApiKeysFilterButton';
import { ApiKeysColumnsButton, ApiKeysColumn, DEFAULT_VISIBLE_COLUMNS } from './ApiKeysColumnsButton';
import type { SecretInfo, Endpoint, EndpointBinding, ModelDefinition } from '../../types';
import { useState } from 'react';

interface ApiKeysListProps {
  onKeyClick?: (secret: SecretInfo) => void;
  onEditClick?: (secret: SecretInfo) => void;
  onDeleteClick?: (
    secret: SecretInfo,
    modelDefinitions: ModelDefinition[],
    endpoints: Endpoint[],
    bindingCount: number,
  ) => void;
  onEndpointsClick?: (secret: SecretInfo, endpoints: Endpoint[]) => void;
  onBindingsClick?: (secret: SecretInfo, bindings: EndpointBinding[]) => void;
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
  const [searchFilter, setSearchFilter] = useState('');
  const [filter, setFilter] = useState<ApiKeysFilter>({ providers: [] });
  const [visibleColumns, setVisibleColumns] = useState<ApiKeysColumn[]>(DEFAULT_VISIBLE_COLUMNS);

  const {
    secrets,
    filteredSecrets,
    isLoading,
    availableProviders,
    getModelDefinitionsForSecret,
    getEndpointsForSecret,
    getBindingsForSecret,
    getEndpointCount,
    getBindingCount,
  } = useApiKeysListData({ searchFilter, filter });

  if (isLoading || !secrets.length) {
    if (isLoading) {
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
          <FormattedMessage defaultMessage="Loading API keys..." description="Loading message for API keys list" />
        </div>
      );
    }
  }

  const isFiltered = searchFilter.trim().length > 0 || filter.providers.length > 0;

  const getEmptyState = () => {
    const isEmptyList = secrets.length === 0;
    if (filteredSecrets.length === 0 && isFiltered) {
      return (
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No API keys found"
              description="Empty state title when filter returns no results"
            />
          }
          description={null}
        />
      );
    }
    if (isEmptyList) {
      return (
        <Empty
          image={<KeyIcon />}
          title={
            <FormattedMessage defaultMessage="No API keys created" description="Empty state title for API keys list" />
          }
          description={
            <FormattedMessage
              defaultMessage='Use "Create API key" button to create a new API key'
              description="Empty state message for API keys list explaining how to create"
            />
          }
        />
      );
    }
    return null;
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Input
          componentId="mlflow.gateway.api-keys-list.search"
          prefix={<SearchIcon />}
          placeholder={formatMessage({
            defaultMessage: 'Search API Keys',
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

      <Table
        scrollable
        empty={getEmptyState()}
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
          const secretModels = getModelDefinitionsForSecret(secret.secret_id);
          const endpointCount = getEndpointCount(secret.secret_id);
          const bindingCount = getBindingCount(secret.secret_id);

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
                      onClick={() => onEndpointsClick?.(secret, getEndpointsForSecret(secret.secret_id))}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          onEndpointsClick?.(secret, getEndpointsForSecret(secret.secret_id));
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
                    <button
                      type="button"
                      onClick={() => {
                        const secretBindings = getBindingsForSecret(secret.secret_id);
                        onBindingsClick?.(secret, secretBindings);
                      }}
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
                        {bindingCount} {bindingCount === 1 ? 'resource' : 'resources'}
                      </Typography.Text>
                    </button>
                  ) : (
                    <Typography.Text color="secondary">-</Typography.Text>
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
                    type="primary"
                    icon={<PencilIcon />}
                    aria-label={formatMessage({
                      defaultMessage: 'Edit API key',
                      description: 'Gateway > API keys list > Edit API key button aria label',
                    })}
                    onClick={() => onEditClick?.(secret)}
                  />
                  <Button
                    componentId="mlflow.gateway.api-keys-list.delete-button"
                    type="primary"
                    icon={<TrashIcon />}
                    aria-label={formatMessage({
                      defaultMessage: 'Delete API key',
                      description: 'Gateway > API keys list > Delete API key button aria label',
                    })}
                    onClick={() =>
                      onDeleteClick?.(secret, secretModels, getEndpointsForSecret(secret.secret_id), bindingCount)
                    }
                  />
                </div>
              </TableCell>
            </TableRow>
          );
        })}
      </Table>
    </div>
  );
};
