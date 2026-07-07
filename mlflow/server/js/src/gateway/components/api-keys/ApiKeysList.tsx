import {
  Alert,
  Button,
  Checkbox,
  Empty,
  Input,
  KeyIcon,
  LinkIcon,
  SearchIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useApiKeysListData } from '../../hooks/useApiKeysListData';
import { formatProviderName } from '../../utils/providerUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { ApiKeysFilterButton, type ApiKeysFilter } from './ApiKeysFilterButton';
import {
  ApiKeysColumnsButton,
  ApiKeysColumn,
  DEFAULT_VISIBLE_COLUMNS,
  type ToggleableApiKeysColumn,
} from './ApiKeysColumnsButton';
import { BulkDeleteApiKeyModal } from './BulkDeleteApiKeyModal';
import type { SecretInfo, Endpoint, EndpointBinding } from '../../types';
import { useMemo, useState } from 'react';

interface ApiKeysListProps {
  onCreateClick?: () => void;
  onKeyClick?: (secret: SecretInfo) => void;
  onEndpointsClick?: (secret: SecretInfo, endpoints: Endpoint[]) => void;
  onBindingsClick?: (secret: SecretInfo, bindings: EndpointBinding[]) => void;
  onApiKeyDeleted?: () => void;
}

export const ApiKeysList = ({
  onCreateClick,
  onKeyClick,
  onEndpointsClick,
  onBindingsClick,
  onApiKeyDeleted,
}: ApiKeysListProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();
  const [searchFilter, setSearchFilter] = useState('');
  const [filter, setFilter] = useState<ApiKeysFilter>({ providers: [] });
  const [visibleColumns, setVisibleColumns] = useState<ToggleableApiKeysColumn[]>(DEFAULT_VISIBLE_COLUMNS);
  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});
  const [deleteModalSecrets, setDeleteModalSecrets] = useState<SecretInfo[]>([]);

  const {
    secrets,
    filteredSecrets,
    isLoading,
    error,
    availableProviders,
    getEndpointsForSecret,
    getBindingsForSecret,
    getEndpointCount,
    getBindingCount,
  } = useApiKeysListData({ searchFilter, filter });

  const selectedSecrets = useMemo(
    () => filteredSecrets.filter((s) => rowSelection[s.secret_id]),
    [filteredSecrets, rowSelection],
  );

  const selectedCount = selectedSecrets.length;
  const allSelected = filteredSecrets.length > 0 && filteredSecrets.every((s) => rowSelection[s.secret_id]);
  const someSelected = selectedCount > 0 && !allSelected;

  const handleSelectAll = () => {
    if (allSelected) {
      setRowSelection({});
    } else {
      const next: Record<string, boolean> = {};
      filteredSecrets.forEach((s) => {
        next[s.secret_id] = true;
      });
      setRowSelection(next);
    }
  };

  const handleSelectRow = (secretId: string) => {
    setRowSelection((prev) => {
      const next = { ...prev };
      if (next[secretId]) {
        delete next[secretId];
      } else {
        next[secretId] = true;
      }
      return next;
    });
  };

  const handleDeleteClick = () => {
    setDeleteModalSecrets(selectedSecrets);
  };

  const handleDeleteSuccess = () => {
    setDeleteModalSecrets([]);
    setRowSelection({});
    onApiKeyDeleted?.();
  };

  if (error && !secrets.length) {
    return (
      <Alert
        componentId="mlflow.gateway.api-keys.error"
        type="error"
        message={
          <FormattedMessage
            defaultMessage="Failed to load API keys. Please check your connection and try again."
            description="Gateway > API keys list > Error loading API keys"
          />
        }
        closable={false}
      />
    );
  }

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
          componentId="mlflow.gateway.api-keys.search"
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
        <div css={{ marginLeft: 'auto', display: 'flex', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.gateway.api-keys.create-button" type="primary" onClick={onCreateClick}>
            <FormattedMessage defaultMessage="Create" description="Gateway > API keys list > Create button" />
          </Button>
          <Button
            componentId="mlflow.gateway.api-keys.bulk-delete-button"
            disabled={selectedCount === 0}
            danger
            onClick={handleDeleteClick}
          >
            {selectedCount > 0 ? (
              <FormattedMessage
                defaultMessage="Delete ({count})"
                description="Gateway > API keys list > Delete button with count"
                values={{ count: selectedCount }}
              />
            ) : (
              <FormattedMessage defaultMessage="Delete" description="Gateway > API keys list > Delete button" />
            )}
          </Button>
        </div>
      </div>

      <Table
        scrollable
        noMinHeight
        empty={getEmptyState()}
        css={{
          borderLeft: `1px solid ${theme.colors.border}`,
          borderRight: `1px solid ${theme.colors.border}`,
          borderTop: `1px solid ${theme.colors.border}`,
          borderBottom: filteredSecrets.length === 0 ? `1px solid ${theme.colors.border}` : 'none',
          borderRadius: theme.general.borderRadiusBase,
          overflow: 'hidden',
        }}
      >
        <TableRow isHeader>
          <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
            <Checkbox
              componentId="mlflow.gateway.api-keys.select-all-checkbox"
              isChecked={someSelected ? null : allSelected}
              onChange={handleSelectAll}
            />
          </TableCell>
          <TableHeader componentId="mlflow.gateway.api-keys.name-header" css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Key name" description="API key name column header" />
          </TableHeader>
          {visibleColumns.includes(ApiKeysColumn.PROVIDER) && (
            <TableHeader componentId="mlflow.gateway.api-keys.provider-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Provider" description="Provider column header" />
            </TableHeader>
          )}
          {visibleColumns.includes(ApiKeysColumn.ENDPOINTS) && (
            <TableHeader componentId="mlflow.gateway.api-keys.endpoints-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Endpoints" description="Endpoints using this key column header" />
            </TableHeader>
          )}
          {visibleColumns.includes(ApiKeysColumn.USED_BY) && (
            <TableHeader componentId="mlflow.gateway.api-keys.used-by-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Used by" description="Used by column header" />
            </TableHeader>
          )}
          {visibleColumns.includes(ApiKeysColumn.LAST_UPDATED) && (
            <TableHeader componentId="mlflow.gateway.api-keys.updated-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Last updated" description="Last updated column header" />
            </TableHeader>
          )}
          {visibleColumns.includes(ApiKeysColumn.CREATED) && (
            <TableHeader componentId="mlflow.gateway.api-keys.created-header" css={{ flex: 1 }}>
              <FormattedMessage defaultMessage="Created" description="Created column header" />
            </TableHeader>
          )}
        </TableRow>
        {filteredSecrets.map((secret) => {
          const endpointCount = getEndpointCount(secret.secret_id);
          const bindingCount = getBindingCount(secret.secret_id);

          return (
            <TableRow key={secret.secret_id}>
              <TableCell css={{ flex: 0, minWidth: 40, maxWidth: 40 }}>
                <Checkbox
                  componentId="mlflow.gateway.api-keys.row-checkbox"
                  isChecked={Boolean(rowSelection[secret.secret_id])}
                  onChange={() => handleSelectRow(secret.secret_id)}
                />
              </TableCell>
              <TableCell css={{ flex: 2 }}>
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
                        <FormattedMessage
                          defaultMessage="{count, plural, one {# resource} other {# resources}}"
                          description="Gateway > API keys list > Used by column binding count"
                          values={{ count: bindingCount }}
                        />
                      </Typography.Text>
                    </button>
                  ) : (
                    <Typography.Text color="secondary">-</Typography.Text>
                  )}
                </TableCell>
              )}
              {visibleColumns.includes(ApiKeysColumn.LAST_UPDATED) && (
                <TableCell css={{ flex: 1 }}>
                  <TimeAgo date={new Date(secret.last_updated_at)} />
                </TableCell>
              )}
              {visibleColumns.includes(ApiKeysColumn.CREATED) && (
                <TableCell css={{ flex: 1 }}>
                  <TimeAgo date={new Date(secret.created_at)} />
                </TableCell>
              )}
            </TableRow>
          );
        })}
      </Table>

      <BulkDeleteApiKeyModal
        open={deleteModalSecrets.length > 0}
        secrets={deleteModalSecrets}
        getEndpointsForSecret={getEndpointsForSecret}
        onClose={() => setDeleteModalSecrets([])}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};
