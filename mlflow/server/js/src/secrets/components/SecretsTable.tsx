import {
  Button,
  ColumnsIcon,
  DangerIcon,
  DropdownMenu,
  Empty,
  OverflowIcon,
  PencilIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowAction,
  TableSkeletonRows,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from '@tanstack/react-table';
import React, { useMemo } from 'react';
import { entries } from 'lodash';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { Secret } from '../types';
import {
  type SecretsColumnDef,
  SecretsTableColumns,
  SecretsTableColumnLabels,
  getColumnSizeClassName,
  getHeaderSizeClassName,
} from './SecretsTable.utils';

export interface SecretsTableProps {
  secrets: Secret[];
  loading: boolean;
  error?: Error;
  onSecretClicked?: (secret: Secret) => void;
  onUpdateSecret?: (secret: Secret) => void;
  onDeleteSecret?: (secret: Secret) => void;
  sorting: SortingState;
  setSorting: React.Dispatch<React.SetStateAction<SortingState>>;
  hiddenColumns?: string[];
  toggleHiddenColumn: (columnId: string) => void;
}

type SecretsTableMeta = {
  onSecretClicked?: SecretsTableProps['onSecretClicked'];
};

const SecretNameCell: SecretsColumnDef['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { onSecretClicked } = meta as SecretsTableMeta;
  return (
    <Typography.Link
      componentId="mlflow.secrets.table.secret_name_link"
      ellipsis
      css={{ maxWidth: '100%', textOverflow: 'ellipsis' }}
      onClick={() => {
        onSecretClicked?.(original);
      }}
    >
      {original.secret_name}
    </Typography.Link>
  );
};

const MaskedValueCell: SecretsColumnDef['cell'] = ({ row: { original } }) => {
  return <Typography.Text ellipsis>{original.masked_value}</Typography.Text>;
};

const IsSharedCell: SecretsColumnDef['cell'] = ({ row: { original } }) => {
  return <Typography.Text>{original.is_shared ? 'Yes' : 'No'}</Typography.Text>;
};

const OwnerCell: SecretsColumnDef['cell'] = ({ row: { original } }) => {
  return <Typography.Text ellipsis>{original.created_by || '-'}</Typography.Text>;
};

const CreatedAtCell: SecretsColumnDef['cell'] = ({ row: { original } }) => {
  return <Typography.Text>{Utils.formatTimestamp(original.created_at)}</Typography.Text>;
};

const UpdatedAtCell: SecretsColumnDef['cell'] = ({ row: { original } }) => {
  return <Typography.Text>{Utils.formatTimestamp(original.last_updated_at)}</Typography.Text>;
};

const BindingCountCell: SecretsColumnDef['cell'] = ({ row: { original } }) => {
  return <Typography.Text>{original.binding_count || 0}</Typography.Text>;
};

type ColumnListItem = {
  key: string;
  label: string;
};

export const SecretsTable = React.memo(
  ({
    secrets,
    loading,
    error,
    onSecretClicked,
    onUpdateSecret,
    onDeleteSecret,
    sorting,
    setSorting,
    hiddenColumns = [],
    toggleHiddenColumn,
  }: SecretsTableProps) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();

    const showEmptyState = !loading && secrets.length === 0;

    const allColumnsList = useMemo<ColumnListItem[]>(() => {
      return entries(SecretsTableColumnLabels).map(([key, label]) => ({
        key,
        label: intl.formatMessage(label),
      }));
    }, [intl]);

    const columns = useMemo<SecretsColumnDef[]>(() => {
      if (showEmptyState && !error) {
        return [];
      }

      const columns: SecretsColumnDef[] = [
        {
          header: intl.formatMessage(SecretsTableColumnLabels[SecretsTableColumns.secretName]),
          enableSorting: true,
          enableResizing: true,
          id: SecretsTableColumns.secretName,
          accessorFn: (data) => data.secret_name,
          cell: SecretNameCell,
          meta: { styles: { minWidth: 200 } },
        },
        {
          header: intl.formatMessage(SecretsTableColumnLabels[SecretsTableColumns.maskedValue]),
          enableSorting: false,
          enableResizing: true,
          id: SecretsTableColumns.maskedValue,
          accessorFn: (data) => data.masked_value,
          cell: MaskedValueCell,
          meta: { styles: { minWidth: 150 } },
        },
        {
          header: intl.formatMessage(SecretsTableColumnLabels[SecretsTableColumns.isShared]),
          enableSorting: true,
          enableResizing: true,
          id: SecretsTableColumns.isShared,
          accessorFn: (data) => data.is_shared,
          cell: IsSharedCell,
          meta: { styles: { minWidth: 100, maxWidth: 120 } },
        },
        {
          header: intl.formatMessage(SecretsTableColumnLabels[SecretsTableColumns.owner]),
          enableSorting: true,
          enableResizing: true,
          id: SecretsTableColumns.owner,
          accessorFn: (data) => data.created_by,
          cell: OwnerCell,
          meta: { styles: { minWidth: 150 } },
        },
        {
          header: intl.formatMessage(SecretsTableColumnLabels[SecretsTableColumns.createdAt]),
          enableSorting: true,
          enableResizing: true,
          id: SecretsTableColumns.createdAt,
          accessorFn: (data) => data.created_at,
          cell: CreatedAtCell,
          meta: { styles: { minWidth: 150 } },
        },
        {
          header: intl.formatMessage(SecretsTableColumnLabels[SecretsTableColumns.updatedAt]),
          enableSorting: true,
          enableResizing: true,
          id: SecretsTableColumns.updatedAt,
          accessorFn: (data) => data.last_updated_at,
          cell: UpdatedAtCell,
          meta: { styles: { minWidth: 150 } },
        },
        {
          header: intl.formatMessage(SecretsTableColumnLabels[SecretsTableColumns.bindingCount]),
          enableSorting: true,
          enableResizing: true,
          id: SecretsTableColumns.bindingCount,
          accessorFn: (data) => data.binding_count,
          cell: BindingCountCell,
          meta: { styles: { minWidth: 100, maxWidth: 120 } },
        },
      ];

      return columns.filter((column) => column.id && !hiddenColumns.includes(column.id));
    }, [intl, hiddenColumns, showEmptyState, error]);

    const table = useReactTable<Secret>({
      columns,
      data: showEmptyState && !error ? [] : secrets,
      state: { sorting },
      getCoreRowModel: getCoreRowModel(),
      getRowId: (row) => row.secret_id,
      getSortedRowModel: getSortedRowModel(),
      onSortingChange: setSorting,
      enableColumnResizing: true,
      columnResizeMode: 'onChange',
      meta: { onSecretClicked } satisfies SecretsTableMeta,
    });

    const getEmptyState = () => {
      if (error) {
        const errorMessage = error.message || '';
        const isFileStoreError = errorMessage.includes('FileStore') || errorMessage.includes('NotImplementedError');
        const isKekPassphraseError = errorMessage.includes('MLFLOW_SECRETS_KEK_PASSPHRASE');

        let title;
        let description;

        if (isFileStoreError) {
          title = (
            <FormattedMessage
              defaultMessage="Secrets not supported"
              description="Secrets table > FileStore error state title"
            />
          );
          description = (
            <FormattedMessage
              defaultMessage="Secrets are only supported with database backends (SQLite, PostgreSQL, MySQL). The current backend is using file storage. Please configure a database backend to use secrets."
              description="Secrets table > FileStore error state description"
            />
          );
        } else if (isKekPassphraseError) {
          title = (
            <FormattedMessage
              defaultMessage="Secrets not configured"
              description="Secrets table > KEK passphrase error state title"
            />
          );
          description = (
            <FormattedMessage
              defaultMessage="Secrets storage is not configured on the tracking server. Please contact an administrator to enable secrets management."
              description="Secrets table > KEK passphrase error state description"
            />
          );
        } else {
          title = (
            <FormattedMessage defaultMessage="Unable to load secrets" description="Secrets table > error state title" />
          );
          description = (
            <FormattedMessage
              defaultMessage="There was a problem loading secrets. Please try refreshing the page or contact your administrator if the problem persists."
              description="Secrets table > error state description"
            />
          );
        }

        return <Empty title={title} description={description} />;
      }
      if (!loading && secrets.length === 0) {
        return (
          <Empty
            title={
              <FormattedMessage defaultMessage="No secrets yet" description="Secrets table > no secrets recorded" />
            }
            description={
              <FormattedMessage
                defaultMessage='Create a secret to get started. Secrets can be used to securely store API keys, credentials, and other sensitive information. Use the "Create Secret" button above to add your first secret.'
                description="Secrets table > no secrets recorded"
              />
            }
          />
        );
      }
      return null;
    };

    const columnSizeVars = React.useMemo(() => {
      if (showEmptyState) {
        return {};
      }
      const headers = table.getFlatHeaders();
      const colSizes: { [key: string]: number } = {};
      headers.forEach((header) => {
        colSizes[getHeaderSizeClassName(header.id)] = header.getSize();
        colSizes[getColumnSizeClassName(header.column.id)] = header.column.getSize();
      });
      return colSizes;
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [table.getState().columnSizingInfo, table, showEmptyState]);

    if (showEmptyState && !error) {
      return (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: 400,
          }}
        >
          {getEmptyState()}
        </div>
      );
    }

    return (
      <div>
        <Table scrollable empty={getEmptyState()} style={columnSizeVars}>
          <TableRow isHeader>
            {table.getLeafHeaders().map((header) => {
              return (
                <TableHeader
                  componentId="mlflow.secrets.table.header"
                  key={header.id}
                  css={(header.column.columnDef as SecretsColumnDef).meta?.styles}
                  sortable={header.column.getCanSort()}
                  sortDirection={header.column.getIsSorted() || 'none'}
                  onToggleSort={header.column.getToggleSortingHandler()}
                  header={header}
                  column={header.column}
                  setColumnSizing={table.setColumnSizing}
                  isResizing={header.column.getIsResizing()}
                  style={{
                    flex: `calc(var(${getHeaderSizeClassName(header.id)}) / 100)`,
                  }}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </TableHeader>
              );
            })}
            <TableRowAction>
              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  <Button
                    componentId="mlflow.secrets.column_selector_dropdown"
                    icon={<ColumnsIcon />}
                    size="small"
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Select columns',
                      description: 'Secrets table > column selector dropdown aria label',
                    })}
                  />
                </DropdownMenu.Trigger>
                <DropdownMenu.Content align="end">
                  {allColumnsList.map(({ key, label }) => (
                    <DropdownMenu.CheckboxItem
                      key={key}
                      componentId="mlflow.secrets.column_toggle_button"
                      checked={!hiddenColumns.includes(key)}
                      onClick={() => toggleHiddenColumn(key)}
                    >
                      <DropdownMenu.ItemIndicator />
                      {label}
                    </DropdownMenu.CheckboxItem>
                  ))}
                </DropdownMenu.Content>
              </DropdownMenu.Root>
            </TableRowAction>
          </TableRow>
          {loading && <TableSkeletonRows table={table} />}
          {!loading &&
            !error &&
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getVisibleCells().map((cell) => {
                  const cellContent = flexRender(cell.column.columnDef.cell, cell.getContext());
                  return (
                    <TableCell
                      key={cell.id}
                      css={(cell.column.columnDef as SecretsColumnDef).meta?.styles}
                      style={{
                        flex: `calc(var(${getColumnSizeClassName(cell.column.id)}) / 100)`,
                      }}
                    >
                      {cellContent}
                    </TableCell>
                  );
                })}
                <TableRowAction>
                  <DropdownMenu.Root>
                    <DropdownMenu.Trigger asChild>
                      <Button
                        componentId="mlflow.secrets.row_actions_dropdown"
                        icon={<OverflowIcon />}
                        size="small"
                        aria-label={intl.formatMessage({
                          defaultMessage: 'Row actions',
                          description: 'Secrets table > row actions dropdown aria label',
                        })}
                      />
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Content align="end">
                      <DropdownMenu.Item
                        componentId="mlflow.secrets.update_secret"
                        onClick={() => onUpdateSecret?.(row.original)}
                      >
                        <DropdownMenu.IconWrapper>
                          <PencilIcon />
                        </DropdownMenu.IconWrapper>
                        <FormattedMessage
                          defaultMessage="Update secret value"
                          description="Secrets table > update secret action"
                        />
                      </DropdownMenu.Item>
                      <DropdownMenu.Item
                        componentId="mlflow.secrets.delete_secret"
                        onClick={() => onDeleteSecret?.(row.original)}
                      >
                        <DropdownMenu.IconWrapper>
                          <TrashIcon />
                        </DropdownMenu.IconWrapper>
                        <FormattedMessage
                          defaultMessage="Delete secret"
                          description="Secrets table > delete secret action"
                        />
                      </DropdownMenu.Item>
                    </DropdownMenu.Content>
                  </DropdownMenu.Root>
                </TableRowAction>
              </TableRow>
            ))}
        </Table>
      </div>
    );
  },
);
