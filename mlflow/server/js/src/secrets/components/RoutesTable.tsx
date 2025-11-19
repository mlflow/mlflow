import {
  Button,
  ColumnsIcon,
  DropdownMenu,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowAction,
  TableSkeletonRows,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { KeyValueTag } from '@mlflow/mlflow/src/common/components/KeyValueTag';
import type { SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from '@tanstack/react-table';
import React, { useMemo } from 'react';
import { entries } from 'lodash';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { useIntl } from '@databricks/i18n';
import type { Endpoint } from '../types';
import { ProviderBadge } from './ProviderBadge';
import {
  type RoutesColumnDef,
  RoutesTableColumns,
  RoutesTableColumnLabels,
  getColumnSizeClassName,
  getHeaderSizeClassName,
} from './RoutesTable.utils';

export interface RoutesTableProps {
  routes: Endpoint[];
  loading: boolean;
  error?: Error;
  sorting: SortingState;
  setSorting: React.Dispatch<React.SetStateAction<SortingState>>;
  hiddenColumns?: string[];
  toggleHiddenColumn: (columnId: string) => void;
  onRowClick?: (route: Endpoint) => void;
}

const SecretNameCell = ({ secretName, secretId }: { secretName?: string; secretId: string }) => {
  const displayName = secretName || secretId;

  return (
    <Typography.Text ellipsis css={{ maxWidth: 200, fontWeight: 500 }}>
      {displayName}
    </Typography.Text>
  );
};

const BindingCountBadge = ({ count }: { count: number }) => {
  const { theme } = useDesignSystemTheme();

  const getColor = (count: number) => {
    if (count === 0) return { bg: theme.colors.backgroundSecondary, text: theme.colors.textSecondary };
    if (count < 3) return { bg: '#4285F415', text: '#4285F4' };
    if (count < 10) return { bg: '#10A37F15', text: '#10A37F' };
    return { bg: '#FF362115', text: '#FF3621' };
  };

  const colors = getColor(count);

  return (
    <div
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '4px 12px',
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: colors.bg,
        color: colors.text,
        fontSize: theme.typography.fontSizeMd,
        fontWeight: 700,
        minWidth: 36,
      }}
    >
      {count}
    </div>
  );
};

const TagsCell = ({ tags }: { tags?: Array<{ key: string; value: string }> | Record<string, string> }) => {
  const { theme } = useDesignSystemTheme();

  // Convert tags to KeyValueEntity format
  const tagEntities = Array.isArray(tags)
    ? tags
    : tags
    ? Object.entries(tags).map(([key, value]) => ({ key, value }))
    : [];

  if (tagEntities.length === 0) {
    return <Typography.Text css={{ color: theme.colors.textSecondary, fontStyle: 'italic' }}>-</Typography.Text>;
  }

  const displayTags = tagEntities.slice(0, 3);
  const remainingCount = tagEntities.length - 3;

  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap', alignItems: 'center', overflow: 'hidden' }}>
      {displayTags.map((tag) => (
        <KeyValueTag key={tag.key} tag={tag} maxWidth={150} />
      ))}
      {remainingCount > 0 && (
        <Tooltip
          componentId="mlflow.routes.table.more_tags_tooltip"
          content={
            <div css={{ padding: theme.spacing.sm, maxWidth: 300 }}>
              {tagEntities.slice(3).map((tag) => (
                <div key={tag.key} css={{ marginBottom: theme.spacing.xs }}>
                  <Typography.Text css={{ fontWeight: 600 }}>{tag.key}: </Typography.Text>
                  <Typography.Text>{tag.value}</Typography.Text>
                </div>
              ))}
            </div>
          }
        >
          <span>
            <Tag componentId="mlflow.routes.table.more_tags">
              <Typography.Text>+{remainingCount}</Typography.Text>
            </Tag>
          </span>
        </Tooltip>
      )}
    </div>
  );
};

type ColumnListItem = {
  key: string;
  label: string;
};

export const RoutesTable = React.memo(
  ({
    routes,
    loading,
    error,
    sorting,
    setSorting,
    hiddenColumns = [],
    toggleHiddenColumn,
    onRowClick,
  }: RoutesTableProps) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();

    const showEmptyState = !loading && routes.length === 0;

    const allColumnsList = useMemo<ColumnListItem[]>(() => {
      return entries(RoutesTableColumnLabels).map(([key, label]) => ({
        key,
        label: intl.formatMessage(label),
      }));
    }, [intl]);

    const columns = useMemo<RoutesColumnDef[]>(() => {
      if (showEmptyState && !error) {
        return [];
      }

      const columns: RoutesColumnDef[] = [
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.name]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.name,
          accessorFn: (data) => data.name || data.endpoint_id,
          cell: ({ row }) => {
            const displayName = row.original.name || row.original.endpoint_id;
            return (
              <Typography.Text ellipsis css={{ maxWidth: 200, fontWeight: 600 }}>
                {displayName}
              </Typography.Text>
            );
          },
          meta: { styles: { minWidth: 150, maxWidth: 250 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.description]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.description,
          accessorFn: (data) => data.description || '',
          cell: ({ row }) => {
            const description = row.original.description;

            if (!description) {
              return (
                <Typography.Text css={{ color: theme.colors.textSecondary, fontStyle: 'italic' }}>-</Typography.Text>
              );
            }

            return (
              <Tooltip componentId="mlflow.routes.table.description_tooltip" content={description}>
                <span>
                  <Typography.Text css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {description}
                  </Typography.Text>
                </span>
              </Tooltip>
            );
          },
          meta: { styles: { minWidth: 200, maxWidth: 400 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.secretName]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.secretName,
          accessorFn: (data) => data.secret_name || data.secret_id,
          cell: ({ row }) => <SecretNameCell secretName={row.original.secret_name} secretId={row.original.secret_id} />,
          meta: { styles: { minWidth: 180, maxWidth: 280 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.modelName]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.modelName,
          accessorFn: (data) => data.model_name,
          cell: ({ row }) => (
            <Tag componentId="mlflow.routes.table.model_tag">
              <Typography.Text ellipsis css={{ maxWidth: 180 }}>
                {row.original.model_name}
              </Typography.Text>
            </Tag>
          ),
          meta: { styles: { minWidth: 150, maxWidth: 250 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.provider]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.provider,
          accessorFn: (data) => data.provider,
          cell: ({ row }) => <ProviderBadge provider={row.original.provider} />,
          meta: { styles: { minWidth: 130, maxWidth: 180 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.tags]),
          enableSorting: false,
          enableResizing: true,
          id: RoutesTableColumns.tags,
          accessorFn: (data) => data.tags,
          cell: ({ row }) => <TagsCell tags={row.original.tags} />,
          meta: { styles: { minWidth: 200, maxWidth: 350 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.bindingCount]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.bindingCount,
          accessorFn: (data) => data.binding_count || 0,
          cell: ({ row }) => <BindingCountBadge count={row.original.binding_count || 0} />,
          meta: { styles: { minWidth: 100, maxWidth: 120 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.createdAt]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.createdAt,
          accessorFn: (data) => data.created_at,
          cell: ({ row }) => (
            <Typography.Text css={{ color: theme.colors.textSecondary }}>
              {Utils.formatTimestamp(row.original.created_at)}
            </Typography.Text>
          ),
          meta: { styles: { minWidth: 120, maxWidth: 180 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.lastUpdatedAt]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.lastUpdatedAt,
          accessorFn: (data) => data.last_updated_at,
          cell: ({ row }) => (
            <Typography.Text css={{ color: theme.colors.textSecondary }}>
              {Utils.formatTimestamp(row.original.last_updated_at)}
            </Typography.Text>
          ),
          meta: { styles: { minWidth: 120, maxWidth: 180 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.createdBy]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.createdBy,
          accessorFn: (data) => data.created_by,
          cell: ({ row }) => (
            <Typography.Text ellipsis css={{ maxWidth: 150 }}>
              {row.original.created_by || '-'}
            </Typography.Text>
          ),
          meta: { styles: { minWidth: 120, maxWidth: 180 } },
        },
        {
          header: intl.formatMessage(RoutesTableColumnLabels[RoutesTableColumns.lastUpdatedBy]),
          enableSorting: true,
          enableResizing: true,
          id: RoutesTableColumns.lastUpdatedBy,
          accessorFn: (data) => data.last_updated_by,
          cell: ({ row }) => (
            <Typography.Text ellipsis css={{ maxWidth: 150 }}>
              {row.original.last_updated_by || '-'}
            </Typography.Text>
          ),
          meta: { styles: { minWidth: 120, maxWidth: 180 } },
        },
      ];

      return columns.filter((column) => column.id && !hiddenColumns.includes(column.id));
    }, [intl, hiddenColumns, showEmptyState, error, theme]);

    const table = useReactTable<Endpoint>({
      columns,
      data: showEmptyState && !error ? [] : routes,
      state: { sorting },
      getCoreRowModel: getCoreRowModel(),
      getRowId: (row) => row.endpoint_id,
      getSortedRowModel: getSortedRowModel(),
      onSortingChange: setSorting,
      enableColumnResizing: true,
      columnResizeMode: 'onChange',
    });

    const columnSizeVars = React.useMemo(() => {
      if (showEmptyState) {
        return {};
      }
      const headers = table.getFlatHeaders();
      const colSizes: { [key: string]: number } = {};
      headers.forEach((header) => {
        colSizes[`${getHeaderSizeClassName(header.id)}`] = header.getSize();
        colSizes[`${getColumnSizeClassName(header.id)}`] = header.column.getSize();
      });
      return colSizes;
    }, [showEmptyState, table]);

    if (showEmptyState) {
      return (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: 400,
          }}
        >
          <Typography.Text css={{ color: theme.colors.textSecondary }}>No endpoints found</Typography.Text>
        </div>
      );
    }

    return (
      <div css={{ overflow: 'auto', flex: 1 }}>
        <Table scrollable style={columnSizeVars}>
          <TableRow isHeader>
            {table.getLeafHeaders().map((header) => {
              return (
                <TableHeader
                  componentId="mlflow.routes.table.header"
                  key={header.id}
                  css={(header.column.columnDef as RoutesColumnDef).meta?.styles}
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
                    componentId="mlflow.routes.column_selector_dropdown"
                    icon={<ColumnsIcon />}
                    size="small"
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Select columns',
                      description: 'Endpoints table > column selector dropdown aria label',
                    })}
                  />
                </DropdownMenu.Trigger>
                <DropdownMenu.Content align="end">
                  {allColumnsList.map(({ key, label }) => (
                    <DropdownMenu.CheckboxItem
                      key={key}
                      componentId="mlflow.routes.column_toggle_button"
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
              <TableRow
                key={row.id}
                onClick={() => onRowClick?.(row.original)}
                css={{
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                  },
                }}
              >
                {row.getVisibleCells().map((cell) => {
                  const cellContent = flexRender(cell.column.columnDef.cell, cell.getContext());
                  return (
                    <TableCell
                      key={cell.id}
                      css={(cell.column.columnDef as RoutesColumnDef).meta?.styles}
                      style={{
                        flex: `calc(var(${getColumnSizeClassName(cell.column.id)}) / 100)`,
                      }}
                    >
                      {cellContent}
                    </TableCell>
                  );
                })}
                <TableRowAction />
              </TableRow>
            ))}
        </Table>
      </div>
    );
  },
);
