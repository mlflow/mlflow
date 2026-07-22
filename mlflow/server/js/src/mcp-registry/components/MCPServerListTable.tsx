import { useCallback, useMemo, useRef, useState } from 'react';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import type { CursorPaginationProps } from '@databricks/design-system';
import {
  Button,
  CursorPagination,
  PencilIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { CellContext, ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPServer } from '../types';

import MCPRegistryRoutes from '../routes';
import { MCPServersEmptyState } from './MCPRegistryEmptyState';
import { MCPServerIcon } from './MCPServerIcon';
import { MCPServerTags } from './MCPServerTags';
import { QuickConnectModal } from './QuickConnectModal';
import { textEllipsisStyles, flexRowStyles, monoFontStyles, noShrinkStyles } from '../styles';
import { useUpdateMCPServerTags } from '../hooks/useUpdateMCPServerTags';
import {
  findLatestEndpoint,
  isServerDimmed,
  formatTransportType,
  resolveDisplayName,
  getServerPermissions,
} from '../utils';
import { Link } from '../../common/utils/RoutingUtils';
import { useIsAuthAvailable } from '../../account/hooks';
import Utils from '../../common/utils/Utils';

const coreRowModel = getCoreRowModel<MCPServer>();
const getRowId = (row: MCPServer) => row.name;

const MCPServerNameCell = ({ getValue, row }: CellContext<MCPServer, unknown>) => {
  const { theme } = useDesignSystemTheme();
  const value = getValue() as string;
  return (
    <span css={flexRowStyles(theme)}>
      <MCPServerIcon icons={row.original.icons} name={value} />
      <Link
        componentId="mlflow.mcp_registry.table.name_link"
        to={MCPRegistryRoutes.getMCPServerDetailRoute(row.original.name)}
      >
        {value}
      </Link>
    </span>
  );
};

const MCPServerDescriptionCell = ({ getValue }: CellContext<MCPServer, unknown>) => {
  const value = getValue() as string | undefined;
  const ref = useRef<HTMLSpanElement>(null);
  const [isTruncated, setIsTruncated] = useState(false);

  const checkTruncation = useCallback(() => {
    if (ref.current) {
      setIsTruncated(ref.current.scrollWidth > ref.current.clientWidth);
    }
  }, []);

  if (!value) return '—';

  const content = (
    <span ref={ref} onMouseEnter={checkTruncation} css={{ display: 'block', ...textEllipsisStyles }}>
      {value}
    </span>
  );

  return isTruncated ? (
    <Tooltip content={value} componentId="mlflow.mcp_registry.table.description_tooltip">
      {content}
    </Tooltip>
  ) : (
    content
  );
};

interface MCPServerTableMeta {
  onEditTags?: (server: MCPServer) => void;
  onOpenConnect?: (server: MCPServer) => void;
}

const MCPServerTagsCell = ({
  row: { original },
  table: {
    options: { meta },
  },
}: CellContext<MCPServer, unknown>) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { onEditTags } = (meta ?? {}) as MCPServerTableMeta;
  const containsTags = Object.keys(original.tags || {}).length > 0;

  return (
    <div css={{ display: 'flex', alignItems: 'center' }}>
      {containsTags && <MCPServerTags tags={original.tags || {}} />}
      {onEditTags && getServerPermissions(original).canUpdate && (
        <Button
          componentId="mlflow.mcp_registry.table.edit_tags"
          size="small"
          icon={containsTags ? <PencilIcon /> : undefined}
          onClick={(e: React.MouseEvent) => {
            e.stopPropagation();
            onEditTags(original);
          }}
          aria-label={intl.formatMessage({
            defaultMessage: 'Edit tags',
            description: 'Label for the edit tags button in the MCP servers table',
          })}
          children={
            !containsTags ? (
              <FormattedMessage
                defaultMessage="Add tags"
                description="Label for the add tags button in the MCP servers table"
              />
            ) : undefined
          }
          css={{
            flexShrink: 0,
            marginLeft: containsTags ? theme.spacing.sm : 0,
            opacity: 0,
            '[role=row]:hover &': { opacity: 1 },
            '[role=row]:focus-within &': { opacity: 1 },
          }}
          type="tertiary"
        />
      )}
    </div>
  );
};

const MCPServerEndpointsCell = ({
  row: { original },
  table: {
    options: { meta },
  },
}: CellContext<MCPServer, unknown>) => {
  const { theme } = useDesignSystemTheme();
  const { onOpenConnect } = (meta ?? {}) as MCPServerTableMeta;

  const latestEndpoint = findLatestEndpoint(original);
  if (!latestEndpoint) return '—';

  const displayUrl = latestEndpoint.url.replace(/^https?:\/\//, '');

  return (
    <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, overflow: 'hidden' }}>
      <Tag componentId="mlflow.mcp_registry.table.transport_tag" color="indigo" css={noShrinkStyles}>
        {formatTransportType(latestEndpoint.transport_type)}
      </Tag>
      <Tooltip
        content={<span css={{ wordBreak: 'break-all' }}>{latestEndpoint.url}</span>}
        componentId="mlflow.mcp_registry.table.endpoint_tooltip"
      >
        <span
          role="button"
          tabIndex={0}
          onClick={(e) => {
            e.stopPropagation();
            onOpenConnect?.(original);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              e.stopPropagation();
              onOpenConnect?.(original);
            }
          }}
          css={{
            ...monoFontStyles,
            fontSize: theme.typography.fontSizeSm,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            cursor: 'pointer',
            color: theme.colors.actionPrimaryBackgroundDefault,
            '&:hover': { textDecoration: 'underline' },
          }}
        >
          {displayUrl}
        </span>
      </Tooltip>
    </span>
  );
};

const useMCPServerTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const columns: ColumnDef<MCPServer>[] = [
      {
        header: intl.formatMessage({
          defaultMessage: 'Name',
          description: 'Header for the name column in the MCP servers table',
        }),
        accessorFn: (row) => resolveDisplayName(row),
        id: 'name',
        cell: MCPServerNameCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Description',
          description: 'Header for the description column in the MCP servers table',
        }),
        accessorKey: 'description',
        id: 'description',
        cell: MCPServerDescriptionCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Latest version',
          description: 'Header for the latest version column in the MCP servers table',
        }),
        id: 'latestVersion',
        accessorFn: (row) => row.latest_version || '—',
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Access endpoint',
          description: 'Header for the access endpoint column in the MCP servers table',
        }),
        id: 'endpoints',
        cell: MCPServerEndpointsCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Last modified',
          description: 'Header for the last modified column in the MCP servers table',
        }),
        id: 'lastModified',
        accessorFn: ({ last_updated_timestamp }) =>
          last_updated_timestamp ? Utils.formatTimestamp(last_updated_timestamp, intl) : '',
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Tags',
          description: 'Header for the tags column in the MCP servers table',
        }),
        id: 'tags',
        cell: MCPServerTagsCell,
      },
    ];
    return columns;
  }, [intl]);
};

export const MCPServerListTable = ({
  servers,
  hasNextPage,
  hasPreviousPage,
  isLoading,
  isFiltered,
  onNextPage,
  onPreviousPage,
  pageSizeSelect,
  onCreateServer,
}: {
  servers?: MCPServer[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  isLoading?: boolean;
  isFiltered?: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  pageSizeSelect?: CursorPaginationProps['pageSizeSelect'];
  onCreateServer?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  // One auth check + pure isServerDimmed per row (avoid N hooks)
  const isAuthAvailable = useIsAuthAvailable();
  const columns = useMCPServerTableColumns();
  const { EditTagsModal, showEditServerTagsModal } = useUpdateMCPServerTags();
  const [connectServer, setConnectServer] = useState<MCPServer | null>(null);

  const tableMeta = useMemo<MCPServerTableMeta>(
    () => ({ onEditTags: showEditServerTagsModal, onOpenConnect: setConnectServer }),
    [showEditServerTagsModal],
  );

  const table = useReactTable('mlflow/server/js/src/mcp-registry/components/MCPServerListTable.tsx', {
    data: servers ?? [],
    columns,
    getCoreRowModel: coreRowModel,
    getRowId,
    meta: tableMeta,
  });

  const isEmptyList = !isLoading && (!servers || servers.length === 0);
  const emptyState = isEmptyList ? (
    <MCPServersEmptyState
      isFiltered={isFiltered}
      componentId="mlflow.mcp_registry.table.empty_state.create_server"
      onCreateServer={onCreateServer}
    />
  ) : null;

  return (
    <>
      <Table
        scrollable
        pagination={
          <CursorPagination
            hasNextPage={hasNextPage}
            hasPreviousPage={hasPreviousPage}
            onNextPage={onNextPage}
            onPreviousPage={onPreviousPage}
            pageSizeSelect={pageSizeSelect}
            componentId="mlflow.mcp_registry.table.pagination"
          />
        }
        empty={emptyState}
      >
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader
              componentId="mlflow.mcp_registry.table.header"
              key={header.id}
              style={
                header.id === 'endpoints' ? { flex: 1.5 } : header.id === 'latestVersion' ? { flex: 0.5 } : undefined
              }
            >
              {flexRender(header.column.columnDef.header, header.getContext())}
            </TableHeader>
          ))}
        </TableRow>
        {isLoading ? (
          <TableSkeletonRows table={table} />
        ) : (
          table.getRowModel().rows.map((row) => {
            const isDimmed = isAuthAvailable && isServerDimmed(row.original);
            return (
              <TableRow key={row.id} css={{ height: theme.general.buttonHeight }}>
                {row.getAllCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    css={{ alignItems: 'center' }}
                    style={{
                      opacity: isDimmed ? 0.5 : 1,
                      ...(cell.column.id === 'endpoints'
                        ? { flex: 1.5 }
                        : cell.column.id === 'latestVersion'
                          ? { flex: 0.5 }
                          : undefined),
                    }}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            );
          })
        )}
      </Table>
      {EditTagsModal}
      {connectServer && <QuickConnectModal visible server={connectServer} onClose={() => setConnectServer(null)} />}
    </>
  );
};
