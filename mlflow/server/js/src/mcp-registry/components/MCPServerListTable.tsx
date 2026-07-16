import { useCallback, useMemo, useRef, useState } from 'react';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import type { CursorPaginationProps } from '@databricks/design-system';
import {
  CursorPagination,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { CellContext, ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { useIntl } from 'react-intl';

import type { MCPServer } from '../types';

import MCPRegistryRoutes from '../routes';
import { MCPServersEmptyState } from './MCPRegistryEmptyState';
import { MCPServerIcon } from './MCPServerIcon';
import { MCPServerTags } from './MCPServerTags';
import { textEllipsisStyles, flexRowStyles, monoFontStyles } from '../styles';
import { isServerDimmed } from '../utils';
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

const MCPServerTagsCell = ({ row: { original } }: CellContext<MCPServer, unknown>) => {
  return <MCPServerTags tags={original.tags || {}} />;
};

const MCPServerEndpointsCell = ({ row: { original } }: CellContext<MCPServer, unknown>) => {
  const latestBinding = original.latest_version
    ? (original.access_bindings ?? []).find((b) => b.resolved_version?.version === original.latest_version)
    : undefined;
  if (!latestBinding) return '—';
  return (
    <Typography.Text
      size="sm"
      css={{
        ...monoFontStyles,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}
    >
      {latestBinding.endpoint_url}
    </Typography.Text>
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
        accessorFn: (row) => row.name,
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
          defaultMessage: 'Endpoints',
          description: 'Header for the endpoints column in the MCP servers table',
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
}: {
  servers?: MCPServer[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  isLoading?: boolean;
  isFiltered?: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  pageSizeSelect?: CursorPaginationProps['pageSizeSelect'];
}) => {
  const { theme } = useDesignSystemTheme();
  // One auth check + pure isServerDimmed per row (avoid N hooks)
  const isAuthAvailable = useIsAuthAvailable();
  const columns = useMCPServerTableColumns();

  const table = useReactTable('mlflow/server/js/src/mcp-registry/components/MCPServerListTable.tsx', {
    data: servers ?? [],
    columns,
    getCoreRowModel: coreRowModel,
    getRowId,
  });

  const isEmptyList = !isLoading && (!servers || servers.length === 0);
  const emptyState = isEmptyList ? (
    <MCPServersEmptyState isFiltered={isFiltered} componentId="mlflow.mcp_registry.table.empty_state.create_server" />
  ) : null;

  return (
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
          <TableHeader componentId="mlflow.mcp_registry.table.header" key={header.id}>
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
                  style={{ opacity: isDimmed ? 0.5 : 1 }}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          );
        })
      )}
    </Table>
  );
};
