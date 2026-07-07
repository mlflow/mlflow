import { useMemo } from 'react';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import {
  ChevronRightIcon,
  Empty,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPServerVersion } from '../types';
import { STATUS_TAG_COLOR } from '../utils';
import { ModelVersionTableAliasesCell } from '../../model-registry/components/aliases/ModelVersionTableAliasesCell';
import Utils from '../../common/utils/Utils';

interface MCPServerVersionListMeta {
  serverName: string;
  serverDisplayName: string;
  aliasesByVersion: Record<string, string[]>;
  showEditAliasesModal?: (versionNumber: string) => void;
}

const MCPServerVersionCell: ColumnDef<MCPServerVersion>['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { serverName, serverDisplayName, aliasesByVersion, showEditAliasesModal } = meta as MCPServerVersionListMeta;
  const aliases = aliasesByVersion[original.version] || [];

  const rawDisplayName = original.display_name || original.server_json?.title;
  const versionDisplayName = rawDisplayName && rawDisplayName !== serverDisplayName ? rawDisplayName : undefined;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flexWrap: 'wrap' }}>
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="{version}"
            description="MCP server version item label"
            values={{ version: original.version }}
          />
        </Typography.Text>
        <Tag componentId="mlflow.mcp_registry.detail.version_status_tag" color={STATUS_TAG_COLOR[original.status]}>
          {original.status}
        </Tag>
        <ModelVersionTableAliasesCell
          modelName={serverName}
          version={original.version}
          aliases={aliases}
          onAddEdit={() => {
            showEditAliasesModal?.(original.version);
          }}
        />
      </div>
      {versionDisplayName && (
        <Typography.Text
          size="sm"
          color="secondary"
          css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
          title={versionDisplayName}
        >
          {versionDisplayName}
        </Typography.Text>
      )}
      {original.creation_timestamp && (
        <Typography.Text size="sm" color="secondary">
          {Utils.formatTimestamp(original.creation_timestamp, intl)}
        </Typography.Text>
      )}
    </div>
  );
};

export const MCPServerVersionList = ({
  versions,
  selectedVersion,
  onSelectVersion,
  isLoading,
  serverName,
  serverDisplayName,
  aliasesByVersion,
  showEditAliasesModal,
}: {
  versions?: MCPServerVersion[];
  selectedVersion?: string;
  onSelectVersion: (version: string) => void;
  isLoading?: boolean;
  serverName: string;
  serverDisplayName: string;
  aliasesByVersion: Record<string, string[]>;
  showEditAliasesModal?: (versionNumber: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const columns = useMemo<ColumnDef<MCPServerVersion>[]>(
    () => [
      {
        id: 'version',
        header: intl.formatMessage({
          defaultMessage: 'Version',
          description: 'Header for the version column in the MCP server versions table',
        }),
        accessorKey: 'version',
        cell: MCPServerVersionCell,
      },
    ],
    [intl],
  );

  const table = useReactTable('mlflow/server/js/src/mcp-registry/components/MCPServerVersionList.tsx', {
    data: versions ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.version,
    meta: { serverName, serverDisplayName, aliasesByVersion, showEditAliasesModal },
  });

  const emptyState =
    !isLoading && (!versions || versions.length === 0) ? (
      <Empty
        title={
          <FormattedMessage defaultMessage="No versions" description="Empty state when MCP server has no versions" />
        }
        description={null}
      />
    ) : null;

  return (
    <div css={{ flex: 1, overflow: 'hidden' }}>
      <Table scrollable empty={emptyState}>
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader componentId="mlflow.mcp_registry.detail.versions.header" key={header.id}>
              {flexRender(header.column.columnDef.header, header.getContext())}
            </TableHeader>
          ))}
        </TableRow>
        {isLoading ? (
          <TableSkeletonRows table={table} />
        ) : (
          table.getRowModel().rows.map((row) => {
            const version = row.original.version;
            const isSelected = selectedVersion === version;
            return (
              <TableRow
                key={row.id}
                css={{
                  backgroundColor: isSelected ? theme.colors.actionDefaultBackgroundPress : 'transparent',
                  cursor: 'pointer',
                }}
                onClick={() => onSelectVersion(version)}
              >
                {row.getAllCells().map((cell) => (
                  <TableCell key={cell.id} css={{ alignItems: 'center' }}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
                {isSelected && (
                  <div
                    css={{
                      width: theme.spacing.md * 2,
                      display: 'flex',
                      alignItems: 'center',
                      paddingRight: theme.spacing.sm,
                    }}
                  >
                    <ChevronRightIcon />
                  </div>
                )}
              </TableRow>
            );
          })
        )}
      </Table>
    </div>
  );
};
