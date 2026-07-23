import { useMemo } from 'react';
import {
  selectedRowIndicatorStyles,
  textEllipsisStyles,
  flexColumnGapStyles,
  flexRowWrapStyles,
  spaceBetweenRowStyles,
} from '../styles';
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
import { MCPServerDetailViewMode } from '../types';
import { STATUS_TAG_COLOR } from '../utils';
import { MCPServerVersionDiffSelectorButton } from './MCPServerVersionDiffSelectorButton';
import { MCPServerAliasesCell } from './MCPServerAliasesCell';
import Utils from '../../common/utils/Utils';

interface MCPServerVersionListMeta {
  serverDisplayName: string;
  aliasesByVersion: Record<string, string[]>;
}

const MCPServerVersionCell: ColumnDef<MCPServerVersion>['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { serverDisplayName, aliasesByVersion } = meta as MCPServerVersionListMeta;
  const aliases = aliasesByVersion[original.version] || [];

  const rawDisplayName = original.display_name || original.server_json?.title;
  const versionDisplayName = rawDisplayName && rawDisplayName !== serverDisplayName ? rawDisplayName : undefined;

  return (
    <div css={flexColumnGapStyles(theme)}>
      <div css={flexRowWrapStyles(theme)}>
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
        <MCPServerAliasesCell aliases={aliases} />
      </div>
      {versionDisplayName && (
        <Typography.Text size="sm" color="secondary" css={textEllipsisStyles} title={versionDisplayName}>
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
  comparedVersion,
  mode = MCPServerDetailViewMode.PREVIEW,
  onSelectVersion,
  onSelectComparedVersion,
  isLoading,
  serverDisplayName,
  aliasesByVersion,
  hasMoreVersions,
}: {
  versions?: MCPServerVersion[];
  selectedVersion?: string;
  comparedVersion?: string;
  mode?: MCPServerDetailViewMode;
  onSelectVersion: (version: string) => void;
  onSelectComparedVersion?: (version: string) => void;
  isLoading?: boolean;
  serverDisplayName: string;
  aliasesByVersion: Record<string, string[]>;
  hasMoreVersions?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isCompareMode = mode === MCPServerDetailViewMode.COMPARE;

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
    meta: { serverDisplayName, aliasesByVersion },
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
            const isCompared = comparedVersion === version;
            return (
              <TableRow
                key={row.id}
                tabIndex={isCompareMode ? undefined : 0}
                aria-selected={isSelected}
                css={{
                  backgroundColor:
                    isCompareMode && (isSelected || isCompared)
                      ? theme.colors.actionDefaultBackgroundHover
                      : !isCompareMode && isSelected
                        ? theme.colors.actionDefaultBackgroundPress
                        : 'transparent',
                  cursor: isCompareMode ? 'default' : 'pointer',
                }}
                onClick={isCompareMode ? undefined : () => onSelectVersion(version)}
                onKeyDown={
                  isCompareMode
                    ? undefined
                    : (e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          onSelectVersion(version);
                        }
                      }
                }
              >
                {row.getAllCells().map((cell) => (
                  <TableCell key={cell.id} css={{ alignItems: 'center' }}>
                    <div css={spaceBetweenRowStyles}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      {!isCompareMode && isSelected && (
                        <div css={selectedRowIndicatorStyles(theme)}>
                          <ChevronRightIcon />
                        </div>
                      )}
                    </div>
                  </TableCell>
                ))}
                {isCompareMode && (
                  <MCPServerVersionDiffSelectorButton
                    isSelectedBaseline={isSelected}
                    isSelectedCompared={isCompared}
                    onSelectBaseline={() => onSelectVersion(version)}
                    onSelectCompared={() => onSelectComparedVersion?.(version)}
                  />
                )}
              </TableRow>
            );
          })
        )}
      </Table>
      {hasMoreVersions && (
        <Typography.Hint css={{ padding: theme.spacing.sm, textAlign: 'center' }}>
          <FormattedMessage
            defaultMessage="Only the most recent 100 versions are shown."
            description="Warning shown when the MCP server has more versions than can be displayed"
          />
        </Typography.Hint>
      )}
    </div>
  );
};
