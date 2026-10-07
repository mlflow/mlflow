import { useMemo, type ReactNode } from 'react';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import {
  ChevronRightIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { FormattedMessage } from 'react-intl';

export interface RegistryVersionCompareMode {
  comparedKey?: string;
  /** Controls that pick a row as the baseline or the compared version, replacing row selection. */
  renderControls: (key: string, state: { isSelected: boolean; isCompared: boolean }) => ReactNode;
}

/**
 * The version column of a registry detail page: one selectable row per version, with a chevron on the selected
 * one. Shared by the MCP and Skill registries, which render their own row content.
 */
export const RegistryVersionList = <V,>({
  versions,
  getVersionKey,
  renderVersion,
  selectedKey,
  onSelect,
  header,
  componentId,
  emptyState,
  isLoading,
  hasMoreVersions,
  compareMode,
}: {
  versions?: V[];
  getVersionKey: (version: V) => string;
  renderVersion: (version: V) => ReactNode;
  selectedKey?: string;
  onSelect: (key: string) => void;
  header: string;
  /** Prefix for the list's component IDs, e.g. `mlflow.mcp_registry.detail.versions`. */
  componentId: string;
  /** Rendered when loading has finished and there are no versions. */
  emptyState: ReactNode;
  isLoading?: boolean;
  /** The page shows a capped number of versions; this says some were left out. */
  hasMoreVersions?: boolean;
  compareMode?: RegistryVersionCompareMode;
}) => {
  const { theme } = useDesignSystemTheme();

  const columns = useMemo<ColumnDef<V>[]>(
    () => [{ id: 'version', header, cell: ({ row }) => renderVersion(row.original) }],
    [header, renderVersion],
  );

  const table = useReactTable('mlflow/server/js/src/common/components/RegistryVersionList.tsx', {
    data: versions ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: getVersionKey,
  });

  const isEmpty = !isLoading && !versions?.length;

  return (
    <div css={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Table scrollable empty={isEmpty ? emptyState : null}>
        <TableRow isHeader>
          {table.getLeafHeaders().map((leafHeader) => (
            <TableHeader componentId={`${componentId}.header`} key={leafHeader.id}>
              {flexRender(leafHeader.column.columnDef.header, leafHeader.getContext())}
            </TableHeader>
          ))}
        </TableRow>
        {isLoading ? (
          <TableSkeletonRows table={table} />
        ) : (
          table.getRowModel().rows.map((row) => {
            const key = row.id;
            const isSelected = selectedKey === key;
            const isCompared = compareMode?.comparedKey === key;
            const selectable = !compareMode;
            return (
              <TableRow
                key={key}
                tabIndex={selectable ? 0 : undefined}
                aria-selected={isSelected}
                css={{
                  backgroundColor: !selectable
                    ? isSelected || isCompared
                      ? theme.colors.actionDefaultBackgroundHover
                      : 'transparent'
                    : isSelected
                      ? theme.colors.actionDefaultBackgroundPress
                      : 'transparent',
                  cursor: selectable ? 'pointer' : 'default',
                }}
                onClick={selectable ? () => onSelect(key) : undefined}
                onKeyDown={
                  selectable
                    ? (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          onSelect(key);
                        }
                      }
                    : undefined
                }
              >
                {row.getAllCells().map((cell) => (
                  <TableCell key={cell.id} css={{ alignItems: 'center' }}>
                    <div
                      css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      {selectable && isSelected && (
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
                    </div>
                  </TableCell>
                ))}
                {compareMode?.renderControls(key, { isSelected, isCompared })}
              </TableRow>
            );
          })
        )}
      </Table>
      {hasMoreVersions && (
        <Typography.Hint css={{ padding: theme.spacing.sm, textAlign: 'center' }}>
          <FormattedMessage
            defaultMessage="Only the most recent 100 versions are shown."
            description="Hint when a registry entity has more versions than its detail page lists"
          />
        </Typography.Hint>
      )}
    </div>
  );
};
