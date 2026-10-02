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

import type { SkillVersion } from '../types';
import { STATUS_TAG_COLOR } from '../utils';
import { flexColumnGapStyles, flexRowWrapStyles, selectedRowIndicatorStyles, spaceBetweenRowStyles } from '../styles';
import Utils from '../../common/utils/Utils';

const SkillVersionCell: ColumnDef<SkillVersion>['cell'] = ({ row: { original } }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={flexColumnGapStyles(theme)}>
      <div css={flexRowWrapStyles(theme)}>
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Version {version}"
            description="Skill version list item label"
            values={{ version: original.version }}
          />
        </Typography.Text>
        <Tag componentId="mlflow.skill_registry.detail.version_status_tag" color={STATUS_TAG_COLOR[original.status]}>
          {original.status}
        </Tag>
      </div>
      {original.creation_timestamp && (
        <Typography.Text size="sm" color="secondary">
          {Utils.formatTimestamp(original.creation_timestamp, intl)}
        </Typography.Text>
      )}
    </div>
  );
};

export const SkillVersionList = ({
  versions,
  selectedVersion,
  onSelectVersion,
  isLoading,
}: {
  versions?: SkillVersion[];
  selectedVersion?: number;
  onSelectVersion: (version: number) => void;
  isLoading?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const columns = useMemo<ColumnDef<SkillVersion>[]>(
    () => [
      {
        id: 'version',
        header: intl.formatMessage({
          defaultMessage: 'Versions',
          description: 'Header for the version column in the Skill versions table',
        }),
        accessorKey: 'version',
        cell: SkillVersionCell,
      },
    ],
    [intl],
  );

  const table = useReactTable('mlflow/server/js/src/skill-registry/components/SkillVersionList.tsx', {
    data: versions ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => String(row.version),
  });

  const emptyState =
    !isLoading && (!versions || versions.length === 0) ? (
      <Empty
        title={<FormattedMessage defaultMessage="No versions" description="Empty state when a Skill has no versions" />}
        description={
          <FormattedMessage
            defaultMessage="This skill does not have any versions yet."
            description="Description for an empty Skill version list"
          />
        }
      />
    ) : null;

  return (
    <div css={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Table scrollable empty={emptyState}>
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader componentId="mlflow.skill_registry.detail.versions.header" key={header.id}>
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
                tabIndex={0}
                aria-selected={isSelected}
                css={{
                  backgroundColor: isSelected ? theme.colors.actionDefaultBackgroundPress : 'transparent',
                  cursor: 'pointer',
                }}
                onClick={() => onSelectVersion(version)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelectVersion(version);
                  }
                }}
              >
                {row.getAllCells().map((cell) => (
                  <TableCell key={cell.id} css={{ alignItems: 'center' }}>
                    <div css={spaceBetweenRowStyles}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      {isSelected && (
                        <div css={selectedRowIndicatorStyles(theme)}>
                          <ChevronRightIcon />
                        </div>
                      )}
                    </div>
                  </TableCell>
                ))}
              </TableRow>
            );
          })
        )}
      </Table>
    </div>
  );
};
