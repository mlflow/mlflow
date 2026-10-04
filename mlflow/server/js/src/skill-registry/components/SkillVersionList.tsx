import { useMemo } from 'react';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import {
  ChevronRightIcon,
  Empty,
  Table,
  Tooltip,
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

import { SkillStatus, type SkillVersion } from '../types';
import { formatSkillStatusLabel, STATUS_TAG_COLOR } from '../utils';
import { flexColumnGapStyles, flexRowWrapStyles, selectedRowIndicatorStyles, spaceBetweenRowStyles } from '../styles';
import Utils from '../../common/utils/Utils';

const SkillVersionCell: ColumnDef<SkillVersion>['cell'] = ({ row: { original } }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  if (original.status === SkillStatus.DELETED) {
    return (
      <div css={flexColumnGapStyles(theme)}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Version {version}"
            description="Skill version list item label"
            values={{ version: original.version }}
          />
        </Typography.Text>
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="Deleted, number not reused"
            description="Subtitle for a deleted skill version in the version list"
          />
        </Typography.Text>
      </div>
    );
  }

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
          {formatSkillStatusLabel(original.status)}
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
  hasMoreVersions,
}: {
  versions?: SkillVersion[];
  selectedVersion?: number;
  onSelectVersion: (version: number) => void;
  isLoading?: boolean;
  hasMoreVersions?: boolean;
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
            const isDeleted = row.original.status === SkillStatus.DELETED;
            const isSelected = !isDeleted && selectedVersion === version;
            const content = (
              <div css={spaceBetweenRowStyles}>
                {row.getAllCells().map((cell) => (
                  <span key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</span>
                ))}
                {isSelected && (
                  <div css={selectedRowIndicatorStyles(theme)}>
                    <ChevronRightIcon />
                  </div>
                )}
              </div>
            );
            return (
              <TableRow
                key={row.id}
                tabIndex={isDeleted ? -1 : 0}
                aria-selected={isSelected}
                aria-disabled={isDeleted}
                css={{
                  backgroundColor: isSelected ? theme.colors.actionDefaultBackgroundPress : 'transparent',
                  cursor: isDeleted ? 'default' : 'pointer',
                }}
                onClick={isDeleted ? undefined : () => onSelectVersion(version)}
                onKeyDown={
                  isDeleted
                    ? undefined
                    : (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          onSelectVersion(version);
                        }
                      }
                }
              >
                <TableCell css={{ alignItems: 'center' }}>
                  {isDeleted ? (
                    <Tooltip
                      componentId="mlflow.skill_registry.detail.version_deleted_tooltip"
                      content={
                        <FormattedMessage
                          defaultMessage="Deleted. The number is never reused, and the registry no longer returns this version."
                          description="Tooltip for a deleted skill version row"
                        />
                      }
                    >
                      <span css={{ display: 'block' }}>{content}</span>
                    </Tooltip>
                  ) : (
                    content
                  )}
                </TableCell>
              </TableRow>
            );
          })
        )}
      </Table>
      {hasMoreVersions && (
        <Typography.Hint css={{ padding: theme.spacing.sm, textAlign: 'center' }}>
          <FormattedMessage
            defaultMessage="Only the most recent 100 versions are shown."
            description="Warning shown when a skill has more versions than the detail page displays"
          />
        </Typography.Hint>
      )}
    </div>
  );
};
