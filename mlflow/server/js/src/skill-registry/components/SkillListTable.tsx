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
  Tag,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { CellContext, ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';

import type { Skill } from '../types';
import { SkillStatus } from '../types';
import SkillRegistryRoutes from '../routes';
import { SkillsEmptyState } from './SkillRegistryEmptyState';
import { SkillIcon } from './SkillIcon';
import { UseSkillButton } from './UseSkillButton';
import { flexRowStyles, noShrinkStyles, textEllipsisStyles } from '../styles';
import {
  formatSkillIdentity,
  formatSkillOrganization,
  formatSkillSourceLabel,
  isSkillDimmed,
  STATUS_TAG_COLOR,
} from '../utils';
import { Link } from '../../common/utils/RoutingUtils';
import Utils from '../../common/utils/Utils';

const coreRowModel = getCoreRowModel<Skill>();
const getRowId = (row: Skill) => formatSkillIdentity(row.name, row.organization);

const getSkillTableColumnFlex = (columnId: string) => {
  if (columnId === 'description') return 2;
  if (columnId === 'use') return '0 0 72px';
  return 1;
};

const SkillNameCell = ({ row }: CellContext<Skill, unknown>) => {
  const { theme } = useDesignSystemTheme();
  return (
    <span css={{ ...flexRowStyles(theme), minWidth: 0, width: '100%' }}>
      <SkillIcon icons={row.original.icons} name={row.original.name} />
      <Tooltip content={row.original.name} componentId="mlflow.skill_registry.table.name_tooltip">
        <span css={{ minWidth: 0, flex: 1, ...textEllipsisStyles }}>
          <Link
            componentId="mlflow.skill_registry.table.name_link"
            to={SkillRegistryRoutes.getSkillDetailRoute(row.original.name, row.original.organization)}
            css={{ ...textEllipsisStyles, display: 'block' }}
          >
            {row.original.name}
          </Link>
        </span>
      </Tooltip>
    </span>
  );
};

const SkillDescriptionCell = ({ getValue }: CellContext<Skill, unknown>) => {
  const value = getValue() as string | null | undefined;
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
    <Tooltip content={value} componentId="mlflow.skill_registry.table.description_tooltip">
      {content}
    </Tooltip>
  ) : (
    content
  );
};

const SkillStatusCell = ({ row }: CellContext<Skill, unknown>) => {
  const status = row.original.status;
  if (!status) return '—';
  return (
    <Tag componentId="mlflow.skill_registry.table.status" color={STATUS_TAG_COLOR[status]}>
      {status === SkillStatus.ACTIVE ? (
        <FormattedMessage defaultMessage="Active" description="Skill catalog status label for active" />
      ) : status === SkillStatus.DRAFT ? (
        <FormattedMessage defaultMessage="Draft" description="Skill catalog status label for draft" />
      ) : status === SkillStatus.DEPRECATED ? (
        <FormattedMessage defaultMessage="Deprecated" description="Skill catalog status label for deprecated" />
      ) : (
        status
      )}
    </Tag>
  );
};

const SkillSourceCell = ({ row }: CellContext<Skill, unknown>) => {
  const label = formatSkillSourceLabel(row.original.source_type);
  if (!label) return '—';
  return (
    <Tag componentId="mlflow.skill_registry.table.source" color="charcoal">
      {label}
    </Tag>
  );
};

const SkillUseCell = ({ row }: CellContext<Skill, unknown>) => <UseSkillButton skill={row.original} />;

const useSkillTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const columns: ColumnDef<Skill>[] = [
      {
        header: intl.formatMessage({
          defaultMessage: 'Name',
          description: 'Header for the name column in the Skill Registry table',
        }),
        id: 'name',
        cell: SkillNameCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Organization',
          description: 'Header for the organization column in the Skill Registry table',
        }),
        id: 'organization',
        accessorFn: (row) => formatSkillOrganization(row.organization) || '—',
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Description',
          description: 'Header for the description column in the Skill Registry table',
        }),
        accessorKey: 'description',
        id: 'description',
        cell: SkillDescriptionCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Latest version',
          description: 'Header for the latest version column in the Skill Registry table',
        }),
        id: 'latestVersion',
        accessorFn: (row) => (row.latest_version != null ? `v${row.latest_version}` : '—'),
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Status',
          description: 'Header for the derived parent status column in the Skill Registry table',
        }),
        id: 'status',
        cell: SkillStatusCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Source',
          description: 'Header for the latest-resolved source type column in the Skill Registry table',
        }),
        id: 'source',
        cell: SkillSourceCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Last modified',
          description: 'Header for the last modified column in the Skill Registry table',
        }),
        id: 'lastModified',
        accessorFn: ({ last_updated_timestamp }) =>
          last_updated_timestamp ? Utils.formatTimestamp(last_updated_timestamp, intl) : '—',
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Use',
          description: 'Header for the use-skill action column in the Skill Registry table',
        }),
        id: 'use',
        cell: SkillUseCell,
      },
    ];
    return columns;
  }, [intl]);
};

export const SkillListTable = ({
  skills,
  hasNextPage,
  hasPreviousPage,
  isLoading,
  isFiltered,
  onNextPage,
  onPreviousPage,
  pageSizeSelect,
}: {
  skills?: Skill[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  isLoading?: boolean;
  isFiltered?: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  pageSizeSelect?: CursorPaginationProps['pageSizeSelect'];
}) => {
  const { theme } = useDesignSystemTheme();
  const columns = useSkillTableColumns();

  const table = useReactTable('mlflow/server/js/src/skill-registry/components/SkillListTable.tsx', {
    data: skills ?? [],
    columns,
    getCoreRowModel: coreRowModel,
    getRowId,
  });

  const isEmptyList = !isLoading && (!skills || skills.length === 0);
  const emptyState = isEmptyList ? <SkillsEmptyState isFiltered={isFiltered} /> : null;

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
          componentId="mlflow.skill_registry.table.pagination"
        />
      }
      empty={emptyState}
    >
      <TableRow isHeader>
        {table.getLeafHeaders().map((header) => (
          <TableHeader
            componentId="mlflow.skill_registry.table.header"
            key={header.id}
            align="left"
            style={{ flex: getSkillTableColumnFlex(header.column.id) }}
          >
            {flexRender(header.column.columnDef.header, header.getContext())}
          </TableHeader>
        ))}
      </TableRow>
      {isLoading ? (
        <TableSkeletonRows table={table} />
      ) : (
        table.getRowModel().rows.map((row) => {
          const isDimmed = isSkillDimmed(row.original);
          return (
            <TableRow key={row.id} css={{ height: theme.general.buttonHeight }}>
              {row.getAllCells().map((cell) => (
                <TableCell
                  key={cell.id}
                  css={{
                    alignItems: 'center',
                    minWidth: 0,
                    overflow: 'hidden',
                    ...(cell.column.id === 'use' ? noShrinkStyles : {}),
                  }}
                  align="left"
                  style={{
                    flex: getSkillTableColumnFlex(cell.column.id),
                    opacity: isDimmed ? 0.5 : 1,
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
  );
};
