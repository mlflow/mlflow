import {
  CursorPagination,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ColumnDef, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { useMemo } from 'react';
import { useIntl } from 'react-intl';
import { RegisteredPrompt } from '../types';
import { PromptsListTableTagsCell } from './PromptsListTableTagsCell';
import { PromptsListTableNameCell } from './PromptsListTableNameCell';
import Utils from '../../../../common/utils/Utils';
import { PromptsListTableVersionCell } from './PromptsListTableVersionCell';
import { PromptsTableMetadata } from '../utils';
import { first } from 'lodash';

type PromptsTableColumnDef = ColumnDef<RegisteredPrompt>;

const usePromptsTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const resultColumns: PromptsTableColumnDef[] = [
      {
        header: intl.formatMessage({ defaultMessage: 'Name', description: 'TODO' }),
        accessorKey: 'name',
        id: 'name',
        cell: PromptsListTableNameCell,
      },
      {
        header: intl.formatMessage({ defaultMessage: 'Latest version', description: 'TODO' }),
        cell: PromptsListTableVersionCell,
        accessorFn: ({ latest_versions }) => first(latest_versions)?.version,
        id: 'latestVersion',
      },
      {
        header: intl.formatMessage({ defaultMessage: 'Created by', description: 'TODO' }),
        accessorKey: 'name',
        id: 'createdBy',
      },
      {
        header: intl.formatMessage({ defaultMessage: 'Last modified', description: 'TODO' }),
        id: 'lastModified',
        accessorFn: ({ last_updated_timestamp }) => Utils.formatTimestamp(last_updated_timestamp, intl),
      },
      {
        header: intl.formatMessage({ defaultMessage: 'Tags', description: 'TODO' }),
        accessorKey: 'name',
        id: 'tags',
        cell: PromptsListTableTagsCell,
      },
    ];

    return resultColumns;
  }, [intl]);
};

export const PromptsListTable = ({
  prompts,
  hasNextPage,
  hasPreviousPage,
  isLoading,
  onNextPage,
  onPreviousPage,
  onEditTags,
}: {
  prompts?: RegisteredPrompt[];
  error?: Error;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  isLoading?: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  onEditTags: (editedEntity: RegisteredPrompt) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const columns = usePromptsTableColumns();

  const table = useReactTable({
    data: prompts ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row, index) => row.name ?? index.toString(),
    meta: { onEditTags } satisfies PromptsTableMetadata,
  });
  return (
    <Table
      scrollable
      pagination={
        <CursorPagination
          hasNextPage={hasNextPage}
          hasPreviousPage={hasPreviousPage}
          onNextPage={onNextPage}
          onPreviousPage={onPreviousPage}
          componentId="TODO"
        />
      }
    >
      <TableRow isHeader>
        {table.getLeafHeaders().map((header) => (
          <TableHeader componentId="TODO" key={header.id}>
            {flexRender(header.column.columnDef.header, header.getContext())}
          </TableHeader>
        ))}
      </TableRow>
      {isLoading ? (
        <TableSkeletonRows table={table} />
      ) : (
        table.getRowModel().rows.map((row) => (
          <TableRow key={row.id} css={{ height: theme.general.buttonHeight }}>
            {row.getAllCells().map((cell) => (
              <TableCell key={cell.id} css={{ alignItems: 'center' }}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))
      )}
    </Table>
  );
};
