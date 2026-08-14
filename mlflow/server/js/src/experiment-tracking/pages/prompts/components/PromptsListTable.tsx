import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import {
  CursorPagination,
  Empty,
  NoIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { RegisteredPrompt } from '../types';
import { PromptsListTableTagsCell } from './PromptsListTableTagsCell';
import { PromptsListTableNameCell } from './PromptsListTableNameCell';
import Utils from '../../../../common/utils/Utils';
import { PromptsListTableVersionCell } from './PromptsListTableVersionCell';
import { PromptsListEmptyState } from './PromptsListEmptyState';
import type { PromptsTableMetadata } from '../utils';
import { first, isEmpty } from 'lodash';

type PromptsTableColumnDef = ColumnDef<RegisteredPrompt>;

const usePromptsTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const resultColumns: PromptsTableColumnDef[] = [
      {
        header: intl.formatMessage({
          defaultMessage: 'Name',
          description: 'Header for the name column in the registered prompts table',
        }),
        accessorKey: 'name',
        id: 'name',
        cell: PromptsListTableNameCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Latest version',
          description: 'Header for the latest version column in the registered prompts table',
        }),
        cell: PromptsListTableVersionCell,
        accessorFn: ({ latest_versions }) => first(latest_versions)?.version,
        id: 'latestVersion',
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Last modified',
          description: 'Header for the last modified column in the registered prompts table',
        }),
        id: 'lastModified',
        accessorFn: ({ last_updated_timestamp }) => Utils.formatTimestamp(last_updated_timestamp, intl),
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Tags',
          description: 'Header for the tags column in the registered prompts table',
        }),
        accessorKey: 'tags',
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
  isFiltered,
  onNextPage,
  onPreviousPage,
  onEditTags,
  experimentId,
  onCreatePrompt,
  componentId,
}: {
  prompts?: RegisteredPrompt[];
  error?: Error;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  isLoading?: boolean;
  isFiltered?: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  onEditTags: (editedEntity: RegisteredPrompt) => void;
  experimentId?: string;
  onCreatePrompt: () => void;
  componentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const columns = usePromptsTableColumns();

  // prettier-ignore
  const table = useReactTable('mlflow/server/js/src/experiment-tracking/pages/prompts/components/PromptsListTable.tsx', {
    data: prompts ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row, index) => row.name ?? index.toString(),
    meta: { onEditTags, experimentId } satisfies PromptsTableMetadata,
  });

  const getEmptyState = () => {
    const isEmptyList = !isLoading && isEmpty(prompts);
    if (isEmptyList && isFiltered) {
      return (
        <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
          <Empty
            image={<NoIcon />}
            title={
              <FormattedMessage
                defaultMessage="No prompts found"
                description="Label for the empty state in the prompts table when no prompts are found"
              />
            }
            description={null}
          />
        </div>
      );
    }
    if (isEmptyList) {
      return <PromptsListEmptyState onCreatePrompt={onCreatePrompt} />;
    }

    return null;
  };

  return (
    <Table
      scrollable
      pagination={
        <CursorPagination
          hasNextPage={hasNextPage}
          hasPreviousPage={hasPreviousPage}
          onNextPage={onNextPage}
          onPreviousPage={onPreviousPage}
          componentId={`${componentId}.pagination`}
        />
      }
      empty={getEmptyState()}
    >
      <TableRow isHeader>
        {table.getLeafHeaders().map((header) => (
          <TableHeader componentId={`${componentId}.table.header`} key={header.id}>
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
