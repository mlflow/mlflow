import { useMemo } from 'react';
import type { CursorPaginationProps } from '@databricks/design-system';
import {
  Checkbox,
  useDesignSystemTheme,
  Empty,
  NoIcon,
  Table,
  CursorPagination,
  TableRow,
  TableHeader,
  TableCell,
  TableSkeletonRows,
} from '@databricks/design-system';
import 'react-virtualized/styles.css';
import type { ExperimentEntity } from '../types';
import type { ColumnDef, OnChangeFn, RowSelectionState, SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { isEmpty } from 'lodash';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../common/utils/Utils';
import { Link } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import { ExperimentListTableTagsCell } from './ExperimentListTableTagsCell';

export type ExperimentTableColumnDef = ColumnDef<ExperimentEntity>;

export type ExperimentTableMetadata = { onEditTags: (editedEntity: ExperimentEntity) => void };

const useExperimentsTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const resultColumns: ExperimentTableColumnDef[] = [
      {
        header: ({ table }) => (
          <Checkbox
            componentId="mlflow.experiment_list_view.check_all_box"
            isChecked={table.getIsSomeRowsSelected() ? null : table.getIsAllRowsSelected()}
            onChange={(_, event) => table.getToggleAllRowsSelectedHandler()(event)}
          />
        ),
        id: 'select',
        cell: ExperimentListCheckbox,
        enableSorting: false,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Name',
          description: 'Header for the name column in the experiments table',
        }),
        accessorKey: 'name',
        id: 'name',
        cell: ExperimentListTableCell,
        enableSorting: true,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Time created',
          description: 'Header for the time created column in the experiments table',
        }),
        id: 'creation_time',
        accessorFn: ({ creationTime }) => Utils.formatTimestamp(creationTime, intl),
        enableSorting: true,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Last modified',
          description: 'Header for the last modified column in the experiments table',
        }),
        id: 'last_update_time',
        accessorFn: ({ lastUpdateTime }) => Utils.formatTimestamp(lastUpdateTime, intl),
        enableSorting: true,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Description',
          description: 'Header for the description column in the experiments table',
        }),
        id: 'description',
        accessorFn: ({ tags }) => tags?.find(({ key }) => key === 'mlflow.note.content')?.value ?? '-',
        enableSorting: false,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Tags',
          description: 'Header for the tags column in the experiments table',
        }),
        id: 'tags',
        accessorKey: 'tags',
        enableSorting: false,
        cell: ExperimentListTableTagsCell,
      },
    ];

    return resultColumns;
  }, [intl]);
};

export const ExperimentListTable = ({
  experiments,
  isFiltered,
  isLoading,
  rowSelection,
  setRowSelection,
  cursorPaginationProps,
  sortingProps: { sorting, setSorting },
  onEditTags,
}: {
  experiments?: ExperimentEntity[];
  isFiltered?: boolean;
  isLoading: boolean;
  rowSelection: RowSelectionState;
  setRowSelection: OnChangeFn<RowSelectionState>;
  cursorPaginationProps?: Omit<CursorPaginationProps, 'componentId'>;
  sortingProps: { sorting: SortingState; setSorting: OnChangeFn<SortingState> };
  onEditTags: (editedEntity: ExperimentEntity) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const columns = useExperimentsTableColumns();

  const table = useReactTable({
    data: experiments ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.experimentId,
    enableRowSelection: true,
    enableMultiRowSelection: true,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
    state: { rowSelection, sorting },
    meta: { onEditTags } satisfies ExperimentTableMetadata,
  });

  const getEmptyState = () => {
    const isEmptyList = !isLoading && isEmpty(experiments);
    if (isEmptyList && isFiltered) {
      return (
        <Empty
          image={<NoIcon />}
          title={
            <FormattedMessage
              defaultMessage="No experiments found"
              description="Label for the empty state in the experiments table when no experiments are found"
            />
          }
          description={null}
        />
      );
    }
    if (isEmptyList) {
      return (
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No experiments created"
              description="A header for the empty state in the experiments table"
            />
          }
          description={
            <FormattedMessage
              defaultMessage='Use "Create experiment" button in order to create a new experiment'
              description="Guidelines for the user on how to create a new experiment in the experiments list page"
            />
          }
        />
      );
    }

    return null;
  };

  const selectColumnStyles = { flex: 'none', height: theme.general.heightBase };

  return (
    <Table
      scrollable
      pagination={
        cursorPaginationProps ? (
          <CursorPagination {...cursorPaginationProps} componentId="mlflow.experiment_list_view.pagination" />
        ) : undefined
      }
      empty={getEmptyState()}
    >
      <TableRow isHeader>
        {table.getLeafHeaders().map((header) => (
          <TableHeader
            componentId="mlflow.experiment_list_view.table.header"
            key={header.id}
            css={header.column.id === 'select' ? selectColumnStyles : undefined}
            sortable={header.column.getCanSort()}
            sortDirection={header.column.getIsSorted() || 'none'}
            onToggleSort={header.column.getToggleSortingHandler()}
          >
            {flexRender(header.column.columnDef.header, header.getContext())}
          </TableHeader>
        ))}
      </TableRow>
      {isLoading ? (
        <TableSkeletonRows table={table} />
      ) : (
        table.getRowModel().rows.map((row) => (
          <TableRow key={row.id} css={{ height: theme.general.buttonHeight }} data-testid="experiment-list-item">
            {row.getAllCells().map((cell) => (
              <TableCell
                key={cell.id}
                css={{ alignItems: 'center', ...(cell.column.id === 'select' ? selectColumnStyles : undefined) }}
              >
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))
      )}
    </Table>
  );
};

const ExperimentListTableCell: ExperimentTableColumnDef['cell'] = ({ row: { original } }) => {
  return (
    <Link
      className="experiment-link"
      css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}
      to={Routes.getExperimentPageRoute(original.experimentId)}
      title={original.name}
      data-testid="experiment-list-item-link"
    >
      {original.name}
    </Link>
  );
};

const ExperimentListCheckbox: ExperimentTableColumnDef['cell'] = ({ row }) => {
  return (
    <Checkbox
      componentId="mlflow.experiment_list_view.check_box"
      id={row.original.experimentId}
      key={row.original.experimentId}
      data-testid="experiment-list-item-check-box"
      isChecked={row.getIsSelected()}
      disabled={!row.getCanSelect()}
      onChange={row.getToggleSelectedHandler()}
    />
  );
};
