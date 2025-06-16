import { useMemo } from 'react';
import {
  Checkbox,
  useDesignSystemTheme,
  Empty,
  NoIcon,
  Table,
  CursorPagination,
  TableRow,
  TableHeader,
  TableSkeletonRows,
  TableCell,
} from '@databricks/design-system';
import 'react-virtualized/styles.css';
import { ExperimentEntity } from '../types';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  OnChangeFn,
  RowSelectionState,
  useReactTable,
} from '@tanstack/react-table';
import { isEmpty } from 'lodash';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../common/utils/Utils';
import { Link } from '../../common/utils/RoutingUtils';
import Routes from '../routes';

type ExperimentTableColumnDef = ColumnDef<ExperimentEntity>;

const useExperimentsTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const resultColumns: ExperimentTableColumnDef[] = [
      {
        header: ({ table }) => (
          <Checkbox
            componentId="mlflow.experiment_list_view.check_all_box"
            isChecked={table.getIsAllRowsSelected()}
            onChange={(_, event) => table.getToggleAllRowsSelectedHandler()(event)}
            // indeterminate={table.getIsSomeRowsSelected()}
          />
        ),
        id: 'select',
        cell: ExperimentListCheckbox,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Name',
          description: 'Header for the name column in the experiments table',
        }),
        accessorKey: 'name',
        id: 'name',
        cell: ExperimentListTableCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Time created',
          description: 'Header for the time created column in the experiments table',
        }),
        id: 'timeCreated',
        accessorFn: ({ creationTime }) => Utils.formatTimestamp(creationTime, intl),
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Last modified',
          description: 'Header for the last modified column in the experiments table',
        }),
        id: 'lastModified',
        accessorFn: ({ lastUpdateTime }) => Utils.formatTimestamp(lastUpdateTime, intl),
      },
    ];

    return resultColumns;
  }, [intl]);
};

export const ExperimentListTable = ({
  experiments,
  hasNextPage,
  hasPreviousPage,
  isLoading,
  isFiltered,
  onNextPage,
  onPreviousPage,
  rowSelection,
  setRowSelection,
}: {
  experiments?: ExperimentEntity[];
  error?: Error;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  isLoading?: boolean;
  isFiltered?: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  rowSelection: RowSelectionState;
  setRowSelection: OnChangeFn<RowSelectionState>;
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
    state: { rowSelection },
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

  const selectColumnStyles = { flex: 'none' };

  return (
    <Table
      scrollable
      pagination={
        <CursorPagination
          hasNextPage={hasNextPage}
          hasPreviousPage={hasPreviousPage}
          onNextPage={onNextPage}
          onPreviousPage={onPreviousPage}
          componentId="mlflow.experiment_list_view.pagination"
        />
      }
      empty={getEmptyState()}
    >
      <TableRow isHeader>
        {table.getLeafHeaders().map((header) => (
          <TableHeader
            componentId="mlflow.experiment_list_view.table.header"
            key={header.id}
            css={header.column.id === 'select' ? selectColumnStyles : undefined}
          >
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

const ExperimentListTableCell: ColumnDef<ExperimentEntity>['cell'] = ({ row: { original } }) => {
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

const ExperimentListCheckbox: ColumnDef<ExperimentEntity>['cell'] = ({ row }) => {
  return (
    <Checkbox
      componentId="mlflow.experiment_list_view.check_box"
      id={row.original.experimentId}
      key={row.original.experimentId}
      data-testid={`experiment-list-item-check-box`}
      isChecked={row.getIsSelected()}
      disabled={!row.getCanSelect()}
      onChange={row.getToggleSelectedHandler()}
    />
  );
};
