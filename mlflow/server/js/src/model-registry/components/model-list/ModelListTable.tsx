import {
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeleton,
  Tooltip,
  Empty,
  PlusIcon,
} from '@databricks/design-system';
import { Interpolation, Theme } from '@emotion/react';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  SortingState,
  useReactTable,
} from '@tanstack/react-table';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from 'react-router-dom';

import Utils from '../../../common/utils/Utils';
import type { ModelEntity, ModelInfoEntity } from '../../../experiment-tracking/types';
import { Stages } from '../../constants';
import { getModelPageRoute } from '../../routes';
import { ModelListVersionLinkCell } from './ModelTableCellRenderers';
import { CreateModelButton } from '../CreateModelButton';

const getLatestVersionNumberByStage = (latestVersions: ModelInfoEntity[], stage: string) => {
  const modelVersion = latestVersions && latestVersions.find((v) => v.current_stage === stage);
  return modelVersion && modelVersion.version;
};

enum ColumnKeys {
  NAME = 'name',
  LATEST_VERSION = 'latest_versions',
  LAST_MODIFIED = 'timestamp',
  CREATED_BY = 'user_id',
  STAGE_STAGING = 'stage_staging',
  STAGE_PRODUCTION = 'stage_production',
}

export interface ModelListTableProps {
  modelsData: ModelEntity[];
  pagination: React.ReactElement;
  orderByKey: string;
  orderByAsc: boolean;
  isLoading: boolean;
  isFiltered: boolean;
  onSortChange: (params: { orderByKey: string; orderByAsc: boolean }) => void;
}

type ModelsColumnDef = ColumnDef<ModelEntity> & {
  // Our experiments column definition houses style definitions in the metadata field
  meta?: { styles?: Interpolation<Theme> };
};

export const ModelListTable = ({
  modelsData,
  orderByAsc,
  orderByKey,
  onSortChange,
  isLoading,
  isFiltered,
  pagination,
}: ModelListTableProps) => {
  const intl = useIntl();
  const tableColumns = useMemo(() => {
    const columns: ModelsColumnDef[] = [
      {
        id: ColumnKeys.NAME,
        enableSorting: true,
        header: intl.formatMessage({
          defaultMessage: 'Name',
          description: 'Column title for model name in the registered model page',
        }),
        accessorKey: 'name',
        cell: ({ getValue }) => (
          <Link to={getModelPageRoute(getValue())}>
            <Tooltip title={getValue()}>{getValue()}</Tooltip>
          </Link>
        ),
        meta: { styles: { minWidth: 200, flex: 1 } },
      },
      {
        id: ColumnKeys.LATEST_VERSION,
        enableSorting: false,

        header: intl.formatMessage({
          defaultMessage: 'Latest version',
          description: 'Column title for latest model version in the registered model page',
        }),
        accessorKey: 'latest_versions',
        cell: ({ getValue, row: { original } }) => {
          const { name } = original;
          const latestVersions = getValue() as ModelInfoEntity[];
          const latestVersionNumber =
            (Boolean(latestVersions?.length) &&
              Math.max(...latestVersions.map((v) => parseInt(v.version, 10))).toString()) ||
            '';
          return <ModelListVersionLinkCell name={name} versionNumber={latestVersionNumber} />;
        },
        meta: { styles: { maxWidth: 120 } },
      },

      {
        id: ColumnKeys.STAGE_STAGING,
        enableSorting: false,

        header: intl.formatMessage({
          defaultMessage: 'Staging',
          description: 'Column title for staging phase version in the registered model page',
        }),
        cell: ({ row: { original } }) => {
          const { latest_versions, name } = original;
          const versionNumber = getLatestVersionNumberByStage(latest_versions, Stages.STAGING);
          return <ModelListVersionLinkCell name={name} versionNumber={versionNumber} />;
        },
        meta: { styles: { maxWidth: 120 } },
      },
      {
        id: ColumnKeys.STAGE_PRODUCTION,
        enableSorting: false,

        header: intl.formatMessage({
          defaultMessage: 'Production',
          description: 'Column title for production phase version in the registered model page',
        }),
        cell: ({ row: { original } }) => {
          const { latest_versions, name } = original;
          const versionNumber = getLatestVersionNumberByStage(latest_versions, Stages.PRODUCTION);
          return <ModelListVersionLinkCell name={name} versionNumber={versionNumber} />;
        },
        meta: { styles: { maxWidth: 120 } },
      },

      {
        id: ColumnKeys.CREATED_BY,
        header: intl.formatMessage({
          defaultMessage: 'Created by',
          description:
            'Column title for created by column for a model in the registered model page',
        }),
        accessorKey: 'user_id',
        enableSorting: false,
        cell: ({ getValue }) => <span title={getValue() as string}>{getValue()}</span>,
        meta: { styles: { flex: 1 } },
      },
      {
        id: ColumnKeys.LAST_MODIFIED,
        enableSorting: true,
        header: intl.formatMessage({
          defaultMessage: 'Last modified',
          description:
            'Column title for last modified timestamp for a model in the registered model page',
        }),
        accessorKey: 'last_updated_timestamp',
        cell: ({ getValue }) => <span>{Utils.formatTimestamp(getValue())}</span>,
        meta: { styles: { flex: 1, maxWidth: 150 } },
      },
    ];

    return columns;
  }, [
    // prettier-ignore
    intl,
  ]);

  const sorting: SortingState = [{ id: orderByKey, desc: !orderByAsc }];

  const setSorting = (stateUpdater: SortingState | ((state: SortingState) => SortingState)) => {
    const [newSortState] =
      typeof stateUpdater === 'function' ? stateUpdater(sorting) : stateUpdater;
    if (newSortState) {
      onSortChange({ orderByKey: newSortState.id, orderByAsc: !newSortState.desc });
    }
  };

  const emptyDescription = (
    <FormattedMessage
      defaultMessage='No models yet. Use the button below to create your first model.'
      description='Models table > no models present yet'
    />
  );
  const noResultsDescription = (
    <FormattedMessage
      defaultMessage='No results. Try using a different keyword or adjusting your filters.'
      description='Models table > no results after filtering'
    />
  );
  const emptyComponent = isFiltered ? (
    // Displayed when there is no results, but any filters have been applied
    <Empty description={noResultsDescription} image={<SearchIcon />} />
  ) : (
    // Displayed when there is no results with no filters applied
    <Empty
      description={emptyDescription}
      image={<PlusIcon />}
      button={
        <CreateModelButton
          buttonType='primary'
          buttonText={<FormattedMessage defaultMessage='Create a model' description='' />}
        />
      }
    />
  );

  const isEmpty = () => !isLoading && table.getRowModel().rows.length === 0;

  const table = useReactTable<ModelEntity>({
    data: modelsData,
    columns: tableColumns,
    state: {
      sorting,
    },
    getCoreRowModel: getCoreRowModel(),
    getRowId: ({ id }) => id,
    onSortingChange: setSorting,
  });

  // Three skeleton rows for the loading state
  const loadingState = (
    <>
      {new Array(3).fill(0).map((_, rowIndex) => (
        <TableRow key={rowIndex}>
          {table.getAllColumns().map((column, columnIndex) => (
            <TableCell key={columnIndex} css={(column.columnDef as ModelsColumnDef).meta?.styles}>
              <TableSkeleton seed={`${rowIndex}-${columnIndex}`} />
            </TableCell>
          ))}
        </TableRow>
      ))}
    </>
  );

  return (
    <>
      <Table pagination={pagination} scrollable empty={isEmpty() ? emptyComponent : undefined}>
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader
              key={header.id}
              sortable={header.column.getCanSort()}
              sortDirection={header.column.getIsSorted() || 'none'}
              onToggleSort={() => {
                const [currentSortColumn] = sorting;
                const changingDirection = header.column.id === currentSortColumn.id;
                const sortDesc = changingDirection ? !currentSortColumn.desc : false;
                header.column.toggleSorting(sortDesc);
              }}
              css={(header.column.columnDef as ModelsColumnDef).meta?.styles}
            >
              {flexRender(header.column.columnDef.header, header.getContext())}
            </TableHeader>
          ))}
        </TableRow>
        {isLoading
          ? loadingState
          : table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getAllCells().map((cell) => (
                  <TableCell
                    ellipsis
                    key={cell.id}
                    css={(cell.column.columnDef as ModelsColumnDef).meta?.styles}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
      </Table>
    </>
  );
};
