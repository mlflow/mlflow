import {
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  LegacyTooltip,
  Empty,
  PlusIcon,
  TableSkeletonRows,
  WarningIcon,
} from '@databricks/design-system';
import type { Interpolation, Theme } from '@emotion/react';
import type { ColumnDef, SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { ModelListTagsCell, ModelListVersionLinkCell } from './ModelTableCellRenderers';
import { RegisteringModelDocUrl } from '../../../common/constants';
import Utils from '../../../common/utils/Utils';
import type { ModelEntity, ModelVersionInfoEntity } from '../../../experiment-tracking/types';
import type { KeyValueEntity } from '../../../common/types';
import { Stages } from '../../constants';
import { ModelRegistryRoutes } from '../../routes';
import { CreateModelButton } from '../CreateModelButton';
import { ModelsTableAliasedVersionsCell } from '../aliases/ModelsTableAliasedVersionsCell';
import { useNextModelsUIContext } from '../../hooks/useNextModelsUI';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';

const getLatestVersionNumberByStage = (latestVersions: ModelVersionInfoEntity[], stage: string) => {
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
  TAGS = 'tags',
  ALIASED_VERSIONS = 'aliased_versions',
}

export interface ModelListTableProps {
  modelsData: ModelEntity[];
  pagination: React.ReactElement;
  orderByKey: string;
  orderByAsc: boolean;
  isLoading: boolean;
  error?: Error;
  isFiltered: boolean;
  onSortChange: (params: { orderByKey: string; orderByAsc: boolean }) => void;
}

type EnrichedModelEntity = ModelEntity;
type ModelsColumnDef = ColumnDef<EnrichedModelEntity> & {
  // Our experiments column definition houses style definitions in the metadata field
  meta?: { styles?: Interpolation<Theme> };
};

export const ModelListTable = ({
  modelsData,
  orderByAsc,
  orderByKey,
  onSortChange,
  isLoading,
  error,
  isFiltered,
  pagination,
}: ModelListTableProps) => {
  const intl = useIntl();

  const { usingNextModelsUI } = useNextModelsUIContext();

  const enrichedModelsData: EnrichedModelEntity[] = modelsData.map((model) => {
    return model;
  });

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
          <Link to={ModelRegistryRoutes.getModelPageRoute(String(getValue()))}>
            <LegacyTooltip title={getValue()}>{getValue()}</LegacyTooltip>
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
          const latestVersions = getValue() as ModelVersionInfoEntity[];
          const latestVersionNumber =
            (Boolean(latestVersions?.length) &&
              Math.max(...latestVersions.map((v) => parseInt(v.version, 10))).toString()) ||
            '';
          return <ModelListVersionLinkCell name={name} versionNumber={latestVersionNumber} />;
        },
        meta: { styles: { maxWidth: 120 } },
      },
    ];
    if (usingNextModelsUI) {
      // Display aliases column only when "new models UI" is flipped
      columns.push({
        id: ColumnKeys.ALIASED_VERSIONS,
        enableSorting: false,

        header: intl.formatMessage({
          defaultMessage: 'Aliased versions',
          description: 'Column title for aliased versions in the registered model page',
        }),
        cell: ({ row: { original: modelEntity } }) => {
          return <ModelsTableAliasedVersionsCell model={modelEntity} />;
        },
        meta: { styles: { minWidth: 150 } },
      });
    } else {
      // If not, display legacy "Stage" columns
      columns.push(
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
      );
    }

    columns.push(
      {
        id: ColumnKeys.CREATED_BY,
        header: intl.formatMessage({
          defaultMessage: 'Created by',
          description: 'Column title for created by column for a model in the registered model page',
        }),
        accessorKey: 'user_id',
        enableSorting: false,
        cell: ({ getValue, row: { original } }) => {
          return <span title={getValue() as string}>{getValue()}</span>;
        },
        meta: { styles: { flex: 1 } },
      },
      {
        id: ColumnKeys.LAST_MODIFIED,
        enableSorting: true,
        header: intl.formatMessage({
          defaultMessage: 'Last modified',
          description: 'Column title for last modified timestamp for a model in the registered model page',
        }),
        accessorKey: 'last_updated_timestamp',
        cell: ({ getValue }) => <span>{Utils.formatTimestamp(getValue(), intl)}</span>,
        meta: { styles: { flex: 1, maxWidth: 150 } },
      },
      {
        id: ColumnKeys.TAGS,
        header: intl.formatMessage({
          defaultMessage: 'Tags',
          description: 'Column title for model tags in the registered model page',
        }),
        enableSorting: false,
        accessorKey: 'tags',
        cell: ({ getValue }) => {
          return <ModelListTagsCell tags={getValue() as KeyValueEntity[]} />;
        },
      },
    );

    return columns;
  }, [intl, usingNextModelsUI]);

  const sorting: SortingState = [{ id: orderByKey, desc: !orderByAsc }];

  const setSorting = (stateUpdater: SortingState | ((state: SortingState) => SortingState)) => {
    const [newSortState] = typeof stateUpdater === 'function' ? stateUpdater(sorting) : stateUpdater;
    if (newSortState) {
      onSortChange({ orderByKey: newSortState.id, orderByAsc: !newSortState.desc });
    }
  };

  // eslint-disable-next-line prefer-const
  let registerModelDocUrl = RegisteringModelDocUrl;

  const noResultsDescription = (() => {
    return (
      <FormattedMessage
        defaultMessage="No results. Try using a different keyword or adjusting your filters."
        description="Models table > no results after filtering"
      />
    );
  })();
  const emptyComponent = error ? (
    <Empty
      image={<WarningIcon />}
      description={error instanceof ErrorWrapper ? error.getMessageField() : error.message}
      title={
        <FormattedMessage
          defaultMessage="Error fetching models"
          description="Workspace models page > Error empty state title"
        />
      }
    />
  ) : isFiltered ? (
    // Displayed when there is no results, but any filters have been applied
    <Empty description={noResultsDescription} image={<SearchIcon />} data-testid="model-list-no-results" />
  ) : (
    // Displayed when there is no results with no filters applied
    <Empty
      description={
        <FormattedMessage
          defaultMessage="No models registered yet. <link>Learn more about registering models</link>."
          description="Models table > no models present yet"
          values={{
            link: (content: any) => (
              <a target="_blank" rel="noopener noreferrer" href={registerModelDocUrl}>
                {content}
              </a>
            ),
          }}
        />
      }
      image={<PlusIcon />}
      button={
        <CreateModelButton
          buttonType="primary"
          buttonText={
            <FormattedMessage defaultMessage="Create a model" description="Create button to register a new model" />
          }
        />
      }
    />
  );

  const isEmpty = () => (!isLoading && table.getRowModel().rows.length === 0) || error;

  const table = useReactTable<EnrichedModelEntity>({
    data: enrichedModelsData,
    columns: tableColumns,
    state: {
      sorting,
    },
    getCoreRowModel: getCoreRowModel(),
    getRowId: ({ id }) => id,
    onSortingChange: setSorting,
  });

  return (
    <>
      <Table
        data-testid="model-list-table"
        pagination={pagination}
        scrollable
        empty={isEmpty() ? emptyComponent : undefined}
      >
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader
              componentId="codegen_mlflow_app_src_model-registry_components_model-list_modellisttable.tsx_412"
              ellipsis
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
        {isLoading ? (
          <TableSkeletonRows table={table} />
        ) : (
          table.getRowModel().rows.map((row) => (
            <TableRow key={row.id}>
              {row.getAllCells().map((cell) => (
                <TableCell ellipsis key={cell.id} css={(cell.column.columnDef as ModelsColumnDef).meta?.styles}>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))
        )}
      </Table>
    </>
  );
};
