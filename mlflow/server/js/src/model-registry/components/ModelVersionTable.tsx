import {
  Empty,
  PlusIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  LegacyTooltip,
  Typography,
  useDesignSystemTheme,
  TableSkeletonRows,
} from '@databricks/design-system';
import type { ColumnDef, RowSelectionState, SortingState, ColumnSort } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from '@tanstack/react-table';
import type { ModelEntity, ModelVersionInfoEntity, ModelAliasMap } from '../../experiment-tracking/types';
import type { KeyValueEntity } from '../../common/types';
import { useEffect, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { RegisteringModelDocUrl } from '../../common/constants';
import {
  ACTIVE_STAGES,
  ModelVersionStatusIcons,
  StageTagComponents,
  mlflowAliasesLearnMoreLink,
  modelVersionStatusIconTooltips,
} from '../constants';
import { Link } from '../../common/utils/RoutingUtils';
import { ModelRegistryRoutes } from '../routes';
import Utils from '../../common/utils/Utils';
import { KeyValueTagsEditorCell } from '../../common/components/KeyValueTagsEditorCell';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '../../redux-types';
import { useEditKeyValueTagsModal } from '../../common/hooks/useEditKeyValueTagsModal';
import { useEditAliasesModal } from '../../common/hooks/useEditAliasesModal';
import { updateModelVersionTagsApi } from '../actions';
import { ModelVersionTableAliasesCell } from './aliases/ModelVersionTableAliasesCell';
import type { Interpolation, Theme } from '@emotion/react';
import { truncateToFirstLineWithMaxLength } from '../../common/utils/StringUtils';
import { setModelVersionAliasesApi } from '../actions';

type ModelVersionTableProps = {
  isLoading: boolean;
  modelName: string;
  pagination: React.ReactElement;
  orderByKey: string;
  orderByAsc: boolean;
  modelVersions?: ModelVersionInfoEntity[];
  activeStageOnly?: boolean;
  onChange: (selectedRowKeys: string[], selectedRows: ModelVersionInfoEntity[]) => void;
  getSortFieldName: (columnId: string) => string | null;
  onSortChange: (params: { sorter: ColumnSort }) => void;
  modelEntity?: ModelEntity;
  onMetadataUpdated: () => void;
  usingNextModelsUI: boolean;
  aliases?: ModelAliasMap;
};

type ModelVersionColumnDef = ColumnDef<ModelVersionInfoEntity> & {
  meta?: { styles?: Interpolation<Theme>; multiline?: boolean; className?: string };
};

enum COLUMN_IDS {
  STATUS = 'status',
  VERSION = 'version',
  CREATION_TIMESTAMP = 'creation_timestamp',
  USER_ID = 'user_id',
  TAGS = 'tags',
  STAGE = 'current_stage',
  DESCRIPTION = 'description',
  ALIASES = 'aliases',
}

const getAliasesModalTitle = (version: string) => (
  <FormattedMessage
    defaultMessage="Add/Edit alias for model version {version}"
    description="Model registry > model version alias editor > Title of the update alias modal"
    values={{ version }}
  />
);

export const ModelVersionTable = ({
  modelName,
  modelVersions,
  activeStageOnly,
  orderByAsc,
  orderByKey,
  onSortChange,
  onChange,
  getSortFieldName,
  modelEntity,
  onMetadataUpdated,
  usingNextModelsUI,
  aliases,
  pagination,
  isLoading,
}: ModelVersionTableProps) => {
  const aliasesByVersion = useMemo(() => {
    const result: Record<string, string[]> = {};
    aliases?.forEach(({ alias, version }) => {
      if (!result[version]) {
        result[version] = [];
      }
      result[version].push(alias);
    });
    return result;
  }, [aliases]);
  const versions = useMemo(
    () =>
      activeStageOnly
        ? (modelVersions || []).filter(({ current_stage }) => ACTIVE_STAGES.includes(current_stage))
        : modelVersions,
    [activeStageOnly, modelVersions],
  );

  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const allTagsKeys = useMemo(() => {
    const allTagsList: KeyValueEntity[] = versions?.map((modelVersion) => modelVersion?.tags || []).flat() || [];

    // Extract keys, remove duplicates and sort the
    return Array.from(new Set(allTagsList.map(({ key }) => key))).sort();
  }, [versions]);

  const dispatch = useDispatch<ThunkDispatch>();

  const { EditTagsModal, showEditTagsModal } = useEditKeyValueTagsModal<ModelVersionInfoEntity>({
    allAvailableTags: allTagsKeys,
    saveTagsHandler: async (modelVersion, existingTags, newTags) =>
      dispatch(updateModelVersionTagsApi(modelVersion, existingTags, newTags)),
    onSuccess: onMetadataUpdated,
  });

  const { EditAliasesModal, showEditAliasesModal } = useEditAliasesModal({
    aliases: modelEntity?.aliases ?? [],
    onSuccess: onMetadataUpdated,
    onSave: async (currentlyEditedVersion: string, existingAliases: string[], draftAliases: string[]) =>
      dispatch(setModelVersionAliasesApi(modelName, currentlyEditedVersion, existingAliases, draftAliases)),
    getTitle: getAliasesModalTitle,
    description: (
      <FormattedMessage
        defaultMessage="Aliases allow you to assign a mutable, named reference to a particular model version. <link>Learn more</link>"
        description="Explanation of registered model aliases"
        values={{
          link: (chunks) => (
            <a href={mlflowAliasesLearnMoreLink} rel="noreferrer" target="_blank">
              {chunks}
            </a>
          ),
        }}
      />
    ),
  });

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  useEffect(() => {
    const selectedVersions = (versions || []).filter(({ version }) => rowSelection[version]);
    const selectedVersionNumbers = selectedVersions.map(({ version }) => version);
    onChange(selectedVersionNumbers, selectedVersions);
  }, [rowSelection, onChange, versions]);

  const tableColumns = useMemo(() => {
    const columns: ModelVersionColumnDef[] = [
      {
        id: COLUMN_IDS.STATUS,
        enableSorting: false,
        header: '', // Status column does not have title
        meta: { styles: { flexBasis: theme.general.heightSm, flexGrow: 0 } },
        cell: ({ row: { original } }) => {
          const { status, status_message } = original || {};
          return (
            <LegacyTooltip title={status_message || modelVersionStatusIconTooltips[status]}>
              <Typography.Text>{ModelVersionStatusIcons[status]}</Typography.Text>
            </LegacyTooltip>
          );
        },
      },
    ];
    columns.push(
      {
        id: COLUMN_IDS.VERSION,
        enableSorting: false,
        header: intl.formatMessage({
          defaultMessage: 'Version',
          description: 'Column title text for model version in model version table',
        }),
        meta: { className: 'model-version' },
        accessorKey: 'version',
        cell: ({ getValue }) => (
          <FormattedMessage
            defaultMessage="<link>Version {versionNumber}</link>"
            description="Link to model version in the model version table"
            values={{
              link: (chunks) => (
                <Link to={ModelRegistryRoutes.getModelVersionPageRoute(modelName, String(getValue()))}>{chunks}</Link>
              ),
              versionNumber: getValue(),
            }}
          />
        ),
      },
      {
        id: COLUMN_IDS.CREATION_TIMESTAMP,
        enableSorting: true,
        meta: { styles: { minWidth: 200 } },
        header: intl.formatMessage({
          defaultMessage: 'Registered at',
          description: 'Column title text for created at timestamp in model version table',
        }),
        accessorKey: 'creation_timestamp',
        cell: ({ getValue }) => Utils.formatTimestamp(getValue(), intl),
      },

      {
        id: COLUMN_IDS.USER_ID,
        enableSorting: false,
        meta: { styles: { minWidth: 100 } },
        header: intl.formatMessage({
          defaultMessage: 'Created by',
          description: 'Column title text for creator username in model version table',
        }),
        accessorKey: 'user_id',
        cell: ({ getValue }) => <span>{getValue()}</span>,
      },
    );

    if (usingNextModelsUI) {
      // Display tags and aliases columns only when "new models UI" is flipped
      columns.push(
        {
          id: COLUMN_IDS.TAGS,
          enableSorting: false,
          header: intl.formatMessage({
            defaultMessage: 'Tags',
            description: 'Column title text for model version tags in model version table',
          }),
          meta: { styles: { flex: 2 } },
          accessorKey: 'tags',
          cell: ({ getValue, row: { original } }) => {
            return (
              <KeyValueTagsEditorCell
                tags={getValue() as KeyValueEntity[]}
                onAddEdit={() => {
                  showEditTagsModal?.(original);
                }}
              />
            );
          },
        },
        {
          id: COLUMN_IDS.ALIASES,
          accessorKey: 'aliases',
          enableSorting: false,
          header: intl.formatMessage({
            defaultMessage: 'Aliases',
            description: 'Column title text for model version aliases in model version table',
          }),
          meta: { styles: { flex: 2 }, multiline: true },
          cell: ({ getValue, row: { original } }) => {
            const mvAliases = aliasesByVersion[original.version] || [];
            return (
              <ModelVersionTableAliasesCell
                modelName={modelName}
                version={original.version}
                aliases={mvAliases}
                onAddEdit={() => {
                  showEditAliasesModal?.(original.version);
                }}
              />
            );
          },
        },
      );
    } else {
      // If not, display legacy "Stage" columns
      columns.push({
        id: COLUMN_IDS.STAGE,
        enableSorting: false,
        header: intl.formatMessage({
          defaultMessage: 'Stage',
          description: 'Column title text for model version stage in model version table',
        }),
        accessorKey: 'current_stage',
        cell: ({ getValue }) => {
          return StageTagComponents[getValue() as string];
        },
      });
    }
    columns.push({
      id: COLUMN_IDS.DESCRIPTION,
      enableSorting: false,
      header: intl.formatMessage({
        defaultMessage: 'Description',
        description: 'Column title text for description in model version table',
      }),
      meta: { styles: { flex: 2 } },
      accessorKey: 'description',
      cell: ({ getValue }) => truncateToFirstLineWithMaxLength(getValue() as string, 32),
    });
    return columns;
  }, [theme, intl, modelName, showEditTagsModal, showEditAliasesModal, usingNextModelsUI, aliasesByVersion]);

  const sorting: SortingState = [{ id: orderByKey, desc: !orderByAsc }];

  const setSorting = (stateUpdater: SortingState | ((state: SortingState) => SortingState)) => {
    const [newSortState] = typeof stateUpdater === 'function' ? stateUpdater(sorting) : stateUpdater;
    if (newSortState) {
      onSortChange({ sorter: newSortState });
    }
  };

  const table = useReactTable<ModelVersionInfoEntity>({
    data: versions || [],
    columns: tableColumns,
    state: {
      rowSelection,
      sorting,
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getRowId: ({ version }) => version,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
  });

  const isEmpty = () => table.getRowModel().rows.length === 0;

  const getLearnMoreLinkUrl = () => {
    return RegisteringModelDocUrl;
  };

  const emptyComponent = (
    <Empty
      description={
        <FormattedMessage
          defaultMessage="No models versions are registered yet. <link>Learn more</link> about how to
          register a model version."
          description="Message text when no model versions are registered"
          values={{
            link: (chunks) => (
              <Typography.Link
                componentId="codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_425"
                target="_blank"
                href={getLearnMoreLinkUrl()}
              >
                {chunks}
              </Typography.Link>
            ),
          }}
        />
      }
      image={<PlusIcon />}
    />
  );
  return (
    <>
      <Table
        data-testid="model-version-table"
        pagination={pagination}
        scrollable
        empty={isEmpty() ? emptyComponent : undefined}
        someRowsSelected={table.getIsSomeRowsSelected() || table.getIsAllRowsSelected()}
      >
        <TableRow isHeader>
          <TableRowSelectCell
            componentId="codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_450"
            checked={table.getIsAllRowsSelected()}
            indeterminate={table.getIsSomeRowsSelected()}
            onChange={table.getToggleAllRowsSelectedHandler()}
          />
          {table.getLeafHeaders().map((header) => (
            <TableHeader
              componentId="codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_458"
              multiline={false}
              key={header.id}
              sortable={header.column.getCanSort()}
              sortDirection={header.column.getIsSorted() || 'none'}
              onToggleSort={() => {
                const [currentSortColumn] = sorting;
                const changingDirection = header.column.id === currentSortColumn.id;
                const sortDesc = changingDirection ? !currentSortColumn.desc : false;
                header.column.toggleSorting(sortDesc);
              }}
              css={(header.column.columnDef as ModelVersionColumnDef).meta?.styles}
            >
              {flexRender(header.column.columnDef.header, header.getContext())}
            </TableHeader>
          ))}
        </TableRow>
        {isLoading ? (
          <TableSkeletonRows table={table} />
        ) : (
          table.getRowModel().rows.map((row) => (
            <TableRow
              key={row.id}
              css={{
                '.table-row-select-cell': {
                  alignItems: 'flex-start',
                },
              }}
            >
              <TableRowSelectCell
                componentId="codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_477"
                checked={row.getIsSelected()}
                onChange={row.getToggleSelectedHandler()}
              />
              {row.getAllCells().map((cell) => (
                <TableCell
                  className={(cell.column.columnDef as ModelVersionColumnDef).meta?.className}
                  multiline={(cell.column.columnDef as ModelVersionColumnDef).meta?.multiline}
                  key={cell.id}
                  css={(cell.column.columnDef as ModelVersionColumnDef).meta?.styles}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))
        )}
      </Table>
      {EditTagsModal}
      {EditAliasesModal}
    </>
  );
};
