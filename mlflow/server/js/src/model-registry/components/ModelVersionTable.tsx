import {
  Empty,
  NotificationIcon,
  Pagination,
  PlusIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import {
  ColumnDef,
  PaginationState,
  RowSelectionState,
  SortingState,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { KeyValueEntity, ModelEntity, ModelVersionInfoEntity, ModelAliasMap } from '../../experiment-tracking/types';
import { useEffect, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { RegisteringModelDocUrl } from '../../common/constants';
import { useNextModelsUIContext } from '../hooks/useNextModelsUI';
import {
  ACTIVE_STAGES,
  EMPTY_CELL_PLACEHOLDER,
  ModelVersionStatusIcons,
  StageTagComponents,
  modelVersionStatusIconTooltips,
} from '../constants';
import { Link } from '../../common/utils/RoutingUtils';
import { ModelRegistryRoutes } from '../routes';
import Utils from '../../common/utils/Utils';
import { KeyValueTagsEditorCell } from '../../common/components/KeyValueTagsEditorCell';
import { useDispatch } from 'react-redux';
import { ThunkDispatch } from '../../redux-types';
import { useEditKeyValueTagsModal } from '../../common/hooks/useEditKeyValueTagsModal';
import { useEditRegisteredModelAliasesModal } from '../hooks/useEditRegisteredModelAliasesModal';
import { updateModelVersionTagsApi } from '../actions';
import { ModelVersionTableAliasesCell } from './aliases/ModelVersionTableAliasesCell';
import { Interpolation, Theme } from '@emotion/react';
import { truncateToFirstLineWithMaxLength } from '../../common/utils/StringUtils';
import ExpandableList from '../../common/components/ExpandableList';

type ModelVersionTableProps = {
  modelName: string;
  modelVersions?: ModelVersionInfoEntity[];
  activeStageOnly?: boolean;
  onChange: (selectedRowKeys: string[], selectedRows: ModelVersionInfoEntity[]) => void;
  modelEntity?: ModelEntity;
  onMetadataUpdated: () => void;
  usingNextModelsUI: boolean;
  aliases?: ModelAliasMap
};

type ModelVersionColumnDef = ColumnDef<ModelVersionInfoEntity> & {
  meta?: { styles?: Interpolation<Theme>; multiline?: boolean; className?: string };
};

enum COLUMN_IDS {
  STATUS = 'STATUS',
  VERSION = 'VERSION',
  CREATION_TIMESTAMP = 'CREATION_TIMESTAMP',
  USER_ID = 'USER_ID',
  TAGS = 'TAGS',
  STAGE = 'STAGE',
  DESCRIPTION = 'DESCRIPTION',
  ALIASES = 'ALIASES',
}

export const ModelVersionTable = ({
  modelName,
  modelVersions,
  activeStageOnly,
  onChange,
  modelEntity,
  onMetadataUpdated,
  usingNextModelsUI,
  aliases
}: ModelVersionTableProps) => {
  const aliasesByVersion = useMemo(() => {
    const result: Record<string, string[]> = {};
    aliases?.forEach(({ alias, version }) => {
      if (!result[version]) {
        result[version] = [];
      }
      result[version].push(alias);
    })
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

  const { EditAliasesModal, showEditAliasesModal } = useEditRegisteredModelAliasesModal({
    model: modelEntity || null,
    onSuccess: onMetadataUpdated,
  });

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  const [pagination, setPagination] = useState<PaginationState>({
    pageSize: 10,
    pageIndex: 0,
  });

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
            <Tooltip title={status_message || modelVersionStatusIconTooltips[status]}>
              <Typography.Text>{ModelVersionStatusIcons[status]}</Typography.Text>
            </Tooltip>
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
        cell: ({ getValue }) => Utils.formatTimestamp(getValue()),
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
      cell: ({ getValue }) => truncateToFirstLineWithMaxLength(getValue(), 32),
    });
    return columns;
  }, [theme, intl, modelName, showEditTagsModal, showEditAliasesModal, usingNextModelsUI, aliasesByVersion]);

  const [sorting, setSorting] = useState<SortingState>([{ id: COLUMN_IDS.CREATION_TIMESTAMP, desc: true }]);

  const table = useReactTable<ModelVersionInfoEntity>({
    data: versions || [],
    columns: tableColumns,
    state: {
      pagination,
      rowSelection,
      sorting,
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getRowId: ({ version }) => version,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
  });

  const isEmpty = () => table.getRowModel().rows.length === 0;

  const getLearnMoreLinkUrl = () => {
    return RegisteringModelDocUrl;
  };

  const paginationComponent = (
    <Pagination
      currentPageIndex={pagination.pageIndex + 1}
      numTotal={(versions || []).length}
      onChange={(page, pageSize) => {
        setPagination({
          pageSize: pageSize || pagination.pageSize,
          pageIndex: page - 1,
        });
      }}
      pageSize={pagination.pageSize}
    />
  );

  const emptyComponent = (
    <Empty
      description={
        <FormattedMessage
          defaultMessage="No models versions are registered yet. <link>Learn more</link> about how to
          register a model version."
          description="Message text when no model versions are registered"
          values={{
            link: (chunks) => (
              <Typography.Link target="_blank" href={getLearnMoreLinkUrl()}>
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
        data-testid="model-list-table"
        pagination={paginationComponent}
        scrollable
        empty={isEmpty() ? emptyComponent : undefined}
        someRowsSelected={table.getIsSomeRowsSelected() || table.getIsAllRowsSelected()}
      >
        <TableRow isHeader>
          <TableRowSelectCell
            checked={table.getIsAllRowsSelected()}
            indeterminate={table.getIsSomeRowsSelected()}
            onChange={table.getToggleAllRowsSelectedHandler()}
          />
          {table.getLeafHeaders().map((header) => (
            <TableHeader
              multiline={false}
              key={header.id}
              sortable={header.column.getCanSort()}
              sortDirection={header.column.getIsSorted() || 'none'}
              onToggleSort={header.column.getToggleSortingHandler()}
              css={(header.column.columnDef as ModelVersionColumnDef).meta?.styles}
            >
              {flexRender(header.column.columnDef.header, header.getContext())}
            </TableHeader>
          ))}
        </TableRow>
        {table.getRowModel().rows.map((row) => (
          <TableRow
            key={row.id}
            css={{
              '.table-row-select-cell': {
                alignItems: 'flex-start',
              },
            }}
          >
            <TableRowSelectCell checked={row.getIsSelected()} onChange={row.getToggleSelectedHandler()} />
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
        ))}
      </Table>
      {EditTagsModal}
      {EditAliasesModal}
    </>
  );
};
