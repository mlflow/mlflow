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
  LegacyTooltip,
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
  aliases?: ModelAliasMap;
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
  STATE = 'STATE',
}

// const handleClick = async () => {
//     try {
//       const state = getValue(); // Get the current state
//       const newState = state === 'Retired' ? 'Live' : state === 'New' ? 'Live' : 'Retired';

//       // Get the serving_image tag from the row data
//       const servingImageTag = row.original.tags?.find(tag => tag.key === 'serving_image')?.value;

//       // Send a POST request to the backend
//       const response = await fetch('/api/2.0/mlflow/update-release', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           model_name: row.original.name, // Pass the model name
//           version: row.original.version, // Pass the model version
//           new_state: newState, // Pass the new state
//           serving_image: servingImageTag, // Pass the serving_image tag
//         }),
//       });

//       if (!response.ok) {
//         throw new Error('Failed to update state');
//       }

//       // Refresh the table or update the state locally
//       window.location.reload(); // Simple refresh for now
//     } catch (error) {
//       console.error('Error updating state:', error);
//     }
//   };

export const ModelVersionTable = ({
  modelName,
  modelVersions,
  activeStageOnly,
  onChange,
  modelEntity,
  onMetadataUpdated,
  usingNextModelsUI,
  aliases,
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
      cell: ({ getValue }) => truncateToFirstLineWithMaxLength(getValue(), 32),
    },
    {
      id: COLUMN_IDS.STATE,
      enableSorting: false,
      header: intl.formatMessage({
        defaultMessage: 'State',
        description: 'Column title for current state of the model; New, Live or Retired',
      }),
      meta: { styles: { flex: 2 } },
      accessorKey: 'state',
      cell: ({ getValue }) => {
        const state = getValue(); // Get the state value
        let buttonColor = '';
  
        // Determine button color based on state
        switch (state) {
          case 'New':
            buttonColor = '#59c3dd';
            break;
          case 'Live':
            buttonColor = '#3b9c3f';
            break;
          case 'Retired':
            buttonColor = '#afafaf';
            break;
          default:
            buttonColor = 'black'; // Default color
        }
  
        return (
          <button
            style={{
              backgroundColor: buttonColor,
              color: 'white',
              padding: '5px 10px',
              border: 'none',
              borderRadius: '4px',
              cursor: 'default',
            }}
            disabled
          >
            {state}
          </button>
        );
      }
    },
    {
      id: "ACTION",
      enableSorting: false,
      header: intl.formatMessage({
        defaultMessage: 'Action',
        description: 'Column title for potential action with model; Release or Retract',
      }),
      meta: { styles: { flex: 2 } },
      accessorKey: 'state',
      cell: ({ getValue, row }) => {
        const state = getValue(); // Get the state value
        const actionText = state === 'Retired' || state === 'New' ? 'Release' : 'Retract';
        const buttonColor = actionText === 'Release' ? '#088708' : '#a20a0a';
    
        // Get the serving_image tag from the row data
        // const servingImageTag = row.original.tags?.find(tag => tag.key === 'serving_container')?.value;
        const handleClick = async () => {
          try {
            const state = getValue(); // Get the current state
            const newState = state === 'Retired' ? 'Live' : state === 'New' ? 'Live' : 'Retired';
        
            // Fetch the run associated with the model version
            let data = null; // Declare `data` outside the try block
            let runTags = null;

            try {
              const runResponse = await fetch(`/api/2.0/mlflow/runs/get?run_id=${row.original.run_id}`, {
                method: 'GET',
                headers: {
                  'Content-Type': 'application/json',
                },
              });

              console.log("Response object:", runResponse);

              // Read response as text before parsing
              const text = await runResponse.text();
              console.log("Raw response text:", text);

              // Try parsing it as JSON
              data = JSON.parse(text); // Assign the parsed JSON to `data`
              console.log("Parsed JSON:", data);

            } catch (error) {
              console.error("Error:", error);
            }

            // Now `data` is accessible outside the try block
            if (data) {
              runTags = data.run.data.tags; // Get the tags from the run
              console.log("Run Tags:", runTags);
            } else {
              console.log("No data available or an error occurred.");
            }
            // const runTags = data.run.data.tags; // Get the tags from the run
        
            // Find the serving_container tag from the run tags
            const servingImageTag = runTags.find(tag => tag.key === 'serving_container')?.value;
        
            // Send a POST request to update the model version state
            const updateResponse = await fetch('/api/2.0/mlflow/model-versions/update', {
              method: 'PATCH',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                name: row.original.name, // Pass the model name
                version: row.original.version, // Pass the model version
                state: newState, // Pass the new state
                serving_image: servingImageTag, // Pass the serving_image tag from the run
              }),
            });
            
            // Log the raw response
            console.log('Update Response:', updateResponse);
            
            if (!updateResponse.ok) {
              const errorText = await updateResponse.text(); // Log the error response
              console.error('Update Error:', errorText);
              throw new Error('Failed to update state');
            }
            
            const updateData = await updateResponse.json();
            console.log('Update Data:', updateData); // Log the parsed JSON data
        
            // Refresh the table or update the state locally
            window.location.reload(); // Simple refresh for now
            } catch (error) {
              console.error('Error updating state:', error);
            }
        };
        return (
          <button
            style={{
              backgroundColor: buttonColor,
              color: 'white',
              padding: '5px 10px',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
            onClick={handleClick} // Add onClick handler
          >
            <b>{actionText}</b>
          </button>
        );
      }
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
      componentId="codegen_mlflow_app_src_model-registry_components_modelversiontable.tsx_403"
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
        data-testid="model-list-table"
        pagination={paginationComponent}
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
        ))}
      </Table>
      {EditTagsModal}
      {EditAliasesModal}
    </>
  );
};
