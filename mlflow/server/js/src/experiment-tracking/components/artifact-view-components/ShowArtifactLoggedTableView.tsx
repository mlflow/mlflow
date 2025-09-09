import {
  Button,
  DangerIcon,
  DropdownMenu,
  Empty,
  GearIcon,
  Pagination,
  SidebarIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeleton,
  LegacyTooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { isArray, isObject, isUndefined } from 'lodash';
import { FormattedMessage, useIntl } from 'react-intl';
import { getArtifactContent, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { SortingState, PaginationState } from '@tanstack/react-table';
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  useReactTable,
} from '@tanstack/react-table';
import React from 'react';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import type { ArtifactLogTableImageObject } from '@mlflow/mlflow/src/experiment-tracking/types';
import { LOG_TABLE_IMAGE_COLUMN_TYPE } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { ImagePlot } from '../runs-charts/components/charts/ImageGridPlot.common';
import { ToggleIconButton } from '../../../common/components/ToggleIconButton';
import { ShowArtifactLoggedTableViewDataPreview } from './ShowArtifactLoggedTableViewDataPreview';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

const MAX_ROW_HEIGHT = 160;
const MIN_COLUMN_WIDTH = 100;
const getDuboisTableHeight = (isCompact?: boolean) => 1 + (isCompact ? 24 : 32);
const DEFAULT_PAGINATION_COMPONENT_HEIGHT = 48;

/**
 * This function ensures we have a valid ID for every column in the table.
 * If the column name is a number, null or undefined we will convert it to a string.
 * If the column name is an empty string, we will use a fallback name with numbered suffix.
 * Refer to the corresponding unit test for more context.
 */
const sanitizeColumnId = (columnName: string, columnIndex: number) =>
  columnName === '' ? `column-${columnIndex + 1}` : String(columnName);

const LoggedTable = ({ data, runUuid }: { data: { columns: string[]; data: any[][] }; runUuid: string }) => {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [isCompactView, setIsCompactView] = useState(false);
  const intl = useIntl();

  const { theme } = useDesignSystemTheme();

  // MAX_IMAGE_SIZE would be the minimum of the max row height and the cell width
  // max(image width, image height) <= MAX_IMAGE_SIZE
  const MAX_IMAGE_SIZE = MAX_ROW_HEIGHT - 2 * theme.spacing.sm;

  const containerRef = useRef<HTMLDivElement>(null);
  // Use resize observer to measure the containerRef width and height
  const [containerDimensions, setContainerDimensions] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect();
      setContainerDimensions({ width, height });
    }
  }, []);

  const columns = useMemo(() => data['columns']?.map(sanitizeColumnId) ?? [], [data]);
  const [hiddenColumns, setHiddenColumns] = useState<string[]>([]);
  const [previewData, setPreviewData] = useState<string | undefined>(undefined);
  const rows = useMemo(() => data['data'], [data]);

  const imageColumns = useMemo(() => {
    // Check if the column is an image column based on the type of element in the first row
    if (rows.length > 0) {
      return columns.filter((col: string, index: number) => {
        // Check that object is of type ArtifactLogTableImageObject
        if (rows[0][index] !== null && typeof rows[0][index] === 'object') {
          const { type } = rows[0][index] as ArtifactLogTableImageObject;
          return type === LOG_TABLE_IMAGE_COLUMN_TYPE;
        } else {
          return false;
        }
      });
    }
    return [];
  }, [columns, rows]);

  // Calculate the number of rows that can fit in the container, flooring the integer value
  const numRowsPerPage = useMemo(() => {
    const tableRowHeight = getDuboisTableHeight(isCompactView);
    if (imageColumns.length > 0) {
      return Math.floor(
        (containerDimensions.height - tableRowHeight - DEFAULT_PAGINATION_COMPONENT_HEIGHT) / MAX_ROW_HEIGHT,
      );
    } else {
      return Math.floor(
        (containerDimensions.height - tableRowHeight - DEFAULT_PAGINATION_COMPONENT_HEIGHT) / tableRowHeight,
      );
    }
  }, [containerDimensions, imageColumns, isCompactView]);

  const [pagination, setPagination] = useState<PaginationState>({
    pageSize: 1,
    pageIndex: 0,
  });

  useEffect(() => {
    // Set pagination when numRowsPerPage changes
    setPagination((pagination) => {
      return { ...pagination, pageSize: numRowsPerPage };
    });
  }, [numRowsPerPage]);

  const tableColumns = useMemo(
    () =>
      columns
        .filter((col) => !hiddenColumns.includes(col))
        .map((col: string) => {
          const col_string = String(col);
          if (imageColumns.includes(col)) {
            return {
              id: col_string,
              header: col_string,
              accessorKey: col_string,
              minSize: MIN_COLUMN_WIDTH,
              cell: (row: any) => {
                try {
                  const parsedRowValue = JSON.parse(row.getValue());
                  const { filepath, compressed_filepath } = parsedRowValue as ArtifactLogTableImageObject;
                  const imageUrl = getArtifactLocationUrl(filepath, runUuid);
                  const compressedImageUrl = getArtifactLocationUrl(compressed_filepath, runUuid);
                  return (
                    <ImagePlot
                      imageUrl={imageUrl}
                      compressedImageUrl={compressedImageUrl}
                      maxImageSize={MAX_IMAGE_SIZE}
                    />
                  );
                } catch {
                  Utils.logErrorAndNotifyUser("Error parsing image data in logged table's image column");
                  return row.getValue();
                }
              },
            };
          }
          return {
            id: col_string,
            header: col_string,
            accessorKey: col_string,
            minSize: MIN_COLUMN_WIDTH,
          };
        }),
    [columns, MAX_IMAGE_SIZE, imageColumns, runUuid, hiddenColumns],
  );
  const tableData = useMemo(
    () =>
      rows.map((row: any[]) => {
        const obj: Record<string, any> = {};
        for (let i = 0; i < columns.length; i++) {
          const cellData = row[i];
          obj[columns[i]] = typeof cellData === 'string' ? cellData : JSON.stringify(cellData);
        }
        return obj;
      }),
    [rows, columns],
  );
  const table = useReactTable({
    columns: tableColumns,
    data: tableData,
    state: {
      pagination,
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
  });

  const paginationComponent = (
    <Pagination
      componentId="codegen_mlflow_app_src_experiment-tracking_components_artifact-view-components_showartifactloggedtableview.tsx_181"
      currentPageIndex={pagination.pageIndex + 1}
      numTotal={rows.length}
      onChange={(page, pageSize) => {
        setPagination({
          pageSize: pageSize || pagination.pageSize,
          pageIndex: page - 1,
        });
      }}
      pageSize={pagination.pageSize}
    />
  );

  return (
    <div
      ref={containerRef}
      css={{
        paddingLeft: theme.spacing.md,
        height: '100%',
        display: 'flex',
        gap: theme.spacing.xs,
        overflow: 'hidden',
      }}
    >
      <div css={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div css={{ overflow: 'auto', flex: 1 }}>
          <Table
            scrollable
            size={isCompactView ? 'small' : 'default'}
            css={{
              '.table-header-icon-container': {
                lineHeight: 0,
              },
            }}
            style={{ width: table.getTotalSize() }}
          >
            {table.getHeaderGroups().map((headerGroup) => {
              return (
                <TableRow isHeader key={headerGroup.id}>
                  {headerGroup.headers.map((header, index) => {
                    return (
                      <TableHeader
                        componentId="codegen_mlflow_app_src_experiment-tracking_components_artifact-view-components_showartifactloggedtableview.tsx_223"
                        key={header.id}
                        sortable
                        sortDirection={header.column.getIsSorted() || 'none'}
                        onToggleSort={header.column.getToggleSortingHandler()}
                        header={header}
                        column={header.column}
                        setColumnSizing={table.setColumnSizing}
                        isResizing={header.column.getIsResizing()}
                        style={{ maxWidth: header.column.getSize() }}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                      </TableHeader>
                    );
                  })}
                </TableRow>
              );
            })}
            {table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getAllCells().map((cell) => {
                  return (
                    <TableCell
                      css={{
                        maxHeight: MAX_ROW_HEIGHT,
                        '&:hover': {
                          backgroundColor: theme.colors.tableBackgroundSelectedHover,
                          cursor: 'pointer',
                        },
                      }}
                      key={cell.id}
                      onClick={() => {
                        setPreviewData(String(cell.getValue()));
                      }}
                      // Enable keyboard navigation
                      tabIndex={0}
                      onKeyDown={({ key }) => {
                        if (key === 'Enter') {
                          setPreviewData(String(cell.getValue()));
                        }
                      }}
                      style={{ maxWidth: cell.column.getSize() }}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </Table>
        </div>
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-end',
            paddingBottom: theme.spacing.sm,
            paddingTop: theme.spacing.sm,
          }}
        >
          {paginationComponent}
        </div>
      </div>
      {!isUndefined(previewData) && (
        <ShowArtifactLoggedTableViewDataPreview data={previewData} onClose={() => setPreviewData(undefined)} />
      )}
      <div
        css={{
          paddingTop: theme.spacing.sm,
          paddingRight: theme.spacing.sm,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
        }}
      >
        <DropdownMenu.Root modal={false}>
          <LegacyTooltip
            title={intl.formatMessage({
              defaultMessage: 'Table settings',
              description: 'Run view > artifact view > logged table > table settings tooltip',
            })}
            useAsLabel
          >
            <DropdownMenu.Trigger
              asChild
              aria-label={intl.formatMessage({
                defaultMessage: 'Table settings',
                description: 'Run view > artifact view > logged table > table settings tooltip',
              })}
            >
              <Button componentId="mlflow.run.artifact_view.table_settings" icon={<GearIcon />} />
            </DropdownMenu.Trigger>
          </LegacyTooltip>
          <DropdownMenu.Content css={{ maxHeight: theme.general.heightSm * 10, overflowY: 'auto' }} side="left">
            <DropdownMenu.Arrow />
            <DropdownMenu.CheckboxItem
              componentId="codegen_mlflow_app_src_experiment-tracking_components_artifact-view-components_showartifactloggedtableview.tsx_315"
              checked={isCompactView}
              onCheckedChange={setIsCompactView}
            >
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Compact view"
                description="Run page > artifact view > logged table view > compact view toggle button"
              />
            </DropdownMenu.CheckboxItem>
            <DropdownMenu.Separator />
            <DropdownMenu.Group>
              <DropdownMenu.Label>
                <FormattedMessage
                  defaultMessage="Columns"
                  description="Run page > artifact view > logged table view > columns selector label"
                />
              </DropdownMenu.Label>
              {columns.map((column) => (
                <DropdownMenu.CheckboxItem
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_artifact-view-components_showartifactloggedtableview.tsx_331"
                  onSelect={(event) => event.preventDefault()}
                  checked={!hiddenColumns.includes(column)}
                  key={column}
                  onCheckedChange={() => {
                    setHiddenColumns((prev) => {
                      if (prev.includes(column)) {
                        return prev.filter((col) => col !== column);
                      } else {
                        return [...prev, column];
                      }
                    });
                  }}
                >
                  <DropdownMenu.ItemIndicator />
                  {column}
                </DropdownMenu.CheckboxItem>
              ))}
            </DropdownMenu.Group>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
        <ToggleIconButton
          onClick={() => {
            setPreviewData(() => {
              return !isUndefined(previewData) ? undefined : '';
            });
          }}
          pressed={!isUndefined(previewData)}
          componentId="mlflow.run.artifact_view.preview_sidebar_toggle"
          icon={<SidebarIcon />}
        />
      </div>
    </div>
  );
};

type ShowArtifactLoggedTableViewProps = {
  runUuid: string;
  path: string;
} & LoggedModelArtifactViewerProps;

export const ShowArtifactLoggedTableView = React.memo(
  ({
    runUuid,
    path,
    isLoggedModelsMode,
    loggedModelId,
    experimentId,
    entityTags,
  }: ShowArtifactLoggedTableViewProps) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<Error>();
    const [curPath, setCurPath] = useState<string | undefined>(undefined);
    const [text, setText] = useState<string>('');

    useEffect(() => {
      setLoading(true);
      fetchArtifactUnified(
        { runUuid, path, isLoggedModelsMode, loggedModelId, experimentId, entityTags },
        getArtifactContent,
      )
        .then((value) => {
          setLoading(false);
          // Check if value is stringified JSON
          if (value && typeof value === 'string') {
            setText(value);
            setError(undefined);
          } else {
            setError(Error('Artifact is not a JSON file'));
          }
        })
        .catch((error: Error) => {
          setError(error);
          setLoading(false);
        });
      setCurPath(path);
    }, [path, runUuid, isLoggedModelsMode, loggedModelId, experimentId, entityTags]);

    const data = useMemo<{
      columns: string[];
      data: any[][];
    }>(() => {
      const parsedJSON = parseJSONSafe(text);
      if (!parsedJSON || !isArray(parsedJSON?.columns) || !isArray(parsedJSON?.data)) {
        return undefined;
      }
      return parsedJSON;
    }, [text]);

    const { theme } = useDesignSystemTheme();

    const renderErrorState = (description: React.ReactNode) => {
      return (
        <div css={{ padding: theme.spacing.md }}>
          <Empty
            image={<DangerIcon />}
            title={
              <FormattedMessage
                defaultMessage="Error occurred"
                description="Run page > artifact view > logged table view > generic error empty state title"
              />
            }
            description={description}
          />
        </div>
      );
    };

    if (loading || path !== curPath) {
      return (
        <div
          css={{
            padding: theme.spacing.md,
          }}
        >
          <TableSkeleton lines={5} />
        </div>
      );
    }
    if (error) {
      return renderErrorState(error.message);
    } else if (text) {
      if (!data) {
        return renderErrorState(
          <FormattedMessage
            defaultMessage="Unable to parse JSON file. The file should contain an object with 'columns' and 'data' keys."
            description="An error message displayed when the logged table JSON file is malformed or does not contain 'columns' and 'data' keys"
          />,
        );
      }
      return <LoggedTable data={data} runUuid={runUuid} />;
    }
    return renderErrorState(null);
  },
);
