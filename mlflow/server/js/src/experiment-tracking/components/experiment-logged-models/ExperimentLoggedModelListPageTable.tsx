import {
  Button,
  Empty,
  getShadowScrollStyles,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import MLFlowAgGrid from '../../../common/components/ag-grid/AgGrid';
import { useExperimentAgGridTableStyles } from '../experiment-page/components/runs/ExperimentViewRunsTable';
import type { LoggedModelProto, RunEntity } from '../../types';
import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import {
  ExperimentLoggedModelListPageTableContextProvider,
  useExperimentLoggedModelListPageTableContext,
} from './ExperimentLoggedModelListPageTableContext';
import { LoggedModelsListPageSortableColumns } from './hooks/useLoggedModelsListPagePageState';
import type { ColumnApi, IsFullWidthRowParams } from '@ag-grid-community/core';
import { type ColDef, type ColGroupDef, type SortChangedEvent } from '@ag-grid-community/core';
import { FormattedMessage } from 'react-intl';
import { useRunsHighlightTableRow } from '../runs-charts/hooks/useRunsHighlightTableRow';
import { ExperimentLoggedModelListPageTableEmpty } from './ExperimentLoggedModelListPageTableEmpty';
import { LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX } from './hooks/useExperimentLoggedModelListPageTableColumns';
import { first, groupBy, isEmpty, orderBy } from 'lodash';
import type {
  LoggedModelDataWithSourceRun,
  LoggedModelsTableGroupByMode,
  LoggedModelsTableRow,
} from './ExperimentLoggedModelListPageTable.utils';
import {
  getLoggedModelsTableRowID,
  LoggedModelsTableDataRow,
  LoggedModelsTableGroupHeaderRowClass,
  LoggedModelsTableGroupingEnabledClass,
  LoggedModelsTableLoadMoreRowSymbol,
  LoggedModelsTableSpecialRowID,
  useLoggedModelTableDataRows,
} from './ExperimentLoggedModelListPageTable.utils';

const LOGGED_MODELS_GRID_ROW_HEIGHT = 36;

interface ExperimentLoggedModelListPageTableProps {
  loggedModels?: LoggedModelProto[];
  isLoading: boolean;
  isLoadingMore: boolean;
  badRequestError?: Error;
  moreResultsAvailable?: boolean;
  onLoadMore?: () => void;
  onOrderByChange?: (orderByColumn: string, orderByAsc: boolean) => void;
  orderByColumn?: string;
  orderByAsc?: boolean;
  columnDefs?: (ColDef | ColGroupDef)[];
  columnVisibility?: Record<string, boolean>;
  relatedRunsData?: RunEntity[] | null;
  className?: string;
  disableLoadMore?: boolean;
  displayShowExampleButton?: boolean;
  isFilteringActive?: boolean;
  groupModelsBy?: LoggedModelsTableGroupByMode | undefined;
}

const ExperimentLoggedModelListPageTableImpl = ({
  loggedModels,
  isLoading,
  isLoadingMore,
  badRequestError,
  onLoadMore,
  orderByColumn,
  orderByAsc,
  moreResultsAvailable,
  onOrderByChange,
  columnDefs = [],
  columnVisibility,
  relatedRunsData,
  className,
  disableLoadMore,
  displayShowExampleButton = true,
  isFilteringActive = true,
  groupModelsBy,
}: ExperimentLoggedModelListPageTableProps) => {
  const { theme } = useDesignSystemTheme();

  const styles = useExperimentAgGridTableStyles({ usingCustomHeaderComponent: false });

  // Keep track of expanded groups in the table
  const [expandedGroups, setExpandedGroups] = React.useState<string[]>([]);

  const columnApiRef = useRef<ColumnApi | null>(null);

  const loggedModelsWithSourceRuns = useMemo<LoggedModelDataWithSourceRun[] | undefined>(() => {
    if (!loggedModels || !relatedRunsData) {
      return loggedModels;
    }
    return loggedModels.map((loggedModel) => {
      const sourceRun = relatedRunsData.find((run) => run?.info?.runUuid === loggedModel?.info?.source_run_id);
      return { ...loggedModel, sourceRun };
    });
  }, [loggedModels, relatedRunsData]);

  // Expand or collapse the group based on its ID
  const onGroupToggle = useCallback((groupId: string) => {
    setExpandedGroups((prev) => (prev.includes(groupId) ? prev.filter((id) => id !== groupId) : [...prev, groupId]));
  }, []);

  // Get all data rows in the table: logged models and groups if applicable
  const loggedModelsDataRows = useLoggedModelTableDataRows({
    loggedModelsWithSourceRuns,
    groupModelsBy,
    expandedGroups,
  });

  // Get all the table rows, including data rows and the "Load more" row if applicable
  const loggedModelsTableRows = useMemo<LoggedModelsTableRow[] | undefined>(() => {
    if (isLoading) {
      return undefined;
    }
    if (disableLoadMore || !loggedModelsDataRows || loggedModelsDataRows.length === 0) {
      return loggedModelsDataRows;
    }
    return [...loggedModelsDataRows, LoggedModelsTableLoadMoreRowSymbol];
  }, [loggedModelsDataRows, isLoading, disableLoadMore]);

  const sortChangedHandler = useCallback(
    (event: SortChangedEvent) => {
      // Find the currently sorted column using ag-grid's column API
      const sortedColumn = event.columnApi.getColumnState().find((col) => col.sort);
      if (!sortedColumn?.colId) {
        return;
      }
      if (
        LoggedModelsListPageSortableColumns.includes(sortedColumn.colId) ||
        sortedColumn.colId.startsWith(LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX)
      ) {
        onOrderByChange?.(sortedColumn?.colId, sortedColumn.sort === 'asc');
      }
    },
    [onOrderByChange],
  );

  const updateSortIndicator = useCallback((field?: string, asc?: boolean) => {
    // Reflect the sort state in the ag-grid's column state
    const column = columnApiRef.current?.getColumn(field);
    if (column) {
      // Find the currently sorted column and if it's no the same one, clear its sort state
      const currentSortedColumnId = columnApiRef.current?.getColumnState().find((col) => col.sort)?.colId;
      if (currentSortedColumnId !== column.getColId()) {
        columnApiRef.current?.getColumn(currentSortedColumnId)?.setSort(null);
      }
      column.setSort(asc ? 'asc' : 'desc');
    }
  }, []);

  const updateColumnVisibility = useCallback((newColumnVisibility?: Record<string, boolean>) => {
    // Reflect the visibility state in the ag-grid's column state
    for (const column of columnApiRef?.current?.getAllColumns() ?? []) {
      columnApiRef.current?.setColumnVisible(column, newColumnVisibility?.[column.getColId()] !== false);
    }
  }, []);

  // Since ag-grid column API is not stateful, we use side effect to update the UI
  useEffect(() => updateSortIndicator(orderByColumn, orderByAsc), [updateSortIndicator, orderByColumn, orderByAsc]);
  useEffect(() => updateColumnVisibility(columnVisibility), [updateColumnVisibility, columnVisibility]);

  const containsGroupedColumns = useMemo(() => columnDefs.some((col) => 'children' in col), [columnDefs]);

  const containerElement = useRef<HTMLDivElement | null>(null);

  const { cellMouseOverHandler, cellMouseOutHandler } = useRunsHighlightTableRow(
    containerElement,
    undefined,
    true,
    getLoggedModelsTableRowID,
  );

  return (
    <ExperimentLoggedModelListPageTableContextProvider
      loadMoreResults={onLoadMore}
      moreResultsAvailable={moreResultsAvailable}
      isLoadingMore={isLoadingMore}
      expandedGroups={expandedGroups}
      onGroupToggle={onGroupToggle}
    >
      <div
        css={{
          overflow: 'hidden',
          flex: 1,
          ...styles,
          '.ag-cell': {
            alignItems: 'center',
            [`&.${LoggedModelsTableGroupHeaderRowClass}`]: {
              overflow: 'visible',
            },
          },
          borderTop: `1px solid ${theme.colors.border}`,
          '.ag-header-cell.is-checkbox-header-cell': {
            paddingLeft: theme.spacing.sm,
          },
          '&& .ag-root-wrapper': { border: 0 },
        }}
        className={[
          'ag-theme-balham',
          className,
          // When using grouping, add a special class to the table
          // to enable padding
          groupModelsBy ? LoggedModelsTableGroupingEnabledClass : '',
        ].join(' ')}
        ref={containerElement}
      >
        <MLFlowAgGrid
          columnDefs={columnDefs}
          rowData={loggedModelsTableRows}
          rowHeight={LOGGED_MODELS_GRID_ROW_HEIGHT}
          rowSelection="multiple"
          suppressRowClickSelection
          suppressMovableColumns
          getRowId={getLoggedModelsTableRowID}
          suppressLoadingOverlay
          suppressNoRowsOverlay
          suppressColumnMoveAnimation
          isFullWidthRow={isFullWidthRow}
          fullWidthCellRenderer={LoadMoreRow}
          onSortChanged={sortChangedHandler}
          onGridReady={({ columnApi }) => {
            columnApiRef.current = columnApi;
            updateSortIndicator(orderByColumn, orderByAsc);
            updateColumnVisibility(columnVisibility);
          }}
          onCellMouseOver={cellMouseOverHandler}
          onCellMouseOut={cellMouseOutHandler}
        />
        {isLoading && (
          <div
            css={{
              inset: 0,
              top: (containsGroupedColumns ? theme.general.heightBase : 0) + theme.spacing.lg,
              position: 'absolute',
              paddingTop: theme.spacing.md,
              paddingRight: theme.spacing.md,
            }}
          >
            <TableSkeleton
              lines={8}
              label={
                <FormattedMessage
                  defaultMessage="Models loading"
                  description="Label for a loading spinner when table containing models is being loaded"
                />
              }
            />
          </div>
        )}
        {!isLoading && loggedModels?.length === 0 && (
          <ExperimentLoggedModelListPageTableEmpty
            displayShowExampleButton={displayShowExampleButton}
            badRequestError={badRequestError}
            isFilteringActive={isFilteringActive}
          />
        )}
      </div>
    </ExperimentLoggedModelListPageTableContextProvider>
  );
};

const LoadMoreRow = () => {
  const { theme } = useDesignSystemTheme();

  const { moreResultsAvailable, loadMoreResults, isLoadingMore } = useExperimentLoggedModelListPageTableContext();

  if (!moreResultsAvailable) {
    return null;
  }
  return (
    <div
      css={{
        pointerEvents: 'all',
        userSelect: 'all',
        padding: theme.spacing.sm,
        display: 'flex',
        justifyContent: 'center',
      }}
    >
      <Button
        componentId="mlflow.logged_models.list.load_more"
        type="primary"
        size="small"
        onClick={loadMoreResults}
        loading={isLoadingMore}
      >
        <FormattedMessage
          defaultMessage="Load more"
          description="Label for a button to load more results in the logged models table"
        />
      </Button>
    </div>
  );
};

export const ExperimentLoggedModelListPageTable = React.memo(ExperimentLoggedModelListPageTableImpl);

const isFullWidthRow: ((params: IsFullWidthRowParams) => boolean) | undefined = ({ rowNode }) =>
  rowNode.data === LoggedModelsTableLoadMoreRowSymbol;
