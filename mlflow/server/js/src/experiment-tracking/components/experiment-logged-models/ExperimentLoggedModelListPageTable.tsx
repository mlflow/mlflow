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
import { useCallback, useEffect, useMemo, useRef } from 'react';
import {
  ExperimentLoggedModelListPageTableContextProvider,
  useExperimentLoggedModelListPageTableContext,
} from './ExperimentLoggedModelListPageTableContext';
import { LoggedModelsListPageSortableColumns } from './hooks/useLoggedModelsListPagePageState';
import { type ColDef, type ColGroupDef, ColumnApi, type SortChangedEvent } from '@ag-grid-community/core';
import { FormattedMessage } from 'react-intl';
import { useRunsHighlightTableRow } from '../runs-charts/hooks/useRunsHighlightTableRow';
import { ExperimentLoggedModelListPageTableEmpty } from './ExperimentLoggedModelListPageTableEmpty';

const LOGGED_MODELS_GRID_ROW_HEIGHT = 36;

interface ExperimentLoggedModelListPageTableProps {
  loggedModels?: LoggedModelProto[];
  isLoading: boolean;
  isLoadingMore: boolean;
  error?: Error | null;
  moreResultsAvailable?: boolean;
  onLoadMore?: () => void;
  onOrderByChange?: (orderByField: string, orderByAsc: boolean) => void;
  orderByField?: string;
  orderByAsc?: boolean;
  columnDefs?: (ColDef | ColGroupDef)[];
  columnVisibility?: Record<string, boolean>;
  relatedRunsData?: RunEntity[] | null;
  className?: string;
  disableLoadMore?: boolean;
  displayShowExampleButton?: boolean;
}

const LoadMoreRowSymbol = Symbol('LoadMoreRow');

const rowDataGetter = ({ data }: { data: LoggedModelProto }) => data?.info?.model_id ?? '';

export const ExperimentLoggedModelListPageTable = ({
  loggedModels,
  isLoading,
  isLoadingMore,
  onLoadMore,
  orderByField,
  orderByAsc,
  moreResultsAvailable,
  onOrderByChange,
  columnDefs = [],
  columnVisibility,
  relatedRunsData,
  className,
  disableLoadMore,
  displayShowExampleButton = true,
}: ExperimentLoggedModelListPageTableProps) => {
  const { theme } = useDesignSystemTheme();
  const styles = useExperimentAgGridTableStyles({ usingCustomHeaderComponent: false });

  const columnApiRef = useRef<ColumnApi | null>(null);

  const loggedModelsWithSourceRuns = useMemo(() => {
    if (!loggedModels || !relatedRunsData) {
      return loggedModels;
    }
    return loggedModels.map((loggedModel) => {
      const sourceRun = relatedRunsData.find((run) => run?.info?.runUuid === loggedModel?.info?.source_run_id);
      return { ...loggedModel, sourceRun };
    });
  }, [loggedModels, relatedRunsData]);

  // We need to add "Load more" as a special row at the end
  // of the result list
  const loggedModelsListWithLoadMore = useMemo(() => {
    if (isLoading) {
      return undefined;
    }
    if (disableLoadMore || !loggedModelsWithSourceRuns || loggedModelsWithSourceRuns.length === 0) {
      return loggedModelsWithSourceRuns;
    }
    return [...loggedModelsWithSourceRuns, LoadMoreRowSymbol];
  }, [loggedModelsWithSourceRuns, isLoading, disableLoadMore]);

  const sortChangedHandler = useCallback(
    (event: SortChangedEvent) => {
      for (const column of event.columnApi.getAllColumns() || []) {
        const id = column?.getId();
        if (LoggedModelsListPageSortableColumns.includes(id)) {
          onOrderByChange?.(id, column.getSort() === 'asc');
        }
      }
    },
    [onOrderByChange],
  );

  const updateSortIndicator = useCallback((field?: string, asc?: boolean) => {
    // Reflect the sort state in the ag-grid's column state
    const column = columnApiRef.current?.getColumn(field);
    if (column) {
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
  useEffect(() => updateSortIndicator(orderByField, orderByAsc), [updateSortIndicator, orderByField, orderByAsc]);
  useEffect(() => updateColumnVisibility(columnVisibility), [updateColumnVisibility, columnVisibility]);

  const containsGroupedColumns = useMemo(() => columnDefs.some((col) => 'children' in col), [columnDefs]);

  const containerElement = useRef<HTMLDivElement | null>(null);

  const { cellMouseOverHandler, cellMouseOutHandler } = useRunsHighlightTableRow(
    containerElement,
    undefined,
    true,
    rowDataGetter,
  );

  return (
    <ExperimentLoggedModelListPageTableContextProvider
      loadMoreResults={onLoadMore}
      moreResultsAvailable={moreResultsAvailable}
      isLoadingMore={isLoadingMore}
    >
      <div
        css={{
          overflow: 'hidden',
          flex: 1,
          ...styles,
          '.ag-cell': {
            alignItems: 'center',
          },
          borderTop: `1px solid ${theme.colors.border}`,
          '.ag-header-cell.is-checkbox-header-cell': {
            paddingLeft: theme.spacing.sm,
          },
          '&& .ag-root-wrapper': { border: 0 },
        }}
        className={['ag-theme-balham', className].join(' ')}
        ref={containerElement}
      >
        <MLFlowAgGrid
          columnDefs={columnDefs}
          rowData={loggedModelsListWithLoadMore}
          rowHeight={LOGGED_MODELS_GRID_ROW_HEIGHT}
          rowSelection="multiple"
          suppressRowClickSelection
          suppressMovableColumns
          getRowId={rowDataGetter}
          suppressLoadingOverlay
          suppressNoRowsOverlay
          suppressColumnMoveAnimation
          isFullWidthRow={({ rowNode }) => rowNode.data === LoadMoreRowSymbol}
          fullWidthCellRenderer={LoadMoreRow}
          onSortChanged={sortChangedHandler}
          onGridReady={({ columnApi }) => {
            columnApiRef.current = columnApi;
            updateSortIndicator(orderByField, orderByAsc);
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
          <ExperimentLoggedModelListPageTableEmpty displayShowExampleButton={displayShowExampleButton} />
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
