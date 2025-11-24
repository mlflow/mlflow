import type { CellClickedEvent, ColumnApi, GridApi, GridReadyEvent } from '@ag-grid-community/core';
import type { Theme } from '@emotion/react';
import { type CSSObject, Interpolation } from '@emotion/react';
import cx from 'classnames';
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { MLFlowAgGridLoader } from '../../../../../common/components/ag-grid/AgGridLoader';
import Utils from '../../../../../common/utils/Utils';
import type {
  ExperimentEntity,
  UpdateExperimentViewStateFn,
  RunDatasetWithTags,
  ExperimentViewRunsCompareMode,
} from '../../../../types';

import { isSearchFacetsFilterUsed } from '../../utils/experimentPage.fetch-utils';
import type { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import {
  EXPERIMENTS_DEFAULT_COLUMN_SETUP,
  getFrameworkComponents,
  getRowIsLoadMore,
  getRowId,
  useRunsColumnDefinitions,
  getAdjustableAttributeColumns,
} from '../../utils/experimentPage.column-utils';
import { makeCanonicalSortKey, extractCanonicalSortKey } from '../../utils/experimentPage.common-utils';
import { EXPERIMENT_RUNS_TABLE_ROW_HEIGHT } from '../../utils/experimentPage.common-utils';
import type { RunRowType } from '../../utils/experimentPage.row-types';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { createLoadMoreRow } from './cells/LoadMoreRowRenderer';
import { ExperimentViewRunsEmptyTable } from './ExperimentViewRunsEmptyTable';
import { ExperimentViewRunsTableAddColumnCTA } from './ExperimentViewRunsTableAddColumnCTA';
import { ExperimentViewRunsTableStatusBar } from './ExperimentViewRunsTableStatusBar';
import { shouldUseRunRowsVisibilityMap } from '../../../../../common/utils/FeatureUtils';
import { getDatasetsCellHeight } from './cells/DatasetsCellRenderer';
import { PreviewSidebar } from '../../../../../common/components/PreviewSidebar';
import { ATTRIBUTE_COLUMN_LABELS, COLUMN_TYPES } from '../../../../constants';
import { Empty, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';
import { useUpdateExperimentPageSearchFacets } from '../../hooks/useExperimentPageSearchFacets';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { useExperimentTableSelectRowHandler } from '../../hooks/useExperimentTableSelectRowHandler';
import { useToggleRowVisibilityCallback } from '../../hooks/useToggleRowVisibilityCallback';
import { ExperimentViewRunsTableHeaderContextProvider } from './ExperimentViewRunsTableHeaderContext';
import { useRunsHighlightTableRow } from '../../../runs-charts/hooks/useRunsHighlightTableRow';
import { isEmpty } from 'lodash';

const ROW_BUFFER = 101; // How many rows to keep rendered, even ones not visible
const LARGE_COLUMN_COUNT_THRESHOLD = 1000; // Threshold to determine if we should optimize column rendering

export interface ExperimentViewRunsTableProps {
  /**
   * Actual set of prepared row data to be rendered
   */
  rowsData: RunRowType[];

  /**
   * Helper data set with metric, param and tag keys
   */
  runsData: ExperimentRunsSelectorResult;

  experiments: ExperimentEntity[];
  searchFacetsState: ExperimentPageSearchFacetsState;
  viewState: ExperimentPageViewState;
  updateViewState: UpdateExperimentViewStateFn;
  isLoading: boolean;
  moreRunsAvailable: boolean;
  onAddColumnClicked: () => void;
  loadMoreRunsFunc: () => void;
  onDatasetSelected?: (dataset: RunDatasetWithTags, run: RunRowType) => void;
  expandRows: boolean;
  uiState: ExperimentPageUIState;
  compareRunsMode: ExperimentViewRunsCompareMode;
}

export const ExperimentViewRunsTable = React.memo(
  // eslint-disable-next-line complexity
  ({
    experiments,
    searchFacetsState,
    runsData,
    isLoading,
    moreRunsAvailable,
    updateViewState,
    onAddColumnClicked,
    rowsData,
    loadMoreRunsFunc,
    onDatasetSelected,
    expandRows,
    viewState,
    uiState,
    compareRunsMode,
  }: ExperimentViewRunsTableProps) => {
    const { theme } = useDesignSystemTheme();
    const updateUIState = useUpdateExperimentViewUIState();
    const setUrlSearchFacets = useUpdateExperimentPageSearchFacets();

    const { orderByKey, orderByAsc } = searchFacetsState;

    // If using new view state model, use `uiState` instead of `searchFacetsState`
    const {
      // Get relevant column and run info from persisted UI state
      selectedColumns,
      runsPinned,
      runsHidden,
      runListHidden,
    } = uiState;

    const isComparingRuns = compareRunsMode !== 'TABLE';

    // Determine if we should optimize by filtering columns based on count
    const shouldOptimize = useMemo(() => {
      const tagKeysCount = Utils.getVisibleTagKeyList(runsData.tagsList).length;
      const totalCount = runsData.metricKeyList.length + runsData.paramKeyList.length + tagKeysCount;
      return totalCount > LARGE_COLUMN_COUNT_THRESHOLD; // Only optimize when there are more than certain number of columns
    }, [runsData]);

    // Use the original data
    const { paramKeyList, metricKeyList, tagsList } = runsData;

    // Conditionally filter keys only when there are more than 1000 metrics+params+tags
    const { filteredMetricKeyList, filteredParamKeyList, filteredTagsList } = useMemo(() => {
      if (shouldOptimize && !isComparingRuns) {
        const filteredMetricKeyList: string[] = [];
        const filteredParamKeyList: string[] = [];
        const filteredTagsList: any[] = [];

        for (const column of selectedColumns) {
          if (column.startsWith(COLUMN_TYPES.METRICS)) {
            filteredMetricKeyList.push(extractCanonicalSortKey(column, COLUMN_TYPES.METRICS));
          } else if (column.startsWith(COLUMN_TYPES.PARAMS)) {
            filteredParamKeyList.push(extractCanonicalSortKey(column, COLUMN_TYPES.PARAMS));
          } else if (column.startsWith(COLUMN_TYPES.TAGS)) {
            const tagKey = extractCanonicalSortKey(column, COLUMN_TYPES.TAGS);
            filteredTagsList.push({
              [tagKey]: {
                key: tagKey,
                // value is unused
                value: null,
              },
            });
          }
        }

        return {
          filteredMetricKeyList,
          filteredParamKeyList,
          filteredTagsList,
        };
      }
      return {
        filteredMetricKeyList: metricKeyList,
        filteredParamKeyList: paramKeyList,
        filteredTagsList: tagsList,
      };
    }, [selectedColumns, shouldOptimize, isComparingRuns, metricKeyList, paramKeyList, tagsList]);

    const [gridApi, setGridApi] = useState<GridApi>();
    const [columnApi, setColumnApi] = useState<ColumnApi>();
    const prevSelectRunUuids = useRef<string[]>([]);

    const filteredTagKeys = useMemo(() => Utils.getVisibleTagKeyList(filteredTagsList), [filteredTagsList]);

    const containerElement = useRef<HTMLDivElement>(null);
    // Flag indicating if there are any rows that can be expanded
    const expandersVisible = useMemo(() => rowsData.some((row) => row.runDateAndNestInfo?.hasExpander), [rowsData]);

    const toggleRowExpanded = useCallback(
      (parentId: string) =>
        updateUIState(({ runsExpanded: currentRunsExpanded, ...state }: ExperimentPageUIState) => ({
          ...state,
          runsExpanded: { ...currentRunsExpanded, [parentId]: !currentRunsExpanded[parentId] },
        })),
      [updateUIState],
    );

    const togglePinnedRow = useCallback(
      (uuid: string) => {
        updateUIState((existingFacets: ExperimentPageUIState) => ({
          ...existingFacets,
          runsPinned: !existingFacets.runsPinned.includes(uuid)
            ? [...existingFacets.runsPinned, uuid]
            : existingFacets.runsPinned.filter((r) => r !== uuid),
        }));
      },
      [updateUIState],
    );

    // A modern version of row visibility toggle function, supports "show all", "show first n runs" options
    const toggleRowVisibility = useToggleRowVisibilityCallback(rowsData, uiState.useGroupedValuesInCharts);

    const gridReadyHandler = useCallback((params: GridReadyEvent) => {
      setGridApi(params.api);
      setColumnApi(params.columnApi);
    }, []);

    const { handleRowSelected, onSelectionChange } = useExperimentTableSelectRowHandler(updateViewState);

    const allRunsHidden = useMemo(() => {
      if (shouldUseRunRowsVisibilityMap()) {
        return rowsData.every((row) => row.hidden);
      }
      return runsData.runInfos.every(({ runUuid }) => runsHidden.includes(runUuid));
    }, [runsData, rowsData, runsHidden]);

    // Check if at least one run has custom visibility settings
    const usingCustomVisibility = shouldUseRunRowsVisibilityMap() && !isEmpty(uiState.runsVisibilityMap);

    const columnDefs = useRunsColumnDefinitions({
      selectedColumns,
      onExpand: toggleRowExpanded,
      compareExperiments: experiments.length > 1,
      onTogglePin: togglePinnedRow,
      onToggleVisibility: toggleRowVisibility,
      metricKeyList: filteredMetricKeyList,
      paramKeyList: filteredParamKeyList,
      tagKeyList: filteredTagKeys,
      columnApi,
      isComparingRuns,
      onDatasetSelected,
      expandRows,
      runsHiddenMode: uiState.runsHiddenMode,
    });

    const gridSizeHandler = useCallback(
      (api: GridApi) => {
        if (api && isComparingRuns) {
          try {
            api.sizeColumnsToFit();
          } catch {
            // ag-grid occasionally throws an error when trying to size columns while its internal ref is lost
            // We can't do much about it, so the exception is consumed
          }
        }
      },
      [isComparingRuns],
    );

    useEffect(() => {
      if (!gridApi) {
        return;
      }

      if (isLoading) {
        gridApi.showLoadingOverlay();
      } else {
        gridApi.hideOverlay();

        // If there are more runs available in the API, append
        // additional special row that will display "Load more" button
        if (rowsData.length && moreRunsAvailable) {
          gridApi.setRowData([...rowsData, createLoadMoreRow()]);
          gridSizeHandler(gridApi);
          return;
        }

        gridApi.setRowData(rowsData);
        gridSizeHandler(gridApi);
      }
    }, [gridApi, rowsData, isLoading, moreRunsAvailable, loadMoreRunsFunc, gridSizeHandler]);

    // Count all columns available for selection
    const allAvailableColumnsCount = useMemo(() => {
      const attributeColumnCount = getAdjustableAttributeColumns(experiments.length > 1).length;

      const valuesColumnCount = metricKeyList.length + paramKeyList.length + filteredTagKeys.length;

      return attributeColumnCount + valuesColumnCount;
    }, [experiments.length, filteredTagKeys.length, metricKeyList.length, paramKeyList.length]);

    const hasSelectedAllColumns = selectedColumns.length >= allAvailableColumnsCount;

    const moreAvailableRunsTableColumnCount = Math.max(0, allAvailableColumnsCount - selectedColumns.length);

    const allRunsCount = useMemo(
      () =>
        runsData.runInfos.filter(
          (r) => runsPinned.includes(r.runUuid) || runsData.runUuidsMatchingFilter.includes(r.runUuid),
        ).length,
      [runsData, runsPinned],
    );

    useLayoutEffect(() => {
      if (!gridApi) {
        return;
      }
      // Each time we switch to "compare runs" mode, we should
      // maximize columns so "run name" column will take up all remaining space
      if (isComparingRuns) {
        // Selection feature is not supported in compare runs mode so we should deselect all
        gridApi.deselectAll();
        gridApi.sizeColumnsToFit();
      }
      gridApi.resetRowHeights();
    }, [gridApi, isComparingRuns]);

    /**
     * Function used by ag-grid to calculate each row's height.
     * In this case, it's based on a datasets cell size.
     */

    const rowHeightGetterFn = useCallback(
      // if is comparing runs, use the default row height
      (row: { data: RunRowType }) => {
        if (isComparingRuns || !expandRows) {
          return EXPERIMENT_RUNS_TABLE_ROW_HEIGHT;
        }
        const datasetColumnId = makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DATASET);
        const datasetColumnShown = selectedColumns.includes(datasetColumnId);
        // if not comparing runs, use the datasets cell height
        return getDatasetsCellHeight(datasetColumnShown, row);
      },
      [selectedColumns, isComparingRuns, expandRows],
    );

    useEffect(() => {
      // Enabling certain columns (datasets) will change our row height calculation,
      // let's recalculate them
      gridApi?.resetRowHeights();
    }, [gridApi, selectedColumns, expandRows]);

    const [sidebarPreviewData, setSidebarPreviewData] = useState<{
      value: string;
      header: string;
    } | null>(null);

    const handleCellClicked = useCallback(
      ({ column, data, value }: CellClickedEvent) => {
        const columnGroupId = column.getParent()?.getGroupId();
        const shouldInvokePreviewSidebar =
          columnGroupId === COLUMN_TYPES.METRICS || columnGroupId === COLUMN_TYPES.PARAMS;

        if (shouldInvokePreviewSidebar) {
          setSidebarPreviewData({
            value,
            header: `Run name: ${data.runName}, Column name: ${column.getColDef().headerTooltip}`,
          });
          updateViewState({ previewPaneVisible: true });
        }
      },
      [updateViewState],
    );

    const displayAddColumnsCTA = !hasSelectedAllColumns && !isComparingRuns && rowsData.length > 0;
    const displayPreviewSidebar = !isComparingRuns && viewState.previewPaneVisible;
    const displayRunsTable = !runListHidden || !isComparingRuns;
    const displayStatusBar = !runListHidden;
    const displayEmptyState = rowsData.length < 1 && !isLoading && displayRunsTable;

    const tableContext = useMemo(() => ({ orderByAsc, orderByKey }), [orderByAsc, orderByKey]);

    const { cellMouseOverHandler, cellMouseOutHandler } = useRunsHighlightTableRow(containerElement);

    const gridStyles = useExperimentAgGridTableStyles();

    return (
      <div
        css={(theme) => ({
          display: 'grid',
          flex: 1,
          gridTemplateColumns: displayPreviewSidebar ? '1fr auto' : '1fr',
          borderTop: `1px solid ${theme.colors.border}`,
        })}
        className={isComparingRuns ? 'is-table-comparing-runs-mode' : undefined}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
          }}
        >
          <div
            ref={containerElement}
            className={cx('ag-theme-balham ag-grid-sticky', {
              'ag-grid-expanders-visible': expandersVisible,
              'is-table-comparing-runs-mode': isComparingRuns,
            })}
            css={[gridStyles, { display: displayRunsTable ? 'block' : 'hidden', height: '100%' }]}
            aria-hidden={!displayRunsTable}
          >
            <ExperimentViewRunsTableHeaderContextProvider
              runsHiddenMode={uiState.runsHiddenMode}
              usingCustomVisibility={usingCustomVisibility}
              useGroupedValuesInCharts={Boolean(uiState.groupBy) && uiState.useGroupedValuesInCharts}
              allRunsHidden={allRunsHidden}
            >
              <MLFlowAgGridLoader
                context={tableContext}
                defaultColDef={EXPERIMENTS_DEFAULT_COLUMN_SETUP}
                columnDefs={columnDefs}
                rowSelection="multiple"
                onGridReady={gridReadyHandler}
                onSelectionChanged={onSelectionChange}
                getRowHeight={rowHeightGetterFn}
                headerHeight={EXPERIMENT_RUNS_TABLE_ROW_HEIGHT}
                onRowSelected={handleRowSelected}
                suppressRowClickSelection
                suppressColumnMoveAnimation
                suppressScrollOnNewData
                isFullWidthRow={getRowIsLoadMore}
                fullWidthCellRenderer="LoadMoreRowRenderer"
                fullWidthCellRendererParams={{ loadMoreRunsFunc }}
                suppressFieldDotNotation
                enableCellTextSelection
                components={getFrameworkComponents()}
                suppressNoRowsOverlay
                loadingOverlayComponent="loadingOverlayComponent"
                loadingOverlayComponentParams={{ showImmediately: true }}
                getRowId={getRowId}
                rowBuffer={ROW_BUFFER}
                onCellClicked={handleCellClicked}
                onGridSizeChanged={({ api }) => gridSizeHandler(api)}
                onCellMouseOver={cellMouseOverHandler}
                onCellMouseOut={cellMouseOutHandler}
                maxBlocksInCache={20} // Increased from 10
                cacheBlockSize={100}
                maxConcurrentDatasourceRequests={2} // Increased from 1
                immutableData // Added for better performance
                getRowNodeId={(data) => data.rowUuid} // Added for better row identification
                suppressPropertyNamesCheck // Added to reduce overhead
                suppressAnimationFrame // Added to reduce rendering overhead
              />
            </ExperimentViewRunsTableHeaderContextProvider>
            {displayAddColumnsCTA && (
              <ExperimentViewRunsTableAddColumnCTA
                gridContainerElement={containerElement.current}
                isInitialized={Boolean(gridApi)}
                onClick={onAddColumnClicked}
                visible={!isLoading}
                moreRunsAvailable={moreRunsAvailable}
                moreAvailableRunsTableColumnCount={moreAvailableRunsTableColumnCount}
              />
            )}
          </div>
          {displayEmptyState && (
            <ExperimentViewRunsEmptyTable
              onClearFilters={() => {
                setUrlSearchFacets(createExperimentPageSearchFacetsState());
              }}
              isFiltered={isSearchFacetsFilterUsed(searchFacetsState)}
            />
          )}
          {displayStatusBar && <ExperimentViewRunsTableStatusBar allRunsCount={allRunsCount} isLoading={isLoading} />}
        </div>
        {displayPreviewSidebar && (
          <PreviewSidebar
            content={sidebarPreviewData?.value}
            copyText={sidebarPreviewData?.value}
            headerText={sidebarPreviewData?.header}
            onClose={() => updateViewState({ previewPaneVisible: false })}
            empty={
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="Select a cell to display preview"
                    description="Experiment page > table view > preview sidebar > nothing selected"
                  />
                }
              />
            }
          />
        )}
      </div>
    );
  },
);

/**
 * Concrete named definitions for colors used in this agGrid
 */
const getGridColors = (theme: Theme) => ({
  rowForeground: theme.colors.textPrimary, // regular row background
  rowBackground: theme.colors.backgroundPrimary, // regular row background
  rowBackgroundHover: theme.colors.tableBackgroundUnselectedHover,
  rowBackgroundSelected: theme.colors.tableBackgroundSelectedDefault,
  rowBackgroundHoverSelected: theme.colors.tableBackgroundSelectedHover,
  columnSortedBy: `${theme.colors.blue400}1F`,
  headerBackground: theme.colors.backgroundSecondary,
  headerTextColor: theme.colors.textSecondary, // directly from Figma design
  headerGroupTextColor: theme.colors.textSecondary, // directly from Figma design
  borderColor: theme.colors.borderDecorative, // border between header and content and scrollable areas
  headerBorderColor: 'transparent', // borders inside the header
  checkboxBorderColor: theme.colors.actionDefaultBorderDefault,
  checkboxBorderColorChecked: theme.colors.backgroundPrimary,
  checkboxBackgroundColorChecked: theme.colors.actionPrimaryBackgroundDefault,
  overlayBackground: `${theme.colors.backgroundSecondary}99`, // color for the loading overlay
});

const styles = {
  agGridOverrides: (theme: Theme, usingCustomHeaderComponent = true): CSSObject => {
    const gridColors = getGridColors(theme);
    return {
      height: '100%',
      position: 'relative',
      '&.ag-theme-balham': {
        // Set up internal variable values
        '--ag-border-color': gridColors.borderColor,
        '--ag-row-border-color': gridColors.borderColor,
        '--ag-foreground-color': gridColors.rowForeground,
        '--ag-background-color': gridColors.rowBackground,
        '--ag-odd-row-background-color': gridColors.rowBackground,
        '--ag-row-hover-color': gridColors.rowBackgroundHover,
        '--ag-selected-row-background-color': gridColors.rowBackgroundSelected,
        '--ag-header-foreground-color': gridColors.headerTextColor,
        '--ag-header-background-color': gridColors.headerBackground,
        '--ag-modal-overlay-background-color': gridColors.overlayBackground,

        // Makes row header sticky
        '&.ag-grid-sticky .ag-header': {
          position: 'sticky',
          top: 0,
          zIndex: 1,
        },
        '&.ag-grid-sticky .ag-root': {
          overflow: 'visible',
        },
        '&.ag-grid-sticky .ag-root-wrapper': {
          border: '0',
          borderRadius: '4px',
          overflow: 'visible',
        },

        // When scrollbars are forced to be visible in the system, ag-grid will sometimes
        // display a scrollbar that is not needed. We hide it here on compact mode.
        '.is-table-comparing-runs-mode & .ag-body-horizontal-scroll.ag-scrollbar-invisible': {
          display: 'none',
        },

        // Adds a static line between column group header row and column headers
        '.ag-header::after': {
          content: '""',
          position: 'absolute',
          top: EXPERIMENT_RUNS_TABLE_ROW_HEIGHT,
          left: 0,
          right: 0,
          height: 1,
          backgroundColor: gridColors.borderColor,
        },

        // Line height for cell contents is the row height minus the border
        '.ag-cell': {
          // lineHeight: `min(var(--ag-line-height, ${ROW_HEIGHT - 2}px), ${ROW_HEIGHT - 2}px)`,
          display: 'flex',
          overflow: 'hidden',
          '& > .ag-cell-wrapper': {
            overflow: 'hidden',
          },
        },

        // Padding fixes for the header (if custom header component is used)
        '.ag-header-cell': usingCustomHeaderComponent
          ? {
              padding: 0,
            }
          : undefined,

        '.ag-header-cell .ag-checkbox': usingCustomHeaderComponent
          ? {
              padding: '0 7px',
              borderLeft: '1px solid transparent', // to match it with the cell sizing
            }
          : undefined,

        '.ag-cell.is-ordered-by, .ag-header-cell > .is-ordered-by': {
          backgroundColor: gridColors.columnSortedBy,
        },
        '.ag-header-row': {
          '--ag-border-color': gridColors.headerBorderColor,
        },
        '.ag-header-row.ag-header-row-column-group': {
          '--ag-header-foreground-color': gridColors.headerGroupTextColor,
        },
        '.ag-row.ag-row-selected.ag-row-hover': {
          backgroundColor: gridColors.rowBackgroundHoverSelected,
        },
        '.ag-row.is-highlighted': {
          backgroundColor: gridColors.rowBackgroundHoverSelected,
        },
        // Hides resize guidelines when header is not hovered
        '.ag-header:not(:hover) .ag-header-cell::after, .ag-header:not(:hover) .ag-header-group-cell::after': {
          opacity: 0,
        },
        '.ag-pinned-left-header': {
          borderRight: 'none',
        },

        // Fixed for loading overlay, should be above "load more" button
        '.ag-overlay-loading-wrapper': {
          paddingTop: theme.spacing.md * 4,
          alignItems: 'center',
          zIndex: 2,
        },
        '.ag-overlay-loading-wrapper .ag-react-container': {
          flex: 1,
        },

        // Adds border after the last column to separate contents from "Add columns" CTA
        '.ag-center-cols-container': {
          minHeight: 0,
        },

        '.ag-full-width-row': {
          borderBottom: 0,
          backgroundColor: 'transparent',
          zIndex: 1,
          '&.ag-row-hover': {
            backgroundColor: 'transparent',
          },
        },

        // Centers vertically and styles the checkbox cell
        '.is-checkbox-cell': {
          display: 'flex',
          alignItems: 'center',
          paddingLeft: 7, // will end up in 8px due to 1px of transparent border on the left
          '.is-multiline-cell .ag-cell-value': {
            height: '100%',
          },
        },

        // Change appearance of the previewable cells
        '.is-previewable-cell': {
          cursor: 'pointer',
        },

        // Header checkbox cell will get the same background as header only if it's unchecked
        '.ag-header-cell .ag-checkbox .ag-input-wrapper:not(.ag-indeterminate):not(.ag-checked)': {
          '--ag-checkbox-background-color': gridColors.headerBackground,
        },

        // Distance from the checkbox to other icons (pin, visibility etc.)
        '.ag-cell-wrapper .ag-selection-checkbox': {
          marginRight: 20,
        },

        // Header and cell checkboxes will get same colors from the palette
        '.is-checkbox-cell, .ag-header-cell .ag-checkbox': {
          '.ag-checkbox-input-wrapper::after': {
            color: gridColors.checkboxBorderColor,
          },
          '.ag-checkbox-input-wrapper.ag-checked': {
            '--ag-checkbox-background-color': gridColors.checkboxBackgroundColorChecked,
            '--ag-checkbox-checked-color': gridColors.checkboxBorderColorChecked,
            '&::after': {
              color: gridColors.checkboxBorderColorChecked,
            },
          },
        },
      },
    };
  },
};

/**
 * Returns common styles to be used in ag-grid powered tables
 */
export const useExperimentAgGridTableStyles = ({
  usingCustomHeaderComponent = true,
}: { usingCustomHeaderComponent?: boolean } = {}) => {
  const { theme } = useDesignSystemTheme();
  return useMemo(() => styles.agGridOverrides(theme, usingCustomHeaderComponent), [theme, usingCustomHeaderComponent]);
};
