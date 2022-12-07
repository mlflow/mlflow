import {
  ColumnApi,
  GridApi,
  GridReadyEvent,
  RowSelectedEvent,
  SelectionChangedEvent,
} from '@ag-grid-community/core';
import { Theme } from '@emotion/react';
import cx from 'classnames';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { MLFlowAgGridLoader } from '../../../../../common/components/ag-grid/AgGridLoader';
import { ExperimentRunsTableEmptyOverlay } from '../../../../../common/components/ExperimentRunsTableEmptyOverlay';
import Utils from '../../../../../common/utils/Utils';
import { ATTRIBUTE_COLUMN_SORT_KEY, COLUMN_TYPES } from '../../../../constants';
import {
  ExperimentEntity,
  UpdateExperimentSearchFacetsFn,
  UpdateExperimentViewStateFn,
} from '../../../../types';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import {
  ADJUSTABLE_ATTRIBUTE_COLUMNS,
  ADJUSTABLE_ATTRIBUTE_COLUMNS_SINGLE_EXPERIMENT,
  EXPERIMENTS_DEFAULT_COLUMN_SETUP,
  getFrameworkComponents,
  getRowId,
  isCanonicalSortKeyOfType,
  useRunsColumnDefinitions,
} from '../../utils/experimentPage.column-utils';
import { RunRowType } from '../../utils/experimentPage.row-types';
import { prepareRunsGridData } from '../../utils/experimentPage.row-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunsTableAddColumnCTA } from './ExperimentViewRunsTableAddColumnCTA';

export interface ExperimentViewRunsTableProps {
  experiments: ExperimentEntity[];
  searchFacetsState: SearchExperimentRunsFacetsState;
  runsData: ExperimentRunsSelectorResult;
  viewState: SearchExperimentRunsViewState;
  updateViewState: UpdateExperimentViewStateFn;
  isLoading: boolean;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  onAddColumnClicked: () => void;
}

/**
 * Creates time with milliseconds set to zero, usable in calculating
 * relative time
 */
const createCurrentTime = () => {
  const mountTime = new Date();
  mountTime.setMilliseconds(0);
  return mountTime;
};

export const ExperimentViewRunsTable = React.memo(
  ({
    experiments,
    searchFacetsState,
    runsData,
    isLoading,
    updateSearchFacets,
    updateViewState,
    onAddColumnClicked,
  }: ExperimentViewRunsTableProps) => {
    const { runsExpanded, orderByKey, searchFilter, runsPinned } = searchFacetsState;
    const {
      paramKeyList,
      metricKeyList,
      tagsList,
      modelVersionsByRunUuid,
      paramsList,
      metricsList,
      runInfos,
      runUuidsMatchingFilter,
    } = runsData;

    const [gridApi, setGridApi] = useState<GridApi>();
    const [columnApi, setColumnApi] = useState<ColumnApi>();
    const prevSelectRunUuids = useRef<string[]>([]);
    const [runsCount, setRunsCount] = useState(runInfos.length);
    // Flag indicating if there are any rows that can be expanded
    const [expandersVisible, setExpandersVisible] = useState(false);

    const filteredTagKeys = useMemo(() => Utils.getVisibleTagKeyList(tagsList), [tagsList]);

    const containerElement = useRef<HTMLDivElement>(null);

    useEffect(() => {
      if (!gridApi) {
        return;
      }

      if (isLoading) {
        gridApi.showLoadingOverlay();
      } else {
        gridApi.hideOverlay();
      }
    }, [gridApi, isLoading]);

    /**
     * Updates selected rows in the view state
     */
    const onSelectionChange = useCallback(
      ({ api }: SelectionChangedEvent) => {
        const selectedUUIDs: string[] = api
          .getSelectedRows()
          .map(({ runInfo }) => runInfo.run_uuid);
        updateViewState({
          runsSelected: selectedUUIDs.reduce(
            (aggregate, curr) => ({ ...aggregate, [curr]: true }),
            {},
          ),
        });
        prevSelectRunUuids.current = selectedUUIDs;
      },
      [updateViewState],
    );

    /**
     * A onRowSelected event handler that runs before onSelectionChange.
     * It checks if the currently (de)selected row contains any children
     * and if true, (de)select them as well.
     */
    const handleRowSelected = useCallback((event: RowSelectedEvent) => {
      const selectedRows = event.api.getSelectedRows();

      // Let's check if the actual number of selected rows have changed
      // to avoid empty runs
      if (prevSelectRunUuids.current && selectedRows.length !== prevSelectRunUuids.current.length) {
        const isSelected = Boolean(event.node.isSelected());

        // We will continue only if the selected row has properly set runDateInfo
        const { runDateAndNestInfo } = event.data as RunRowType;
        if (!runDateAndNestInfo) {
          return;
        }
        const { isParent, expanderOpen, childrenIds } = runDateAndNestInfo;

        // We will continue only if the selected row is a parent containing
        // children and is actually expanded
        if (isParent && expanderOpen && childrenIds) {
          const childrenIdsToSelect = childrenIds;

          event.api.forEachNode((node) => {
            const { runInfo, runDateAndNestInfo: childRunDateInfo } = node.data as RunRowType;

            const childrenRunUuid = runInfo.run_uuid;
            if (childrenIdsToSelect.includes(childrenRunUuid)) {
              // If we found children being parents, mark their children
              // to be selected as well.
              if (childRunDateInfo?.childrenIds) {
                childrenIdsToSelect.push(...childRunDateInfo.childrenIds);
              }

              node.setSelected(isSelected, false, true);
            }
          });
        }
      }
    }, []);

    const onSortBy = useCallback(
      (newOrderByKey: string, newOrderByAsc: boolean) => {
        updateSearchFacets({ orderByKey: newOrderByKey, orderByAsc: newOrderByAsc });
      },
      [updateSearchFacets],
    );

    const toggleRowExpanded = useCallback(
      (parentId: string) =>
        updateSearchFacets(({ runsExpanded: currentRunsExpanded, ...state }) => ({
          ...state,
          runsExpanded: { ...currentRunsExpanded, [parentId]: !currentRunsExpanded[parentId] },
        })),
      [updateSearchFacets],
    );

    const shouldNestChildrenAndFetchParents = useMemo(
      () => (!orderByKey && !searchFilter) || orderByKey === ATTRIBUTE_COLUMN_SORT_KEY.DATE,
      [orderByKey, searchFilter],
    );

    // Value used a reference for the "date" column
    const [referenceTime, setReferenceTime] = useState(createCurrentTime);

    // We're setting new reference date only when new runs data package has arrived
    useEffect(() => {
      setReferenceTime(createCurrentTime);
    }, [runInfos]);

    useEffect(() => {
      if (!gridApi || isLoading) {
        return;
      }
      const data = prepareRunsGridData({
        experiments,
        paramKeyList,
        metricKeyList,
        modelVersionsByRunUuid,
        runsExpanded,
        tagKeyList: filteredTagKeys,
        nestChildren: shouldNestChildrenAndFetchParents,
        referenceTime,
        runData: runInfos.map((runInfo, index) => ({
          runInfo,
          params: paramsList[index],
          metrics: metricsList[index],
          tags: tagsList[index],
        })),
        runUuidsMatchingFilter,
        runsPinned,
      });
      gridApi.setRowData(data);

      setRunsCount(data.length);
      setExpandersVisible(data.some((row) => row.runDateAndNestInfo?.hasExpander));
    }, [
      gridApi,
      isLoading,
      experiments,
      metricKeyList,
      metricsList,
      modelVersionsByRunUuid,
      paramKeyList,
      paramsList,
      runInfos,
      runsExpanded,
      tagsList,
      filteredTagKeys,
      shouldNestChildrenAndFetchParents,
      referenceTime,
      runsPinned,
      runUuidsMatchingFilter,
    ]);

    const togglePinnedRow = useCallback(
      (uuid: string) => {
        updateSearchFacets((existingFacets) => ({
          ...existingFacets,
          runsPinned: !existingFacets.runsPinned.includes(uuid)
            ? [...existingFacets.runsPinned, uuid]
            : existingFacets.runsPinned.filter((r) => r !== uuid),
        }));
        // In the next frame, redraw the toggled row in to update the hover state
        // and tooltips so they won't dangle in the previous mouse position.
        requestAnimationFrame(() => {
          if (!gridApi) {
            return;
          }
          const rowNode = gridApi.getRowNode(uuid);
          if (rowNode) {
            gridApi.redrawRows({ rowNodes: [rowNode] });
          }
        });
      },
      [gridApi, updateSearchFacets],
    );

    const columnDefs = useRunsColumnDefinitions({
      searchFacetsState,
      onSortBy,
      onExpand: toggleRowExpanded,
      compareExperiments: experiments.length > 1,
      onTogglePin: togglePinnedRow,
      metricKeyList,
      paramKeyList,
      tagKeyList: filteredTagKeys,
      columnApi,
    });

    const gridReadyHandler = useCallback((params: GridReadyEvent) => {
      setGridApi(params.api);
      setColumnApi(params.columnApi);
    }, []);

    // Count all columns available for selection
    const allAvailableColumnsCount = useMemo(() => {
      const attributeColumnCount =
        experiments.length > 1
          ? ADJUSTABLE_ATTRIBUTE_COLUMNS.length
          : ADJUSTABLE_ATTRIBUTE_COLUMNS_SINGLE_EXPERIMENT.length;

      const valuesColumnCount = metricKeyList.length + paramKeyList.length + filteredTagKeys.length;

      return attributeColumnCount + valuesColumnCount;
    }, [experiments.length, filteredTagKeys.length, metricKeyList.length, paramKeyList.length]);

    const hasSelectedAllColumns =
      searchFacetsState.selectedColumns.length >= allAvailableColumnsCount;

    // Count metrics and params columns that were not selected yet so it can be displayed in CTA
    const moreAvailableParamsAndMetricsColumns = useMemo(() => {
      const selectedMetricsAndParamsColumns = searchFacetsState.selectedColumns.filter(
        (s) =>
          isCanonicalSortKeyOfType(s, COLUMN_TYPES.METRICS) ||
          isCanonicalSortKeyOfType(s, COLUMN_TYPES.PARAMS),
      ).length;

      const allMetricsAndParamsColumns = metricKeyList.length + paramKeyList.length;

      return Math.max(0, allMetricsAndParamsColumns - selectedMetricsAndParamsColumns);
    }, [metricKeyList.length, paramKeyList.length, searchFacetsState.selectedColumns]);

    return (
      <>
        <div css={styles.runsCount}>
          <FormattedMessage
            // eslint-disable-next-line max-len
            defaultMessage='Showing {length} matching {length, plural, =0 {runs} =1 {run} other {runs}}'
            // eslint-disable-next-line max-len
            description='Message for displaying how many runs match search criteria on experiment page'
            values={{ length: runsCount }}
          />
        </div>
        <div
          ref={containerElement}
          className={cx('ag-theme-balham ag-grid-sticky', {
            'ag-grid-expanders-visible': expandersVisible,
          })}
          css={styles.agGridOverrides}
        >
          <MLFlowAgGridLoader
            defaultColDef={EXPERIMENTS_DEFAULT_COLUMN_SETUP}
            columnDefs={columnDefs}
            domLayout='autoHeight'
            rowSelection='multiple'
            onGridReady={gridReadyHandler}
            onSelectionChanged={onSelectionChange}
            onRowSelected={handleRowSelected}
            suppressRowClickSelection
            suppressColumnMoveAnimation
            suppressScrollOnNewData
            suppressFieldDotNotation
            enableCellTextSelection
            components={getFrameworkComponents()}
            suppressNoRowsOverlay
            loadingOverlayComponent='loadingOverlayComponent'
            loadingOverlayComponentParams={{ showImmediately: true }}
            getRowId={getRowId}
          />
          {!hasSelectedAllColumns && (
            <ExperimentViewRunsTableAddColumnCTA
              gridContainerElement={containerElement.current}
              isInitialized={Boolean(gridApi)}
              onClick={onAddColumnClicked}
              visible={!isLoading}
              moreAvailableParamsAndMetricsColumnCount={moreAvailableParamsAndMetricsColumns}
            />
          )}
        </div>
        {runsCount < 1 && !isLoading && (
          <div css={styles.noResultsWrapper}>
            <ExperimentRunsTableEmptyOverlay />
          </div>
        )}
      </>
    );
  },
);

const styles = {
  runsCount: (theme: Theme) => ({ margin: `${theme.spacing.md}px 0` }),
  noResultsWrapper: (theme: Theme) => ({
    marginTop: -theme.spacing.md * 4,
    textAlign: 'center' as const,
    backgroundColor: theme.colors.backgroundPrimary,
    position: 'relative' as const,
  }),
  agGridOverrides: (theme: Theme) => ({
    marginTop: 12,
    position: 'relative' as const,
    '&.ag-theme-balham': {
      '--ag-border-color': 'rgba(0, 0, 0, 0.06)',
      '--ag-header-foreground-color': '#20272e',
      '--ag-header-background-color': `${theme.colors.grey100}`,
      '--ag-row-hover-color': `${theme.colors.grey200}`,
      '&.ag-grid-sticky .ag-header': {
        position: 'sticky' as const,
        top: 0,
        zIndex: 1,
      },
      '&.ag-grid-sticky .ag-root': {
        overflow: 'visible' as const,
      },
      '&.ag-grid-sticky .ag-root-wrapper': {
        border: '0',
        borderRadius: '4px',
        overflow: 'visible' as const,
      },
      '.ag-cell.is-ordered-by, .ag-header-cell > .is-ordered-by': {
        backgroundColor: theme.colors.blue100,
      },
      '.ag-header-cell': {
        padding: 0,
      },
      '.ag-header-cell .ag-checkbox': {
        padding: '0 12px',
      },
      '.ag-overlay-loading-wrapper': {
        paddingTop: theme.spacing.md * 4,
        alignItems: 'center' as const,
      },
      '.ag-overlay-loading-wrapper .ag-react-container': {
        flex: 1,
        zIndex: 1,
      },
      '.ag-layout-auto-height .ag-center-cols-container': {
        borderRight: `1px solid ${theme.colors.border}`,
        minHeight: 0,
      },
    },
  }),
};
