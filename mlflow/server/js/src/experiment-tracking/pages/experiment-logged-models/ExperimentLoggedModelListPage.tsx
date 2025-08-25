import { Alert, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { ExperimentLoggedModelPageWrapper } from './ExperimentLoggedModelPageWrapper';

import { isLoggedModelsFilteringAndSortingEnabled } from '../../../common/utils/FeatureUtils';
import { useEffect, useState } from 'react';
import { ExperimentLoggedModelListPageTable } from '../../components/experiment-logged-models/ExperimentLoggedModelListPageTable';
import { useSearchLoggedModelsQuery } from '../../hooks/logged-models/useSearchLoggedModelsQuery';
import { ExperimentLoggedModelListPageControls } from '../../components/experiment-logged-models/ExperimentLoggedModelListPageControls';
import {
  parseLoggedModelMetricOrderByColumnId,
  useExperimentLoggedModelListPageTableColumns,
} from '../../components/experiment-logged-models/hooks/useExperimentLoggedModelListPageTableColumns';
import { ExperimentLoggedModelOpenDatasetDetailsContextProvider } from '../../components/experiment-logged-models/hooks/useExperimentLoggedModelOpenDatasetDetails';
import { useLoggedModelsListPageState } from '../../components/experiment-logged-models/hooks/useLoggedModelsListPagePageState';
import { useRelatedRunsDataForLoggedModels } from '../../hooks/logged-models/useRelatedRunsDataForLoggedModels';
import {
  ExperimentLoggedModelListPageMode,
  useExperimentLoggedModelListPageMode,
} from '../../components/experiment-logged-models/hooks/useExperimentLoggedModelListPageMode';
import { ExperimentViewRunsTableResizer } from '../../components/experiment-page/components/runs/ExperimentViewRunsTableResizer';
import { ExperimentLoggedModelListCharts } from '../../components/experiment-logged-models/ExperimentLoggedModelListCharts';
import { ExperimentLoggedModelListPageRowVisibilityContextProvider } from '../../components/experiment-logged-models/hooks/useExperimentLoggedModelListPageRowVisibility';
import { RunsChartsSetHighlightContextProvider } from '../../components/runs-charts/hooks/useRunsChartTraceHighlight';
import { BadRequestError } from '@databricks/web-shared/errors';
import { useResizableMaxWidth } from '@mlflow/mlflow/src/shared/web-shared/hooks/useResizableMaxWidth';

const INITIAL_RUN_COLUMN_SIZE = 295;
const CHARTS_MIN_WIDTH = 350;

const ExperimentLoggedModelListPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const sortingAndFilteringEnabled = isLoggedModelsFilteringAndSortingEnabled();

  const {
    state: {
      orderByColumn,
      orderByAsc,
      columnVisibility,
      rowVisibilityMap,
      rowVisibilityMode,
      selectedFilterDatasets,
      groupBy,
    },
    searchQuery,
    isFilteringActive,
    setOrderBy,
    setColumnVisibility,
    setRowVisibilityMode,
    toggleRowVisibility,
    updateSearchQuery,
    toggleDataset,
    clearSelectedDatasets,
    setGroupBy,
  } = useLoggedModelsListPageState();

  invariant(experimentId, 'Experiment ID must be defined');

  const { viewMode, setViewMode } = useExperimentLoggedModelListPageMode();

  // Translate currently sorting column to the format accepted by the API query.
  // If the column is a metric, we need to parse and pass the dataset name and digest if found.
  const getOrderByRequestData = () => {
    if (!orderByColumn) {
      return { orderByField: undefined };
    }
    const parsedMetricOrderByColumn = parseLoggedModelMetricOrderByColumnId(orderByColumn);
    if (parsedMetricOrderByColumn.datasetDigest && parsedMetricOrderByColumn.datasetName) {
      return {
        orderByField: `metrics.${parsedMetricOrderByColumn.metricKey}`,
        orderByDatasetName: parsedMetricOrderByColumn.datasetName,
        orderByDatasetDigest: parsedMetricOrderByColumn.datasetDigest,
      };
    }
    return { orderByField: orderByColumn };
  };

  const {
    data: loggedModels,
    isFetching: isFetchingLoggedModels,
    isLoading: isLoadingLoggedModels,
    error: loggedModelsError,
    nextPageToken,
    loadMoreResults,
  } = useSearchLoggedModelsQuery({
    experimentIds: [experimentId],
    orderByAsc,
    searchQuery,
    selectedFilterDatasets,
    ...getOrderByRequestData(),
  });

  // Find and extract 400 error from the logged models error
  const badRequestError = loggedModelsError instanceof BadRequestError ? loggedModelsError : undefined;

  const { data: relatedRunsData } = useRelatedRunsDataForLoggedModels({ loggedModels });

  const { columnDefs, compactColumnDefs } = useExperimentLoggedModelListPageTableColumns({
    loggedModels,
    columnVisibility,
    isLoading: isLoadingLoggedModels,
    orderByColumn,
    orderByAsc,
    enableSortingByMetrics: sortingAndFilteringEnabled,
  });

  const [tableAreaWidth, setTableAreaWidth] = useState<number>(INITIAL_RUN_COLUMN_SIZE);
  const [tableHidden, setTableHidden] = useState(false);

  const isCompactTableMode = viewMode !== ExperimentLoggedModelListPageMode.TABLE;

  const tableElement =
    isCompactTableMode && tableHidden ? (
      <div css={{ width: theme.spacing.md }} />
    ) : (
      <ExperimentLoggedModelListPageTable
        columnDefs={isCompactTableMode ? compactColumnDefs : columnDefs}
        loggedModels={loggedModels ?? []}
        isLoading={isLoadingLoggedModels}
        isLoadingMore={isFetchingLoggedModels}
        badRequestError={badRequestError}
        moreResultsAvailable={Boolean(nextPageToken)}
        onLoadMore={loadMoreResults}
        onOrderByChange={setOrderBy}
        orderByAsc={orderByAsc}
        orderByColumn={orderByColumn}
        columnVisibility={columnVisibility}
        relatedRunsData={relatedRunsData}
        isFilteringActive={isFilteringActive}
        groupModelsBy={groupBy}
      />
    );

  const { resizableMaxWidth, ref } = useResizableMaxWidth(CHARTS_MIN_WIDTH);

  return (
    <ExperimentLoggedModelOpenDatasetDetailsContextProvider>
      <ExperimentLoggedModelListPageRowVisibilityContextProvider
        visibilityMap={rowVisibilityMap}
        visibilityMode={rowVisibilityMode}
        setRowVisibilityMode={setRowVisibilityMode}
        toggleRowVisibility={toggleRowVisibility}
      >
        <ExperimentLoggedModelListPageControls
          columnDefs={columnDefs}
          columnVisibility={columnVisibility}
          onChangeOrderBy={setOrderBy}
          onUpdateColumns={setColumnVisibility}
          orderByColumn={orderByColumn}
          orderByAsc={orderByAsc}
          viewMode={viewMode}
          setViewMode={setViewMode}
          searchQuery={searchQuery}
          onChangeSearchQuery={updateSearchQuery}
          loggedModelsData={loggedModels ?? []}
          sortingAndFilteringEnabled={sortingAndFilteringEnabled}
          selectedFilterDatasets={selectedFilterDatasets}
          onToggleDataset={toggleDataset}
          onClearSelectedDatasets={clearSelectedDatasets}
          groupBy={groupBy}
          onChangeGroupBy={setGroupBy}
        />
        <Spacer size="sm" shrinks={false} />
        {/* Display error message, but not if it's 400 - in that case, the error message is displayed in the table */}
        {loggedModelsError?.message && !badRequestError && (
          <>
            <Alert
              componentId="mlflow.logged_models.list.error"
              message={loggedModelsError.message}
              type="error"
              closable={false}
            />
            <Spacer size="sm" shrinks={false} />
          </>
        )}
        {isCompactTableMode ? (
          <RunsChartsSetHighlightContextProvider>
            <div ref={ref} css={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>
              <ExperimentViewRunsTableResizer
                onResize={setTableAreaWidth}
                runListHidden={tableHidden}
                width={tableAreaWidth}
                onHiddenChange={setTableHidden}
                maxWidth={resizableMaxWidth}
              >
                {tableElement}
              </ExperimentViewRunsTableResizer>
              {viewMode === ExperimentLoggedModelListPageMode.CHART && (
                <ExperimentLoggedModelListCharts
                  loggedModels={loggedModels ?? []}
                  experimentId={experimentId}
                  minWidth={CHARTS_MIN_WIDTH}
                />
              )}
            </div>
          </RunsChartsSetHighlightContextProvider>
        ) : (
          tableElement
        )}
      </ExperimentLoggedModelListPageRowVisibilityContextProvider>
    </ExperimentLoggedModelOpenDatasetDetailsContextProvider>
  );
};

const ExperimentLoggedModelListPage = () => (
  <ExperimentLoggedModelPageWrapper>
    <ExperimentLoggedModelListPageImpl />
  </ExperimentLoggedModelPageWrapper>
);

export default ExperimentLoggedModelListPage;
