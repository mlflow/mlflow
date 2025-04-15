import { Alert, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import invariant from 'invariant';
import { useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { ExperimentLoggedModelPageWrapper } from './ExperimentLoggedModelPageWrapper';

import { isExperimentLoggedModelsUIEnabled } from '../../../common/utils/FeatureUtils';
import { useEffect, useState } from 'react';
import { ExperimentLoggedModelListPageTable } from '../../components/experiment-logged-models/ExperimentLoggedModelListPageTable';
import { useSearchLoggedModelsQuery } from '../../hooks/logged-models/useSearchLoggedModelsQuery';
import { ExperimentLoggedModelListPageControls } from '../../components/experiment-logged-models/ExperimentLoggedModelListPageControls';
import { useExperimentLoggedModelListPageTableColumns } from '../../components/experiment-logged-models/hooks/useExperimentLoggedModelListPageTableColumns';
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
import { RUNS_VISIBILITY_MODE } from '../../components/experiment-page/models/ExperimentPageUIState';
import { RunsChartsSetHighlightContextProvider } from '../../components/runs-charts/hooks/useRunsChartTraceHighlight';

const INITIAL_RUN_COLUMN_SIZE = 295;

const ExperimentLoggedModelListPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();

  const {
    state: { orderByField, orderByAsc, columnVisibility, rowVisibilityMap, rowVisibilityMode },
    setOrderBy,
    setColumnVisibility,
    setRowVisibilityMode,
    toggleRowVisibility,
  } = useLoggedModelsListPageState();

  invariant(experimentId, 'Experiment ID must be defined');

  useEffect(() => {
    if (!isExperimentLoggedModelsUIEnabled() && experimentId) {
      navigate(Routes.getExperimentPageRoute(experimentId), { replace: true });
    }
  }, [experimentId, navigate]);

  const { viewMode, setViewMode } = useExperimentLoggedModelListPageMode();

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
    orderByField,
  });

  const { data: relatedRunsData } = useRelatedRunsDataForLoggedModels({ loggedModels });

  const columnDefs = useExperimentLoggedModelListPageTableColumns({
    loggedModels,
    columnVisibility,
    isCompactMode: viewMode !== ExperimentLoggedModelListPageMode.TABLE,
  });

  const [tableAreaWidth, setTableAreaWidth] = useState<number>(INITIAL_RUN_COLUMN_SIZE);
  const [tableHidden, setTableHidden] = useState(false);

  const isCompactTableMode = viewMode !== ExperimentLoggedModelListPageMode.TABLE;

  const tableElement =
    isCompactTableMode && tableHidden ? (
      <div css={{ width: theme.spacing.md }} />
    ) : (
      <ExperimentLoggedModelListPageTable
        columnDefs={columnDefs}
        loggedModels={loggedModels ?? []}
        isLoading={isLoadingLoggedModels}
        isLoadingMore={isFetchingLoggedModels}
        error={loggedModelsError}
        moreResultsAvailable={Boolean(nextPageToken)}
        onLoadMore={loadMoreResults}
        onOrderByChange={setOrderBy}
        orderByAsc={orderByAsc}
        orderByField={orderByField}
        columnVisibility={columnVisibility}
        relatedRunsData={relatedRunsData}
      />
    );

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
          orderByField={orderByField}
          orderByAsc={orderByAsc}
          viewMode={viewMode}
          setViewMode={setViewMode}
        />
        <Spacer size="sm" shrinks={false} />
        {loggedModelsError?.message && (
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
            <div css={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>
              <ExperimentViewRunsTableResizer
                onResize={setTableAreaWidth}
                runListHidden={tableHidden}
                width={tableAreaWidth}
                onHiddenChange={setTableHidden}
              >
                {tableElement}
              </ExperimentViewRunsTableResizer>
              {viewMode === ExperimentLoggedModelListPageMode.CHART && (
                <ExperimentLoggedModelListCharts loggedModels={loggedModels ?? []} experimentId={experimentId} />
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
