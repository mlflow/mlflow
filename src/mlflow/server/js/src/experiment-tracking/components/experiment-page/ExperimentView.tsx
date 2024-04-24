import { LegacySkeleton } from '@databricks/design-system';

import { useEffect, useMemo, useState } from 'react';
import { ErrorCodes } from '../../../common/constants';
import NotFoundPage from '../NotFoundPage';
import { PermissionDeniedView } from '../PermissionDeniedView';
import { ExperimentViewDescriptions } from './components/ExperimentViewDescriptions';
import { ExperimentViewNotes } from './components/ExperimentViewNotes';
import { ExperimentViewHeader } from './components/header/ExperimentViewHeader';
import { ExperimentViewHeaderCompare } from './components/header/ExperimentViewHeaderCompare';
import { ExperimentViewRuns, ExperimentViewRunsImpl } from './components/runs/ExperimentViewRuns';
import { useExperimentIds } from './hooks/useExperimentIds';
import { useExperiments } from './hooks/useExperiments';
import { useFetchExperiments } from './hooks/useFetchExperiments';
import { useElementHeight } from '../../../common/utils/useElementHeight';
import { searchDatasetsApi } from '../../actions';
import Utils from '../../../common/utils/Utils';
import { ExperimentPageUIStateContextProvider } from './contexts/ExperimentPageUIStateContext';
import { first } from 'lodash';
import {
  shouldEnableExperimentPageCompactHeader,
  shouldEnableShareExperimentViewByTags,
} from '../../../common/utils/FeatureUtils';
import { useExperimentPageSearchFacets } from './hooks/useExperimentPageSearchFacets';
import { SearchExperimentRunsFacetsState } from './models/SearchExperimentRunsFacetsState';
import { usePersistExperimentPageViewState } from './hooks/usePersistExperimentPageViewState';
import { useDispatch, useSelector } from 'react-redux';
import { ThunkDispatch } from '../../../redux-types';
import { useExperimentRuns } from './hooks/useExperimentRuns';
import { ExperimentRunsSelectorResult } from './utils/experimentRuns.selector';
import { useSharedExperimentViewState } from './hooks/useSharedExperimentViewState';
import { useInitializeUIState } from './hooks/useInitializeUIState';
import { ExperimentViewDescriptionNotes } from './components/ExperimentViewDescriptionNotes';
import { ExperimentViewHeaderV2 } from './components/header/ExperimentViewHeaderV2';

export const ExperimentView = () => {
  const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
  const dispatch = useDispatch<ThunkDispatch>();

  const experimentIdsV1 = useExperimentIds();
  const [searchFacets, experimentIdsV2] = useExperimentPageSearchFacets();

  const experimentIds = usingNewViewStateModel ? experimentIdsV2 : experimentIdsV1;

  const experiments = useExperiments(experimentIds);

  const [firstExperiment] = experiments;

  const { fetchExperiments, isLoadingExperiment, requestError } = useFetchExperiments();

  const { elementHeight: hideableElementHeight, observeHeight } = useElementHeight();

  const [isMaximizedV1, setIsMaximizedV1] = useState(false);

  const [editing, setEditing] = useState(false);

  const [showAddDescriptionButton, setShowAddDescriptionButton] = useState(true);

  // Create new version of the UI state for the experiment page on this level
  const [uiState, setUIState, seedInitialUIState] = useInitializeUIState(experimentIds);

  const { isViewStateShared } = useSharedExperimentViewState(setUIState, first(experiments), !usingNewViewStateModel);

  // Get the maximized state from the new view state model if flag is set
  const isMaximized = usingNewViewStateModel ? uiState.viewMaximized : isMaximizedV1;
  // In the new version, toggle button manipulates the UI state directly so noop function is provided
  const setIsMaximized = usingNewViewStateModel ? () => {} : setIsMaximizedV1;

  const {
    isLoadingRuns,
    loadMoreRuns,
    runsData,
    moreRunsAvailable,
    requestError: runsRequestError,
    refreshRuns,
  } = useExperimentRuns(uiState, searchFacets, experimentIds, !usingNewViewStateModel);

  useEffect(() => {
    fetchExperiments(experimentIds);
  }, [fetchExperiments, experimentIds]);

  useEffect(() => {
    // Seed the initial UI state when the experiments and runs are loaded.
    // Should only run once.
    seedInitialUIState(experiments, runsData);
  }, [seedInitialUIState, experiments, runsData]);

  useEffect(() => {
    const requestAction = searchDatasetsApi(experimentIds);
    dispatch(requestAction).catch((e) => {
      Utils.logErrorAndNotifyUser(e);
    });
  }, [dispatch, experimentIds]);

  const isComparingExperiments = experimentIds.length > 1;

  usePersistExperimentPageViewState(uiState, searchFacets, experimentIds, isViewStateShared);

  const isViewInitialized = Boolean(!isLoadingExperiment && experiments[0] && runsData && searchFacets);

  if (usingNewViewStateModel && !isViewInitialized) {
    // In the new view state model, wait for search facets to initialize
    return <LegacySkeleton />;
  }

  if (requestError && requestError.getErrorCode() === ErrorCodes.PERMISSION_DENIED) {
    return <PermissionDeniedView errorMessage={requestError.getMessageField()} />;
  }

  if (requestError && requestError.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
    return <NotFoundPage />;
  }

  const isLoading = isLoadingExperiment || !experiments[0];

  const renderExperimentHeader = () =>
    shouldEnableExperimentPageCompactHeader() ? (
      <>
        <ExperimentViewHeaderV2
          experiment={firstExperiment}
          searchFacetsState={searchFacets || undefined}
          uiState={uiState}
          showAddDescriptionButton={showAddDescriptionButton}
          setEditing={setEditing}
        />
        <div
          style={{
            maxHeight: isMaximized ? 0 : hideableElementHeight,
          }}
          css={{ overflowY: 'hidden', flexShrink: 0, transition: 'max-height .12s' }}
        >
          <div ref={observeHeight}>
            <ExperimentViewDescriptionNotes
              experiment={firstExperiment}
              setShowAddDescriptionButton={setShowAddDescriptionButton}
              editing={editing}
              setEditing={setEditing}
            />
          </div>
        </div>
      </>
    ) : (
      <>
        <ExperimentViewHeader
          experiment={firstExperiment}
          searchFacetsState={searchFacets || undefined}
          uiState={uiState}
        />
        <div
          style={{
            maxHeight: isMaximized ? 0 : hideableElementHeight,
          }}
          css={{ overflowY: 'hidden', flexShrink: 0, transition: 'max-height .12s' }}
        >
          <div ref={observeHeight}>
            <ExperimentViewDescriptions experiment={firstExperiment} />
            <ExperimentViewNotes experiment={firstExperiment} />
          </div>
        </div>
      </>
    );

  return (
    <ExperimentPageUIStateContextProvider setUIState={setUIState}>
      <div css={styles.experimentViewWrapper}>
        {isLoading ? (
          <LegacySkeleton title paragraph={false} active />
        ) : (
          <>
            {isComparingExperiments ? (
              <ExperimentViewHeaderCompare experiments={experiments} />
            ) : (
              renderExperimentHeader()
            )}
          </>
        )}

        {!usingNewViewStateModel ? (
          <ExperimentViewRuns
            experiments={experiments}
            isLoading={isLoading}
            // We don't keep the view state on this level to maximize <ExperimentViewRuns>'s performance
            onMaximizedChange={setIsMaximized}
            searchFacetsState={searchFacets as SearchExperimentRunsFacetsState}
            uiState={uiState}
          />
        ) : (
          <ExperimentViewRunsImpl
            isLoading={false}
            experiments={experiments}
            isLoadingRuns={isLoadingRuns}
            runsData={runsData as ExperimentRunsSelectorResult}
            searchFacetsState={searchFacets as SearchExperimentRunsFacetsState}
            loadMoreRuns={loadMoreRuns}
            moreRunsAvailable={moreRunsAvailable}
            isPristine={() => false}
            requestError={runsRequestError}
            refreshRuns={refreshRuns}
            uiState={uiState}
          />
        )}
      </div>
    </ExperimentPageUIStateContextProvider>
  );
};

const styles = {
  experimentViewWrapper: { height: '100%', display: 'flex', flexDirection: 'column' as const },
};

export default ExperimentView;
