import { LegacySkeleton } from '@databricks/design-system';

import { useEffect, useMemo, useState } from 'react';
import { ErrorCodes } from '../../../common/constants';
import NotFoundPage from '../NotFoundPage';
import { PermissionDeniedView } from '../PermissionDeniedView';
import { ExperimentViewDescriptions } from './components/ExperimentViewDescriptions';
import { ExperimentViewNotes } from './components/ExperimentViewNotes';
import { ExperimentViewHeader } from './components/header/ExperimentViewHeader';
import { ExperimentViewHeaderCompare } from './components/header/ExperimentViewHeaderCompare';
import { ExperimentViewRuns } from './components/runs/ExperimentViewRuns';
import { useExperimentIds } from './hooks/useExperimentIds';
import { useExperiments } from './hooks/useExperiments';
import { useFetchExperiments } from './hooks/useFetchExperiments';
import { useElementHeight } from '../../../common/utils/useElementHeight';
import { searchDatasetsApi } from '../../actions';
import Utils from '../../../common/utils/Utils';
import { ExperimentPageUIStateContextProvider } from './contexts/ExperimentPageUIStateContext';
import { first } from 'lodash';
import { shouldEnableExperimentPageCompactHeader } from '../../../common/utils/FeatureUtils';
import { useExperimentPageSearchFacets } from './hooks/useExperimentPageSearchFacets';
import { usePersistExperimentPageViewState } from './hooks/usePersistExperimentPageViewState';
import { useDispatch, useSelector } from 'react-redux';
import { ThunkDispatch } from '../../../redux-types';
import { useExperimentRuns } from './hooks/useExperimentRuns';
import { ExperimentRunsSelectorResult } from './utils/experimentRuns.selector';
import { useSharedExperimentViewState } from './hooks/useSharedExperimentViewState';
import { useInitializeUIState } from './hooks/useInitializeUIState';
import { ExperimentViewDescriptionNotes } from './components/ExperimentViewDescriptionNotes';
import { ExperimentViewHeaderV2 } from './components/header/ExperimentViewHeaderV2';
import invariant from 'invariant';

export const ExperimentView = () => {
  const dispatch = useDispatch<ThunkDispatch>();

  const [searchFacets, experimentIds] = useExperimentPageSearchFacets();

  const experiments = useExperiments(experimentIds);

  const [firstExperiment] = experiments;

  const { fetchExperiments, isLoadingExperiment, requestError } = useFetchExperiments();

  const { elementHeight: hideableElementHeight, observeHeight } = useElementHeight();

  const [isMaximizedV1, setIsMaximizedV1] = useState(false);

  const [editing, setEditing] = useState(false);

  const [showAddDescriptionButton, setShowAddDescriptionButton] = useState(true);

  // Create new version of the UI state for the experiment page on this level
  const [uiState, setUIState, seedInitialUIState] = useInitializeUIState(experimentIds);

  const { isViewStateShared } = useSharedExperimentViewState(setUIState, first(experiments));

  // Get the maximized state from the new view state model if flag is set
  const isMaximized = uiState.viewMaximized;

  const {
    isLoadingRuns,
    loadMoreRuns,
    runsData,
    moreRunsAvailable,
    requestError: runsRequestError,
    refreshRuns,
  } = useExperimentRuns(uiState, searchFacets, experimentIds);

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

  if (!isViewInitialized) {
    // In the new view state model, wait for search facets to initialize
    return <LegacySkeleton />;
  }

  if (requestError && requestError.getErrorCode() === ErrorCodes.PERMISSION_DENIED) {
    return <PermissionDeniedView errorMessage={requestError.getMessageField()} />;
  }

  if (requestError && requestError.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
    return <NotFoundPage />;
  }

  invariant(searchFacets, 'searchFacets should be initialized at this point');

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

        <ExperimentViewRuns
          isLoading={false}
          experiments={experiments}
          isLoadingRuns={isLoadingRuns}
          runsData={runsData as ExperimentRunsSelectorResult}
          searchFacetsState={searchFacets}
          loadMoreRuns={loadMoreRuns}
          moreRunsAvailable={moreRunsAvailable}
          requestError={runsRequestError}
          refreshRuns={refreshRuns}
          uiState={uiState}
        />
      </div>
    </ExperimentPageUIStateContextProvider>
  );
};

const styles = {
  experimentViewWrapper: { height: '100%', display: 'flex', flexDirection: 'column' as const },
};

export default ExperimentView;
