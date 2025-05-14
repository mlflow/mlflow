import { Alert, LegacySkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { useEffect, useState } from 'react';
import { ErrorCodes } from '../../../common/constants';
import NotFoundPage from '../NotFoundPage';
import { PermissionDeniedView } from '../PermissionDeniedView';
import { ExperimentViewHeaderCompare } from './components/header/ExperimentViewHeaderCompare';
import { ExperimentViewRuns } from './components/runs/ExperimentViewRuns';
import { useExperiments } from './hooks/useExperiments';
import { useFetchExperiments } from './hooks/useFetchExperiments';
import { useElementHeight } from '../../../common/utils/useElementHeight';
import { searchDatasetsApi } from '../../actions';
import Utils from '../../../common/utils/Utils';
import { ExperimentPageUIStateContextProvider } from './contexts/ExperimentPageUIStateContext';
import { first } from 'lodash';
import {
  isExperimentEvalResultsMonitoringUIEnabled,
  isExperimentLoggedModelsUIEnabled,
  shouldEnableTracingUI,
  shouldUsePredefinedErrorsInExperimentTracking,
} from '../../../common/utils/FeatureUtils';
import { useExperimentPageSearchFacets } from './hooks/useExperimentPageSearchFacets';
import { usePersistExperimentPageViewState } from './hooks/usePersistExperimentPageViewState';
import { useDispatch } from 'react-redux';
import { ThunkDispatch } from '../../../redux-types';
import { useExperimentRuns } from './hooks/useExperimentRuns';
import { ExperimentRunsSelectorResult } from './utils/experimentRuns.selector';
import { useSharedExperimentViewState } from './hooks/useSharedExperimentViewState';
import { useInitializeUIState } from './hooks/useInitializeUIState';
import { ExperimentViewDescriptionNotes } from './components/ExperimentViewDescriptionNotes';
import { ExperimentViewHeader } from './components/header/ExperimentViewHeader';
import invariant from 'invariant';
import { useExperimentPageViewMode } from './hooks/useExperimentPageViewMode';
import { ExperimentViewTraces } from './components/ExperimentViewTraces';
import { FormattedMessage } from 'react-intl';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { NotFoundError, PermissionError } from '@databricks/web-shared/errors';
import { ExperimentViewNotFound } from './components/ExperimentViewNotFound';
import { ExperimentViewNoPermissionsError } from './components/ExperimentViewNoPermissionsError';
import { ErrorViewV2 } from '../../../common/components/ErrorViewV2';

// END-EDGE
export const ExperimentView = () => {
  const dispatch = useDispatch<ThunkDispatch>();
  const { theme } = useDesignSystemTheme();

  const [searchFacets, experimentIds, isPreview] = useExperimentPageSearchFacets();
  const [viewMode] = useExperimentPageViewMode();

  const experiments = useExperiments(experimentIds);

  const [firstExperiment] = experiments;

  const { fetchExperiments, isLoadingExperiment, requestError: experimentRequestError } = useFetchExperiments();

  const { elementHeight: hideableElementHeight, observeHeight } = useElementHeight();

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
    // If the new tabbed UI is enabled, fetch the experiments only if they are not already loaded.
    // Helps with the smooth page transition.
    if (
      (isExperimentLoggedModelsUIEnabled() || isExperimentEvalResultsMonitoringUIEnabled()) &&
      experimentIds.every((id) => experiments.find((exp) => exp.experimentId === id))
    ) {
      return;
    }
    fetchExperiments(experimentIds);
  }, [fetchExperiments, experimentIds, experiments]);

  useEffect(() => {
    // Seed the initial UI state when the experiments and runs are loaded.
    // Should only run once.
    seedInitialUIState(experiments, runsData);
  }, [seedInitialUIState, experiments, runsData]);

  useEffect(() => {
    const requestAction = searchDatasetsApi(experimentIds);
    dispatch(requestAction).catch((e) => {
      // In V2 error handling, do not display datasets retrieval error
      if (!shouldUsePredefinedErrorsInExperimentTracking()) {
        Utils.logErrorAndNotifyUser(e);
      }
    });
  }, [dispatch, experimentIds]);

  const isComparingExperiments = experimentIds.length > 1;

  usePersistExperimentPageViewState(uiState, searchFacets, experimentIds, isViewStateShared || isPreview);

  const isViewInitialized = Boolean(!isLoadingExperiment && experiments[0] && runsData && searchFacets);

  if (
    // Scenario for 404: either request error is resolved to NotFoundError or the code of ErrorWrapper is "RESOURCE_DOES_NOT_EXIST"
    experimentRequestError instanceof NotFoundError ||
    (experimentRequestError instanceof ErrorWrapper &&
      experimentRequestError.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST)
  ) {
    return <ExperimentViewNotFound />;
  }

  if (
    // Scenario for 401: either request error is resolved to PermissionError or the code of ErrorWrapper is "PERMISSION_DENIED"
    experimentRequestError instanceof PermissionError ||
    (experimentRequestError instanceof ErrorWrapper &&
      experimentRequestError.getErrorCode() === ErrorCodes.PERMISSION_DENIED)
  ) {
    return <ExperimentViewNoPermissionsError />;
  }

  if (experimentRequestError instanceof Error) {
    return <ErrorViewV2 css={{ height: '100%' }} error={experimentRequestError} />;
  }

  if (!isViewInitialized) {
    // In the new view state model, wait for search facets to initialize
    return <LegacySkeleton />;
  }

  invariant(searchFacets, 'searchFacets should be initialized at this point');

  const isLoading = isLoadingExperiment || !experiments[0];

  const renderExperimentHeader = () => (
    <>
      <ExperimentViewHeader
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
  );

  const getRenderedView = () => {
    if (shouldEnableTracingUI() && viewMode === 'TRACES') {
      return <ExperimentViewTraces experimentIds={experimentIds} />;
    }

    return (
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
    );
  };

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
        {getRenderedView()}
      </div>
    </ExperimentPageUIStateContextProvider>
  );
};

const styles = {
  experimentViewWrapper: { height: '100%', display: 'flex', flexDirection: 'column' as const },
};
