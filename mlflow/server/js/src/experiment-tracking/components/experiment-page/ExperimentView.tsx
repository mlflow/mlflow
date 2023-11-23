import { LegacySkeleton } from '@databricks/design-system';

import { useEffect, useState } from 'react';
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
import { useAsyncDispatch } from './hooks/useAsyncDispatch';
import { searchDatasetsApi } from '../../actions';
import Utils from '../../../common/utils/Utils';

export const ExperimentView = () => {
  const dispatch = useAsyncDispatch();

  const experimentIds = useExperimentIds();
  const experiments = useExperiments(experimentIds);

  const [firstExperiment] = experiments;

  const { fetchExperiments, isLoadingExperiment, requestError } = useFetchExperiments();

  const { elementHeight: hideableElementHeight, observeHeight } = useElementHeight();

  const [isMaximized, setIsMaximized] = useState(false);

  useEffect(() => {
    fetchExperiments(experimentIds);
  }, [fetchExperiments, experimentIds]);

  useEffect(() => {
    const requestAction = searchDatasetsApi(experimentIds);
    dispatch(requestAction).catch((e) => {
      Utils.logErrorAndNotifyUser(e);
    });
  }, [dispatch, experimentIds]);

  const isComparingExperiments = experimentIds.length > 1;

  if (requestError && requestError.getErrorCode() === ErrorCodes.PERMISSION_DENIED) {
    return <PermissionDeniedView errorMessage={requestError.getMessageField()} />;
  }

  if (requestError && requestError.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
    return <NotFoundPage />;
  }

  const isLoading = isLoadingExperiment || !experiments[0];

  return (
    <div css={styles.experimentViewWrapper}>
      {isLoading ? (
        <LegacySkeleton title paragraph={false} active />
      ) : (
        <>
          {isComparingExperiments ? (
            <ExperimentViewHeaderCompare experiments={experiments} />
          ) : (
            <>
              <ExperimentViewHeader experiment={firstExperiment} />
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
          )}
        </>
      )}

      <ExperimentViewRuns
        experiments={experiments}
        isLoading={isLoading}
        // We don't keep the view state on this level to maximize <ExperimentViewRuns>'s performance
        onMaximizedChange={setIsMaximized}
      />
    </div>
  );
};

const styles = {
  experimentViewWrapper: { height: '100%', display: 'flex', flexDirection: 'column' as const },
};

export default ExperimentView;
