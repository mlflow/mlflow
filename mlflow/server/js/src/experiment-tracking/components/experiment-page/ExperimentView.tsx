import { Skeleton } from '@databricks/design-system';

import { useEffect, useMemo } from 'react';
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

export const ExperimentView = () => {
  const experimentIds = useExperimentIds();
  const experiments = useExperiments(experimentIds);

  const [firstExperiment] = experiments;

  const { fetchExperiments, isLoadingExperiment, requestError } = useFetchExperiments();

  useEffect(() => {
    fetchExperiments(experimentIds);
  }, [fetchExperiments, experimentIds]);

  const isComparingExperiments = experimentIds.length > 1;

  const experimentIdsHash = useMemo(() => JSON.stringify(experimentIds.sort()), [experimentIds]);

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
        <Skeleton title paragraph={false} active />
      ) : (
        <>
          {isComparingExperiments ? (
            <ExperimentViewHeaderCompare experiments={experiments} />
          ) : (
            <>
              <ExperimentViewHeader experiment={firstExperiment} />
              <ExperimentViewDescriptions experiment={firstExperiment} />
              <ExperimentViewNotes experiment={firstExperiment} />
            </>
          )}
        </>
      )}

      <ExperimentViewRuns experiments={experiments} isLoading={isLoading} />
    </div>
  );
};

const styles = {
  experimentViewWrapper: { height: '100%', display: 'flex', flexDirection: 'column' as const },
};

export default ExperimentView;
