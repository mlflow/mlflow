import React from 'react';
import { PageWrapper, Spacer, TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { PredefinedError } from '@databricks/web-shared/errors';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentViewRunsModeSwitchV2 } from '../../components/experiment-page/components/runs/ExperimentViewRunsModeSwitchV2';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { useExperimentReduxStoreCompat } from '../../hooks/useExperimentReduxStoreCompat';
import { ExperimentPageHeaderWithDescription } from '../../components/experiment-page/components/ExperimentPageHeaderWithDescription';
import { coerceToEnum } from '@databricks/web-shared/utils';
import { ExperimentPageTabName } from '../../constants';

const ExperimentLoggedModelListPage = React.lazy(
  () => import('../experiment-logged-models/ExperimentLoggedModelListPage'),
);

const ExperimentPageTabsImpl = () => {
  const { experimentId, tabName } = useParams();

  const activeTab = coerceToEnum(ExperimentPageTabName, tabName, ExperimentPageTabName.Models);

  invariant(experimentId, 'Experiment ID must be defined');
  invariant(tabName, 'Tab name must be defined');

  const {
    data: experiment,
    loading: loadingExperiment,
    refetch: refetchExperiment,
    apiError: experimentApiError,
    apolloError: experimentApolloError,
  } = useGetExperimentQuery({
    experimentId,
  });

  const experimentError = experimentApiError ?? experimentApolloError;

  // Put the experiment in the redux store so that the logged models page can transition smoothly
  useExperimentReduxStoreCompat(experiment);

  // For showstopper experiment fetch errors, we want it to hit the error boundary
  // so that the user can see the error message
  if (experimentError instanceof PredefinedError) {
    throw experimentError;
  }

  return (
    <>
      <ExperimentPageHeaderWithDescription
        experiment={experiment}
        loading={loadingExperiment}
        onNoteUpdated={refetchExperiment}
        error={experimentError}
      />
      <Spacer size="sm" shrinks={false} />
      <ExperimentViewRunsModeSwitchV2 experimentId={experimentId} activeTab={activeTab} />
      <Spacer size="sm" shrinks={false} />
      <React.Suspense fallback={<TableSkeleton lines={8} />}>
        {activeTab === ExperimentPageTabName.Models && <ExperimentLoggedModelListPage />}
      </React.Suspense>
    </>
  );
};

const ExperimentPageTabs = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        flex: 1,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing.md,
        paddingTop: theme.spacing.lg,
      }}
    >
      <ExperimentPageTabsImpl />
    </div>
  );
};

export default ExperimentPageTabs;
