import React, { useEffect } from 'react';
import { Button, PageWrapper, Spacer, ParagraphSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { PredefinedError } from '@databricks/web-shared/errors';
import invariant from 'invariant';
import { useNavigate, useParams, Outlet, matchPath, useLocation } from '../../../common/utils/RoutingUtils';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { useExperimentReduxStoreCompat } from '../../hooks/useExperimentReduxStoreCompat';
import { ExperimentPageHeaderWithDescription } from '../../components/experiment-page/components/ExperimentPageHeaderWithDescription';
import { coerceToEnum } from '@databricks/web-shared/utils';
import { ExperimentKind, ExperimentPageTabName } from '../../constants';
import {
  shouldEnableExperimentKindInference,
  shouldEnableExperimentPageChildRoutes,
} from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { useUpdateExperimentKind } from '../../components/experiment-page/hooks/useUpdateExperimentKind';
import { ExperimentViewHeaderKindSelector } from '../../components/experiment-page/components/header/ExperimentViewHeaderKindSelector';
import { getExperimentKindFromTags } from '../../utils/ExperimentKindUtils';
import { useInferExperimentKind } from '../../components/experiment-page/hooks/useInferExperimentKind';
import { ExperimentViewInferredKindModal } from '../../components/experiment-page/components/header/ExperimentViewInferredKindModal';
import Routes, { RoutePaths } from '../../routes';
import { EvaluationSubTabSelector } from './EvaluationSubTabSelector';
import { LabelingSubTabSelector } from './LabelingSubTabSelector';
import { useGetExperimentPageActiveTabByRoute } from '../../components/experiment-page/hooks/useGetExperimentPageActiveTabByRoute';
import { useNavigateToExperimentPageTab } from '../../components/experiment-page/hooks/useNavigateToExperimentPageTab';

const ExperimentRunsPage = React.lazy(() => import('../experiment-runs/ExperimentRunsPage'));
const ExperimentTracesPage = React.lazy(() => import('../experiment-traces/ExperimentTracesPage'));

const ExperimentLoggedModelListPage = React.lazy(
  () => import('../experiment-logged-models/ExperimentLoggedModelListPage'),
);

const ExperimentEvaluationRunsPage = React.lazy(
  () => import('../experiment-evaluation-runs/ExperimentEvaluationRunsPage'),
);

const ExperimentEvaluationDatasetsPage = React.lazy(
  () => import('../experiment-evaluation-datasets/ExperimentEvaluationDatasetsPage'),
);

const ExperimentPageTabsImpl = () => {
  const { experimentId, tabName } = useParams();
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();

  const { tabName: activeTabByRoute } = useGetExperimentPageActiveTabByRoute();
  const activeTab = activeTabByRoute ?? coerceToEnum(ExperimentPageTabName, tabName, ExperimentPageTabName.Models);

  invariant(experimentId, 'Experiment ID must be defined');
  if (!shouldEnableExperimentPageChildRoutes()) {
    // When child routes mode are not enabled, we expect the `tabName` to be defined.
    // Otherwise, this component is just an outlet for the experiment page child routes.
    invariant(tabName, 'Tab name must be defined');
  }

  const {
    data: experiment,
    loading: loadingExperiment,
    refetch: refetchExperiment,
    apiError: experimentApiError,
    apolloError: experimentApolloError,
  } = useGetExperimentQuery({
    experimentId,
  });

  const { mutate: updateExperimentKind, isLoading: updatingExperimentKind } =
    useUpdateExperimentKind(refetchExperiment);

  const experimentError = experimentApiError ?? experimentApolloError;

  // Put the experiment in the redux store so that the logged models page can transition smoothly
  useExperimentReduxStoreCompat(experiment);

  // For showstopper experiment fetch errors, we want it to hit the error boundary
  // so that the user can see the error message
  if (experimentError instanceof PredefinedError) {
    throw experimentError;
  }

  const experimentTags = experiment && 'tags' in experiment ? experiment?.tags : [];
  const canUpdateExperimentKind = true;

  const experimentKind = getExperimentKindFromTags(experimentTags);

  const {
    inferredExperimentKind,
    inferredExperimentPageTab,
    isLoading: inferringExperimentType,
    dismiss,
  } = useInferExperimentKind({
    experimentId,
    isLoadingExperiment: loadingExperiment,
    enabled: shouldEnableExperimentKindInference() && !experimentKind,
    experimentTags,
    updateExperimentKind,
  });

  // Check if the user landed on the experiment page without a specific tab (sub-route)...
  const { pathname } = useLocation();
  const matchedExperimentPageWithoutTab = Boolean(matchPath(RoutePaths.experimentPage, pathname));
  // ...if true, we want to navigate to the appropriate tab based on the experiment kind
  useNavigateToExperimentPageTab({
    enabled: shouldEnableExperimentPageChildRoutes() && matchedExperimentPageWithoutTab,
    experimentId,
  });

  useEffect(() => {
    // If the experiment kind is inferred, we want to navigate to the appropriate tab.
    // Should fire once when the experiment kind is inferred.
    if (inferredExperimentPageTab) {
      navigate(Routes.getExperimentPageTabRoute(experimentId, inferredExperimentPageTab), { replace: true });
    }
    // `navigate` reference changes whenever navigate is called, causing the
    // use to be unable to visit any tab until confirming the experiment type
    // if it is included in the deps array
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experimentId, inferredExperimentPageTab]);

  if (
    inferredExperimentKind === ExperimentKind.NO_INFERRED_TYPE &&
    canUpdateExperimentKind &&
    shouldEnableExperimentKindInference()
  ) {
    return (
      <ExperimentViewInferredKindModal
        onConfirm={(kind) => {
          updateExperimentKind(
            { experimentId, kind },
            {
              onSettled: () => {
                dismiss();
                if (shouldEnableExperimentPageChildRoutes() && kind === ExperimentKind.GENAI_DEVELOPMENT) {
                  // If the experiment kind is GENAI_DEVELOPMENT, we want to navigate to the Traces tab
                  navigate(Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces), {
                    replace: true,
                  });
                }
              },
            },
          );
        }}
        onDismiss={dismiss}
      />
    );
  }

  return (
    <>
      <ExperimentPageHeaderWithDescription
        experiment={experiment}
        loading={loadingExperiment || inferringExperimentType}
        onNoteUpdated={refetchExperiment}
        error={experimentError}
        inferredExperimentKind={inferredExperimentKind}
        refetchExperiment={refetchExperiment}
        experimentKindSelector={
          <ExperimentViewHeaderKindSelector
            value={experimentKind}
            inferredExperimentKind={inferredExperimentKind}
            onChange={(kind) => updateExperimentKind({ experimentId, kind })}
            isUpdating={updatingExperimentKind || inferringExperimentType}
            key={inferredExperimentKind}
            readOnly={!canUpdateExperimentKind}
          />
        }
      />
      {activeTab === ExperimentPageTabName.EvaluationRuns || activeTab === ExperimentPageTabName.Datasets ? (
        <EvaluationSubTabSelector experimentId={experimentId} activeTab={activeTab} />
      ) : activeTab === ExperimentPageTabName.LabelingSessions ||
        activeTab === ExperimentPageTabName.LabelingSchemas ? (
        <LabelingSubTabSelector experimentId={experimentId} activeTab={activeTab} />
      ) : (
        <>
          <Spacer size="sm" shrinks={false} />
          <div css={{ width: '100%', borderTop: `1px solid ${theme.colors.border}` }} />
        </>
      )}
      <Spacer size="sm" shrinks={false} />
      <React.Suspense
        fallback={
          <>
            {[...Array(8).keys()].map((i) => (
              <ParagraphSkeleton label="Loading..." key={i} seed={`s-${i}`} />
            ))}
          </>
        }
      >
        {shouldEnableExperimentPageChildRoutes() ? (
          <Outlet />
        ) : (
          <>
            {activeTab === ExperimentPageTabName.Traces && <ExperimentTracesPage />}
            {activeTab === ExperimentPageTabName.Runs && <ExperimentRunsPage />}
            {activeTab === ExperimentPageTabName.Models && <ExperimentLoggedModelListPage />}
            {activeTab === ExperimentPageTabName.EvaluationRuns && <ExperimentEvaluationRunsPage />}
            {activeTab === ExperimentPageTabName.Datasets && <ExperimentEvaluationDatasetsPage />}
          </>
        )}
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
        height: '100%',
      }}
    >
      <ExperimentPageTabsImpl />
    </div>
  );
};

export default ExperimentPageTabs;
