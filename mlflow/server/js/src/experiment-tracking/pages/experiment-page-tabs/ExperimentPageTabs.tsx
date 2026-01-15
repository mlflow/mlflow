import React, { useEffect } from 'react';
import { Button, PageWrapper, Spacer, ParagraphSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { PredefinedError } from '@databricks/web-shared/errors';
import invariant from 'invariant';
import { useNavigate, useParams, Outlet, useLocation } from '../../../common/utils/RoutingUtils';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { useExperimentReduxStoreCompat } from '../../hooks/useExperimentReduxStoreCompat';
import { ExperimentPageHeaderWithDescription } from '../../components/experiment-page/components/ExperimentPageHeaderWithDescription';
import { coerceToEnum } from '@databricks/web-shared/utils';
import { ExperimentKind, ExperimentPageTabName } from '../../constants';
import { shouldEnableExperimentPageSideTabs } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { useUpdateExperimentKind } from '../../components/experiment-page/hooks/useUpdateExperimentKind';
import { ExperimentViewHeaderKindSelector } from '../../components/experiment-page/components/header/ExperimentViewHeaderKindSelector';
import { getExperimentKindFromTags } from '../../utils/ExperimentKindUtils';
import { useInferExperimentKind } from '../../components/experiment-page/hooks/useInferExperimentKind';
import { ExperimentViewInferredKindModal } from '../../components/experiment-page/components/header/ExperimentViewInferredKindModal';
import Routes, { RoutePaths } from '../../routes';
import { matchPathWithWorkspace } from '../../../common/utils/WorkspaceRouteUtils';
import { useGetExperimentPageActiveTabByRoute } from '../../components/experiment-page/hooks/useGetExperimentPageActiveTabByRoute';
import { useNavigateToExperimentPageTab } from '../../components/experiment-page/hooks/useNavigateToExperimentPageTab';
import { ExperimentPageSubTabSelector } from './ExperimentPageSubTabSelector';
import { ExperimentPageSideNav, ExperimentPageSideNavSkeleton } from './side-nav/ExperimentPageSideNav';

const ExperimentPageTabsImpl = () => {
  const { experimentId, tabName } = useParams();
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();

  const { tabName: activeTabByRoute } = useGetExperimentPageActiveTabByRoute();
  const activeTab = activeTabByRoute ?? coerceToEnum(ExperimentPageTabName, tabName, ExperimentPageTabName.Models);

  invariant(experimentId, 'Experiment ID must be defined');

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
  // We won't try to infer the experiment kind if it's already set, but we also wait for experiment to load
  const isExperimentKindInferenceEnabled = Boolean(experiment && !experimentKind);

  const {
    inferredExperimentKind,
    inferredExperimentPageTab,
    isLoading: inferringExperimentType,
    dismiss,
  } = useInferExperimentKind({
    experimentId,
    isLoadingExperiment: loadingExperiment,
    enabled: isExperimentKindInferenceEnabled,
    experimentTags,
    updateExperimentKind,
  });

  // Check if the user landed on the experiment page without a specific tab (sub-route)...
  const { pathname } = useLocation();
  const matchedExperimentPageWithoutTab = Boolean(matchPathWithWorkspace(RoutePaths.experimentPage, pathname));
  // ...if true, we want to navigate to the appropriate tab based on the experiment kind
  useNavigateToExperimentPageTab({
    enabled: matchedExperimentPageWithoutTab,
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

  if (inferredExperimentKind === ExperimentKind.NO_INFERRED_TYPE && canUpdateExperimentKind) {
    return (
      <ExperimentViewInferredKindModal
        onConfirm={(kind) => {
          updateExperimentKind(
            { experimentId, kind },
            {
              onSettled: () => {
                dismiss();
                if (kind === ExperimentKind.GENAI_DEVELOPMENT) {
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

  const outletComponent = (
    <React.Suspense
      fallback={
        <>
          {[...Array(8).keys()].map((i) => (
            <ParagraphSkeleton label="Loading..." key={i} seed={`s-${i}`} />
          ))}
        </>
      }
    >
      <Outlet />
    </React.Suspense>
  );

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
      {!shouldEnableExperimentPageSideTabs() && (
        <>
          <ExperimentPageSubTabSelector experimentId={experimentId} activeTab={activeTab} />
          <Spacer size="sm" shrinks={false} />
        </>
      )}
      {shouldEnableExperimentPageSideTabs() ? (
        <div css={{ display: 'flex', flex: 1, minWidth: 0, minHeight: 0 }}>
          {loadingExperiment || inferringExperimentType ? (
            <ExperimentPageSideNavSkeleton />
          ) : (
            <ExperimentPageSideNav
              experimentKind={experimentKind ?? inferredExperimentKind ?? ExperimentKind.CUSTOM_MODEL_DEVELOPMENT}
              activeTab={activeTab}
            />
          )}
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              padding: theme.spacing.sm,
              flex: 1,
              minWidth: 0,
              minHeight: 0,
            }}
          >
            {outletComponent}
          </div>
        </div>
      ) : (
        outletComponent
      )}
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
