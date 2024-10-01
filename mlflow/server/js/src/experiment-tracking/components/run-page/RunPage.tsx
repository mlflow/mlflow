import { DangerIcon, Empty, ParagraphSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { useSelector } from 'react-redux';
import invariant from 'invariant';
import { useMemo, useState } from 'react';

import { PageContainer } from '../../../common/components/PageContainer';
import { useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import Utils from '../../../common/utils/Utils';
import { RunPageTabName } from '../../constants';
import { RenameRunModal } from '../modals/RenameRunModal';
import { RunViewArtifactTab } from './RunViewArtifactTab';
import { RunViewHeader } from './RunViewHeader';
import { RunViewMetricCharts } from './RunViewMetricCharts';
import { RunViewOverview } from './RunViewOverview';
import { useRunDetailsPageData } from './hooks/useRunDetailsPageData';
import { useRunViewActiveTab } from './useRunViewActiveTab';
import { ReduxState } from '../../../redux-types';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { RunNotFoundView } from '../RunNotFoundView';
import { ErrorCodes } from '../../../common/constants';
import NotFoundPage from '../NotFoundPage';
import { FormattedMessage } from 'react-intl';
import { isSystemMetricKey } from '../../utils/MetricsUtils';
import DeleteRunModal from '../modals/DeleteRunModal';
import Routes from '../../routes';
import { RunViewMetricChartsV2 } from './RunViewMetricChartsV2';
import {
  shouldEnableRunDetailsPageTracesTab,
  shouldUseUnifiedRunCharts,
  shouldEnableGraphQLRunDetailsPage,
} from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { useMediaQuery } from '@databricks/web-shared/hooks';
import { RunViewTracesTab } from './RunViewTracesTab';
import { getGraphQLErrorMessage } from '../../../graphql/get-graphql-error';

const RunPageLoadingState = () => (
  <PageContainer>
    <TitleSkeleton
      loading
      label={<FormattedMessage defaultMessage="Run page loading" description="Run page > Loading state" />}
    />
    {[...Array(3).keys()].map((i) => (
      <ParagraphSkeleton key={i} seed={`s-${i}`} />
    ))}
  </PageContainer>
);

export const RunPage = () => {
  const { runUuid, experimentId } = useParams<{
    runUuid: string;
    experimentId: string;
  }>();
  const navigate = useNavigate();
  const { theme } = useDesignSystemTheme();
  const [renameModalVisible, setRenameModalVisible] = useState(false);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);

  invariant(runUuid, '[RunPage] Run UUID route param not provided');
  invariant(experimentId, '[RunPage] Experiment ID route param not provided');

  const {
    experiment,
    error,
    latestMetrics,
    loading,
    params,
    refetchRun,
    runInfo,
    tags,
    experimentFetchError,
    runFetchError,
    apiError,
    datasets,
  } = useRunDetailsPageData({
    experimentId,
    runUuid,
  });

  const [modelMetricKeys, systemMetricKeys] = useMemo<[string[], string[]]>(() => {
    if (!latestMetrics) {
      return [[], []];
    }

    return [
      Object.keys(latestMetrics).filter((metricKey) => !isSystemMetricKey(metricKey)),
      Object.keys(latestMetrics).filter((metricKey) => isSystemMetricKey(metricKey)),
    ];
  }, [latestMetrics]);

  const { comparedExperimentIds = [], hasComparedExperimentsBefore = false } = useSelector(
    (state: ReduxState) => state.comparedExperiments || {},
  );

  const activeTab = useRunViewActiveTab();

  const renderActiveTab = () => {
    if (!runInfo) {
      return null;
    }
    switch (activeTab) {
      case RunPageTabName.MODEL_METRIC_CHARTS:
        if (shouldUseUnifiedRunCharts()) {
          return (
            <RunViewMetricChartsV2
              key="model"
              mode="model"
              metricKeys={modelMetricKeys}
              runInfo={runInfo}
              latestMetrics={latestMetrics}
              tags={tags}
              params={params}
            />
          );
        } else {
          return <RunViewMetricCharts mode="model" metricKeys={modelMetricKeys} runInfo={runInfo} />;
        }
      case RunPageTabName.SYSTEM_METRIC_CHARTS:
        if (shouldUseUnifiedRunCharts()) {
          return (
            <RunViewMetricChartsV2
              key="system"
              mode="system"
              metricKeys={systemMetricKeys}
              runInfo={runInfo}
              latestMetrics={latestMetrics}
              tags={tags}
              params={params}
            />
          );
        } else {
          return <RunViewMetricCharts mode="system" metricKeys={systemMetricKeys} runInfo={runInfo} />;
        }
      case RunPageTabName.ARTIFACTS:
        return (
          <RunViewArtifactTab
            runUuid={runUuid}
            runTags={tags}
            experimentId={experimentId}
            artifactUri={runInfo.artifactUri ?? undefined}
          />
        );
      case RunPageTabName.TRACES:
        if (shouldEnableRunDetailsPageTracesTab()) {
          return <RunViewTracesTab runUuid={runUuid} runTags={tags} experimentId={experimentId} />;
        }
    }

    return (
      <RunViewOverview
        runInfo={runInfo}
        tags={tags}
        params={params}
        latestMetrics={latestMetrics}
        runUuid={runUuid}
        onRunDataUpdated={refetchRun}
        datasets={datasets}
      />
    );
  };

  // Use full height page with scrollable tab area only for non-xs screens
  const useFullHeightPage = useMediaQuery(`(min-width: ${theme.responsive.breakpoints.sm}px)`);

  const initialLoading = loading && (!runInfo || !experiment);

  // Handle "run not found" error
  if (
    // For REST API:
    (runFetchError instanceof ErrorWrapper && runFetchError.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) ||
    // For GraphQL:
    apiError?.code === ErrorCodes.RESOURCE_DOES_NOT_EXIST ||
    (error && getGraphQLErrorMessage(error).match(/not found$/))
  ) {
    return <RunNotFoundView runId={runUuid} />;
  }

  // Handle experiment not found error
  if (
    experimentFetchError instanceof ErrorWrapper &&
    experimentFetchError.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST
  ) {
    return <NotFoundPage />;
  }

  if (runFetchError || experimentFetchError) {
    return null;
  }

  if (shouldEnableGraphQLRunDetailsPage() && (error || apiError)) {
    return (
      <div css={{ marginTop: theme.spacing.lg }}>
        <Empty
          title={
            <FormattedMessage
              defaultMessage="Can't load run details"
              description="Run page > error loading page title"
            />
          }
          description={getGraphQLErrorMessage(apiError ?? error)}
          image={<DangerIcon />}
        />
      </div>
    );
  }

  // Display spinner/skeleton for the initial data load
  if (initialLoading || !runInfo || !experiment) {
    return <RunPageLoadingState />;
  }

  return (
    <>
      <PageContainer usesFullHeight={useFullHeightPage}>
        {/* Header fixed on top */}
        <RunViewHeader
          comparedExperimentIds={comparedExperimentIds}
          experiment={experiment}
          handleRenameRunClick={() => setRenameModalVisible(true)}
          handleDeleteRunClick={() => setDeleteModalVisible(true)}
          hasComparedExperimentsBefore={hasComparedExperimentsBefore}
          runDisplayName={Utils.getRunDisplayName(runInfo, runUuid)}
          runTags={tags}
          runParams={params}
          runUuid={runUuid}
          artifactRootUri={runInfo?.artifactUri ?? undefined}
        />
        {/* Scroll tab contents independently within own container */}
        <div css={{ flex: 1, overflow: 'auto', marginBottom: theme.spacing.sm, display: 'flex' }}>
          {renderActiveTab()}
        </div>
      </PageContainer>
      <RenameRunModal
        runUuid={runUuid}
        onClose={() => setRenameModalVisible(false)}
        runName={runInfo.runName ?? ''}
        isOpen={renameModalVisible}
        onSuccess={refetchRun}
      />
      <DeleteRunModal
        selectedRunIds={[runUuid]}
        onClose={() => setDeleteModalVisible(false)}
        isOpen={deleteModalVisible}
        onSuccess={() => {
          navigate(Routes.getExperimentPageRoute(experimentId));
        }}
      />
    </>
  );
};

export default RunPage;
