import React, { useEffect, useState } from 'react';
import { Header, HomeIcon, TableSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../common/components/ScrollablePageWrapper';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useSearchParams } from '../common/utils/RoutingUtils';
import { ErrorView } from '../common/components/ErrorView';
import { ErrorCodes } from '../common/constants';
import { ErrorWrapper } from '../common/utils/ErrorWrapper';
import { MlflowService } from '../experiment-tracking/sdk/MlflowService';
import type { SearchExperimentsApiResponse } from '../experiment-tracking/types';
import { CreateExperimentModal } from '../experiment-tracking/components/modals/CreateExperimentModal';
import { useInvalidateExperimentList } from '../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';
import { FeaturesSection } from './components/features';
import {
  extractWorkspaceFromSearchParams,
  setActiveWorkspace,
  setLastUsedWorkspace,
} from '../workspaces/utils/WorkspaceUtils';
// Loaders and lazy imports for expensive components
import LogTracesDrawerLoader from './components/LogTracesDrawerLoader';
import { TelemetryInfoAlert } from '../telemetry/TelemetryInfoAlert';

const ExperimentsHomeView = React.lazy(() => import('./components/ExperimentsHomeView'));

type ExperimentQueryKey = ['home', 'recent-experiments'];

const RECENT_EXPERIMENTS_QUERY_KEY: ExperimentQueryKey = ['home', 'recent-experiments'];

const HomePage = () => {
  const { theme } = useDesignSystemTheme();
  const invalidateExperiments = useInvalidateExperimentList();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [searchParams] = useSearchParams();
  const workspaceFromQuery = extractWorkspaceFromSearchParams(searchParams);

  const { data, error, isLoading, refetch } = useQuery<
    SearchExperimentsApiResponse,
    Error,
    SearchExperimentsApiResponse,
    ExperimentQueryKey
  >(RECENT_EXPERIMENTS_QUERY_KEY, {
    queryFn: () =>
      MlflowService.searchExperiments([
        ['max_results', '5'],
        ['order_by', 'last_update_time DESC'],
      ]),
    staleTime: 30_000,
  });

  const experiments = data?.experiments;
  const errorMessage = error instanceof ErrorWrapper ? error.getMessageField() : null;
  const isWorkspaceNotFoundError =
    workspaceFromQuery &&
    error instanceof ErrorWrapper &&
    error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST &&
    typeof errorMessage === 'string' &&
    errorMessage.includes(`'${workspaceFromQuery}'`) &&
    /^Workspace '.+' not found(?:\b|$)/.test(errorMessage);

  useEffect(() => {
    if (!isWorkspaceNotFoundError) {
      return;
    }

    setActiveWorkspace(null);
    setLastUsedWorkspace(null);
  }, [isWorkspaceNotFoundError]);

  const handleOpenCreateModal = () => setIsCreateModalOpen(true);
  const handleCloseCreateModal = () => setIsCreateModalOpen(false);
  const handleExperimentCreated = () => {
    handleCloseCreateModal();
    invalidateExperiments();
    refetch();
  };

  if (isWorkspaceNotFoundError) {
    return (
      <ErrorView
        statusCode={404}
        subMessage={`Workspace "${workspaceFromQuery}" was not found`}
        fallbackHomePageReactRoute="/"
        disableWorkspacePrefixOnFallback
      />
    );
  }

  return (
    <ScrollablePageWrapper
      css={{
        padding: theme.spacing.md,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        height: 'min-content',
      }}
    >
      <Header
        title={
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span
              css={{
                display: 'flex',
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
                padding: theme.spacing.sm,
              }}
            >
              <HomeIcon />
            </span>
            <FormattedMessage defaultMessage="Welcome to MLflow" description="Home page hero title" />
          </span>
        }
      />
      <TelemetryInfoAlert />
      <FeaturesSection />
      <React.Suspense fallback={<HomePageSectionSkeleton />}>
        <ExperimentsHomeView
          experiments={experiments}
          isLoading={isLoading}
          error={error}
          onCreateExperiment={handleOpenCreateModal}
          onRetry={refetch}
          onTagsUpdated={refetch}
        />
      </React.Suspense>

      <CreateExperimentModal
        isOpen={isCreateModalOpen}
        onClose={handleCloseCreateModal}
        onExperimentCreated={handleExperimentCreated}
      />
      <LogTracesDrawerLoader />
    </ScrollablePageWrapper>
  );
};

const HomePageSectionSkeleton = () => (
  <div>
    <TitleSkeleton />
    <TableSkeleton lines={3} />
  </div>
);

export default HomePage;
