import React, { useState } from 'react';
import { Header, TableSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../common/components/ScrollablePageWrapper';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../experiment-tracking/sdk/MlflowService';
import type { SearchExperimentsApiResponse } from '../experiment-tracking/types';
import { CreateExperimentModal } from '../experiment-tracking/components/modals/CreateExperimentModal';
import { useInvalidateExperimentList } from '../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';

// Loaders and lazy imports for expensive components
import LogTracesDrawerLoader from './components/LogTracesDrawerLoader';
import { TelemetryInfoAlert } from '../telemetry/TelemetryInfoAlert';
const GetStarted = React.lazy(() => import('./components/GetStarted'));
const DiscoverNews = React.lazy(() => import('./components/DiscoverNews'));
const ExperimentsHomeView = React.lazy(() => import('./components/ExperimentsHomeView'));

type ExperimentQueryKey = ['home', 'recent-experiments'];

const RECENT_EXPERIMENTS_QUERY_KEY: ExperimentQueryKey = ['home', 'recent-experiments'];

const HomePage = () => {
  const { theme } = useDesignSystemTheme();
  const invalidateExperiments = useInvalidateExperimentList();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

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

  const handleOpenCreateModal = () => setIsCreateModalOpen(true);
  const handleCloseCreateModal = () => setIsCreateModalOpen(false);
  const handleExperimentCreated = () => {
    handleCloseCreateModal();
    invalidateExperiments();
    refetch();
  };

  return (
    <ScrollablePageWrapper
      css={{
        padding: theme.spacing.md,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.lg,
        height: 'min-content',
      }}
    >
      <Header title={<FormattedMessage defaultMessage="Welcome to MLflow" description="Home page hero title" />} />
      <TelemetryInfoAlert />
      <React.Suspense fallback={<HomePageSectionSkeleton />}>
        <GetStarted />
      </React.Suspense>
      <React.Suspense fallback={<HomePageSectionSkeleton />}>
        <ExperimentsHomeView
          experiments={experiments}
          isLoading={isLoading}
          error={error}
          onCreateExperiment={handleOpenCreateModal}
          onRetry={refetch}
        />
      </React.Suspense>
      <React.Suspense fallback={<HomePageSectionSkeleton />}>
        <DiscoverNews />
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

const HomePageSectionSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div>
      <TitleSkeleton />
      <TableSkeleton lines={3} />
    </div>
  );
};

export default HomePage;
