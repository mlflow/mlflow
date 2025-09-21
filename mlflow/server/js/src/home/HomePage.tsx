import { useState } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { ScrollablePageWrapper } from '../common/components/ScrollablePageWrapper';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../experiment-tracking/sdk/MlflowService';
import type { SearchExperimentsApiResponse } from '../experiment-tracking/types';
import { CreateExperimentModal } from '../experiment-tracking/components/modals/CreateExperimentModal';
import { useInvalidateExperimentList } from '../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';
import { HomeHeader } from './components/HomeHeader';
import { GetStarted } from './components/GetStarted';
import { ExperimentsHomeView } from './components/ExperimentsHomeView';
import { DiscoverNews } from './components/DiscoverNews';

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
        padding: `0 ${theme.spacing.lg}px`,
        display: 'flex',
        justifyContent: 'center',
      }}
    >
      <div
        css={{
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.lg,
        }}
      >
        <HomeHeader onCreateExperiment={handleOpenCreateModal} />
        <GetStarted />
        <ExperimentsHomeView
          experiments={experiments}
          isLoading={isLoading}
          error={error}
          onCreateExperiment={handleOpenCreateModal}
          onRetry={refetch}
        />
        <DiscoverNews />
      </div>

      <CreateExperimentModal
        isOpen={isCreateModalOpen}
        onClose={handleCloseCreateModal}
        onExperimentCreated={handleExperimentCreated}
      />
    </ScrollablePageWrapper>
  );
};

export default HomePage;
