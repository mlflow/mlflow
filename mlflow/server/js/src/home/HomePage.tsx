import { useState } from 'react';
import { Alert, Button, Header, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../common/components/ScrollablePageWrapper';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../experiment-tracking/sdk/MlflowService';
import type { SearchExperimentsApiResponse } from '../experiment-tracking/types';
import { CreateExperimentModal } from '../experiment-tracking/components/modals/CreateExperimentModal';
import { useInvalidateExperimentList } from '../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';
import { GetStarted } from './components/GetStarted';
import { ExperimentsHomeView } from './components/ExperimentsHomeView';
import { DiscoverNews } from './components/DiscoverNews';
import { LogTracesDrawer } from './components/LogTracesDrawer';
import { Link } from '../common/utils/RoutingUtils';
import Routes from '../experiment-tracking/routes';
import { TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_KEY } from '../telemetry/utils';
import { useLocalStorage } from '../shared/web-shared/hooks';

type ExperimentQueryKey = ['home', 'recent-experiments'];

const RECENT_EXPERIMENTS_QUERY_KEY: ExperimentQueryKey = ['home', 'recent-experiments'];

const HomePage = () => {
  const { theme } = useDesignSystemTheme();
  const invalidateExperiments = useInvalidateExperimentList();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isTelemetryAlertDismissed, setIsTelemetryAlertDismissed] = useLocalStorage({
    key: TELEMETRY_INFO_ALERT_DISMISSED_STORAGE_KEY,
    version: 1,
    initialValue: false,
  });

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
      {!isTelemetryAlertDismissed && (
        <Alert
          componentId="mlflow.home.telemetry-alert"
          message="Information about UI telemetry"
          type="info"
          onClose={() => setIsTelemetryAlertDismissed(true)}
          description={
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              <span>
                <FormattedMessage
                  defaultMessage="MLflow collects usage data to improve the product. To confirm your settings, please visit the <settings>settings</settings> page. To learn more about what data is collected, please visit the <documentation>documentation</documentation>."
                  description="Telemetry alert description"
                  values={{
                    settings: (chunks: any) => (
                      <Link to={Routes.settingsPageRoute} onClick={() => setIsTelemetryAlertDismissed(true)}>
                        {chunks}
                      </Link>
                    ),
                    documentation: (chunks: any) => (
                      <Link
                        to="https://mlflow.org/docs/latest/community/usage-tracking.html"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {chunks}
                      </Link>
                    ),
                  }}
                />
              </span>
            </div>
          }
        />
      )}
      <GetStarted />
      <ExperimentsHomeView
        experiments={experiments}
        isLoading={isLoading}
        error={error}
        onCreateExperiment={handleOpenCreateModal}
        onRetry={refetch}
      />
      <DiscoverNews />

      <CreateExperimentModal
        isOpen={isCreateModalOpen}
        onClose={handleCloseCreateModal}
        onExperimentCreated={handleExperimentCreated}
      />
      <LogTracesDrawer />
    </ScrollablePageWrapper>
  );
};

export default HomePage;
