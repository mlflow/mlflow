import React from 'react';
import { Header, TableSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../common/components/ScrollablePageWrapper';
import { useCreateWorkspaceModal } from './components/CreateWorkspaceModal';
import { useNavigate } from '../common/utils/RoutingUtils';
import { setActiveWorkspace, WORKSPACE_QUERY_PARAM } from '../workspaces/utils/WorkspaceUtils';

// Loaders and lazy imports for expensive components
import LogTracesDrawerLoader from './components/LogTracesDrawerLoader';
import { TelemetryInfoAlert } from '../telemetry/TelemetryInfoAlert';
const GetStarted = React.lazy(() => import('./components/GetStarted'));
const WorkspacesHomeView = React.lazy(() => import('./components/WorkspacesHomeView'));

const WorkspaceLandingPage = () => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();

  const { CreateWorkspaceModal, openModal } = useCreateWorkspaceModal({
    onSuccess: (workspaceName: string) => {
      // Set the newly created workspace as active and navigate to home with workspace query param
      setActiveWorkspace(workspaceName);
      navigate(`/?${WORKSPACE_QUERY_PARAM}=${encodeURIComponent(workspaceName)}`);
    },
  });

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
      <Header
        title={<FormattedMessage defaultMessage="Welcome to MLflow" description="Workspace landing page title" />}
      />
      <TelemetryInfoAlert />
      <React.Suspense fallback={<HomePageSectionSkeleton />}>
        <GetStarted />
      </React.Suspense>
      <React.Suspense fallback={<HomePageSectionSkeleton />}>
        <WorkspacesHomeView onCreateWorkspace={openModal} />
      </React.Suspense>

      {CreateWorkspaceModal}
      <LogTracesDrawerLoader />
    </ScrollablePageWrapper>
  );
};

const HomePageSectionSkeleton = () => {
  return (
    <div>
      <TitleSkeleton />
      <TableSkeleton lines={3} />
    </div>
  );
};

export default WorkspaceLandingPage;
