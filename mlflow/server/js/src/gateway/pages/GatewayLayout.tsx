import { Outlet } from '../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import {
  Header,
  Spacer,
  useDesignSystemTheme,
  CloudModelIcon,
  Empty,
  DatabaseIcon,
  Spinner,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { GatewaySideNav, type GatewayTabName } from '../components/side-nav/GatewaySideNav';
import { useLocation } from '../../common/utils/RoutingUtils';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';

const isUnsupportedBackendError = (error: Error | null | undefined): boolean => {
  if (!error) return false;

  const message = (error.message || '').toLowerCase();

  return (
    message.includes('notimplementederror') ||
    message.includes('filestore') ||
    message.includes('not implemented') ||
    // Flask returns "Internal Server Error" for uncaught exceptions
    message.includes('internal server error') ||
    // Check for generic 500 error messages
    message.includes('500')
  );
};

const GatewayLayout = () => {
  const { theme } = useDesignSystemTheme();
  const location = useLocation();
  const { data, error, isLoading } = useEndpointsQuery();

  const getActiveTab = (): GatewayTabName => {
    if (location.pathname.includes('/api-keys')) {
      return 'api-keys';
    }
    if (location.pathname.includes('/models')) {
      return 'models';
    }
    return 'endpoints';
  };

  // Show loading state during initial fetch only.
  // isLoading is true when the query is in flight and there's no cached data.
  if (isLoading) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Spacer shrinks={false} />
        <Header
          title={
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <CloudModelIcon />
              <FormattedMessage defaultMessage="Gateway" description="Header title for the gateway page" />
            </div>
          }
        />
        <Spacer shrinks={false} />
        <div
          css={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: theme.spacing.lg,
          }}
        >
          <Spinner />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (isUnsupportedBackendError(error)) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Spacer shrinks={false} />
        <Header
          title={
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <CloudModelIcon />
              <FormattedMessage defaultMessage="Gateway" description="Header title for the gateway page" />
            </div>
          }
        />
        <Spacer shrinks={false} />
        <div
          css={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: theme.spacing.lg,
          }}
        >
          <Empty
            image={<DatabaseIcon css={{ fontSize: 48, color: theme.colors.textSecondary }} />}
            title={
              <FormattedMessage
                defaultMessage="Gateway requires a SQL backend"
                description="Title for unsupported backend message in gateway"
              />
            }
            description={
              <FormattedMessage
                defaultMessage="Gateway features for managing endpoints and API keys require a SQL-based tracking store. Please configure your MLflow server with a SQL backend (e.g., SQLite or PostgreSQL) to use these features."
                description="Description for unsupported backend message in gateway"
              />
            }
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <CloudModelIcon />
            <FormattedMessage defaultMessage="Gateway" description="Header title for the gateway page" />
          </div>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <GatewaySideNav activeTab={getActiveTab()} />
        <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Outlet />
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, GatewayLayout);
