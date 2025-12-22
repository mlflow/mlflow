import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useProvidersQuery } from '../hooks/useProvidersQuery';
import { Alert, Header, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Outlet, useLocation } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';

const GatewayPage = () => {
  const { theme } = useDesignSystemTheme();
  const { data, error, isLoading } = useProvidersQuery();
  const location = useLocation();

  const isIndexRoute = location.pathname === '/gateway' || location.pathname === '/gateway/';

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <FormattedMessage defaultMessage="Gateway" description="Header title for the gateway configuration page" />
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {isIndexRoute ? (
          <>
            {error && (
              <>
                <Alert type="error" message={error.message} componentId="mlflow.gateway.error" closable={false} />
                <Spacer />
              </>
            )}
            {isLoading && (
              <div css={{ padding: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Loading providers..."
                  description="Loading message for providers list"
                />
              </div>
            )}
            {data && !isLoading && (
              <div css={{ padding: theme.spacing.md }}>
                <h3>
                  <FormattedMessage defaultMessage="Available Providers" description="Title for providers list" />
                </h3>
                <Spacer size="sm" />
                <ul css={{ listStyle: 'none', padding: 0 }}>
                  {data.map((provider) => (
                    <li key={provider} css={{ padding: theme.spacing.sm, marginBottom: theme.spacing.xs }}>
                      {provider}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        ) : (
          <Outlet />
        )}
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, GatewayPage);
