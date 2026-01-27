import { useMemo } from 'react';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import {
  Button,
  ChainIcon,
  CloudModelIcon,
  Header,
  PlusIcon,
  Spacer,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, Outlet, useLocation } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { EndpointsList } from '../components/endpoints';
import { GatewaySideNav, type GatewayTab } from '../components/side-nav';
import { GatewaySetupGuide } from '../components/SecretsSetupGuide';
import { DefaultPassphraseBanner } from '../components/DefaultPassphraseBanner';
import { useSecretsConfigQuery } from '../hooks/useSecretsConfigQuery';
import ApiKeysPage from './ApiKeysPage';
import GatewayUsagePage from './GatewayUsagePage';
import GatewayRoutes from '../routes';
import { shouldEnableWorkflowBasedNavigation } from '../../common/utils/FeatureUtils';

const GatewayPageTitle = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm }}>
      <CloudModelIcon />
      <FormattedMessage defaultMessage="AI Gateway" description="Header title for the AI Gateway configuration page" />
    </span>
  );
};

const GatewayPage = () => {
  const { theme } = useDesignSystemTheme();
  const location = useLocation();
  const { data: secretsConfig, isLoading: isLoadingConfig } = useSecretsConfigQuery();

  const activeTab: GatewayTab = useMemo(() => {
    if (location.pathname.includes('/api-keys')) {
      return 'api-keys';
    }
    if (location.pathname.includes('/usage')) {
      return 'usage';
    }
    return 'endpoints';
  }, [location.pathname]);

  const isIndexRoute = location.pathname === '/gateway' || location.pathname === '/gateway/';
  const isApiKeysRoute = location.pathname.includes('/api-keys');
  const isUsageRoute = location.pathname.includes('/usage');
  const isNestedRoute = !isIndexRoute && !isApiKeysRoute && !isUsageRoute;

  if (isLoadingConfig) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Spacer shrinks={false} />
        <Header title={<GatewayPageTitle />} />
        <div
          css={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: theme.spacing.sm,
          }}
        >
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading..." description="Loading message for gateway page" />
        </div>
      </ScrollablePageWrapper>
    );
  }

  const secretsAvailable = secretsConfig?.secrets_available ?? false;
  const isUsingDefaultPassphrase = secretsConfig?.using_default_passphrase ?? false;

  if (!secretsAvailable) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Spacer shrinks={false} />
        <Header title={<GatewayPageTitle />} />
        <div
          css={{
            flex: 1,
            overflow: 'auto',
          }}
        >
          <GatewaySetupGuide />
        </div>
      </ScrollablePageWrapper>
    );
  }

  const enableWorkflowBasedNavigation = shouldEnableWorkflowBasedNavigation();

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header title={<GatewayPageTitle />} />
      {isUsingDefaultPassphrase && <DefaultPassphraseBanner />}
      <div css={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {!enableWorkflowBasedNavigation && <GatewaySideNav activeTab={activeTab} />}
        <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {isNestedRoute ? (
            <Outlet />
          ) : (
            <>
              {isIndexRoute && (
                <div css={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: theme.spacing.md,
                      borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                    }}
                  >
                    <Typography.Title
                      level={3}
                      css={{ margin: 0, display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
                    >
                      <ChainIcon />
                      <FormattedMessage defaultMessage="Endpoints" description="Endpoints page title" />
                    </Typography.Title>
                    <Link to={GatewayRoutes.createEndpointPageRoute}>
                      <Button componentId="mlflow.gateway.endpoints.create-button" type="primary" icon={<PlusIcon />}>
                        <FormattedMessage
                          defaultMessage="Create endpoint"
                          description="Gateway > Endpoints page > Create endpoint button"
                        />
                      </Button>
                    </Link>
                  </div>
                  <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
                    <EndpointsList />
                  </div>
                </div>
              )}
              {isApiKeysRoute && <ApiKeysPage />}
              {isUsageRoute && <GatewayUsagePage />}
            </>
          )}
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, GatewayPage);
