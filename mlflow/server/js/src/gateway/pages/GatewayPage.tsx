import { useMemo } from 'react';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { Breadcrumb, ChainIcon, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, Outlet, useLocation } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { EndpointsList } from '../components/endpoints';
import { GatewaySideNav, type GatewayTab } from '../components/side-nav';
import { GatewayLabel } from '../../common/components/GatewayNewTag';
import { GatewaySetupGuide } from '../components/SecretsSetupGuide';
import { useSecretsConfigQuery } from '../hooks/useSecretsConfigQuery';
import ApiKeysPage from './ApiKeysPage';
import BudgetsPage from './BudgetsPage';
import GatewayUsagePage from './GatewayUsagePage';
import GatewayRoutes from '../routes';
import { shouldEnableWorkflowBasedNavigation } from '../../common/utils/FeatureUtils';

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
    if (location.pathname.includes('/budgets')) {
      return 'budgets';
    }
    return 'endpoints';
  }, [location.pathname]);

  const isIndexRoute = location.pathname === '/gateway' || location.pathname === '/gateway/';
  const isApiKeysRoute = location.pathname.includes('/api-keys');
  const isUsageRoute = location.pathname.includes('/usage');
  const isBudgetsRoute = location.pathname.includes('/budgets');
  const isNestedRoute = !isIndexRoute && !isApiKeysRoute && !isUsageRoute && !isBudgetsRoute;

  if (isLoadingConfig) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
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

  if (!secretsAvailable) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
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
                      padding: theme.spacing.md,
                      borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                    }}
                  >
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                      <Breadcrumb includeTrailingCaret>
                        <Breadcrumb.Item>
                          <Link
                            componentId="mlflow.gateway.endpoints.breadcrumb_gateway_link"
                            to={GatewayRoutes.gatewayPageRoute}
                          >
                            <GatewayLabel />
                          </Link>
                        </Breadcrumb.Item>
                      </Breadcrumb>
                      <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                        <div
                          css={{
                            borderRadius: theme.borders.borderRadiusSm,
                            backgroundColor: theme.colors.backgroundSecondary,
                            padding: theme.spacing.sm,
                            display: 'flex',
                          }}
                        >
                          <ChainIcon />
                        </div>
                        <Typography.Title withoutMargins level={2}>
                          <FormattedMessage defaultMessage="Endpoints" description="Endpoints page title" />
                        </Typography.Title>
                      </div>
                    </div>
                  </div>
                  <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
                    <EndpointsList />
                  </div>
                </div>
              )}
              {isApiKeysRoute && <ApiKeysPage />}
              {isUsageRoute && <GatewayUsagePage />}
              {isBudgetsRoute && <BudgetsPage />}
            </>
          )}
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, GatewayPage);
