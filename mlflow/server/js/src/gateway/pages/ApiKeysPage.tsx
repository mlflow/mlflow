import { Breadcrumb, KeyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../common/utils/RoutingUtils';
import { GatewayLabel } from '../../common/components/GatewayNewTag';
import GatewayRoutes from '../routes';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ApiKeysList } from '../components/api-keys/ApiKeysList';
import { CreateApiKeyModal } from '../components/api-keys/CreateApiKeyModal';
import { ApiKeyDetailsDrawer } from '../components/api-keys/ApiKeyDetailsDrawer';
import { EndpointsUsingKeyDrawer } from '../components/api-keys/EndpointsUsingKeyDrawer';
import { BindingsUsingKeyDrawer } from '../components/api-keys/BindingsUsingKeyDrawer';
import { useApiKeysPage } from '../hooks/useApiKeysPage';

/**
 * Container component for the API Keys page.
 * Uses the container/renderer pattern:
 * - useApiKeysPage: Contains all business logic (state, data fetching, handlers)
 * - This component: Handles page layout and renders child components
 */
const ApiKeysPage = () => {
  const { theme } = useDesignSystemTheme();

  const {
    // Data
    allEndpoints,

    // Modal/drawer state
    isCreateModalOpen,
    isDetailsDrawerOpen,
    isEndpointsDrawerOpen,
    isBindingsDrawerOpen,

    // Modal/drawer data
    selectedSecret,
    endpointsDrawerData,
    bindingsDrawerData,

    // Handlers for ApiKeysList
    handleKeyClick,
    handleEndpointsClick,
    handleBindingsClick,

    // Handlers for Create modal
    handleCreateClick,
    handleCreateModalClose,
    handleCreateSuccess,

    // Handlers for Details drawer
    handleDrawerClose,
    handleEditSuccess,

    // Handlers for Delete modal
    handleDeleteSuccess,

    // Handlers for Endpoints drawer
    handleEndpointsDrawerClose,

    // Handlers for Bindings drawer
    handleBindingsDrawerClose,
  } = useApiKeysPage();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      {/* Header */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: theme.spacing.md,
          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link componentId="mlflow.gateway.api-keys.breadcrumb_gateway_link" to={GatewayRoutes.gatewayPageRoute}>
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
              <KeyIcon />
            </div>
            <Typography.Title withoutMargins level={2}>
              <FormattedMessage defaultMessage="API Keys" description="API Keys page title" />
            </Typography.Title>
          </div>
        </div>
      </div>

      {/* Content */}
      <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
        <ApiKeysList
          onCreateClick={handleCreateClick}
          onKeyClick={handleKeyClick}
          onEndpointsClick={handleEndpointsClick}
          onBindingsClick={handleBindingsClick}
          onApiKeyDeleted={handleDeleteSuccess}
        />
      </div>

      {/* Create Modal */}
      <CreateApiKeyModal open={isCreateModalOpen} onClose={handleCreateModalClose} onSuccess={handleCreateSuccess} />

      {/* Details Drawer (with inline editing) */}
      <ApiKeyDetailsDrawer
        open={isDetailsDrawerOpen}
        secret={selectedSecret}
        onClose={handleDrawerClose}
        onEditSuccess={handleEditSuccess}
      />

      {/* Endpoints Using Key Drawer */}
      <EndpointsUsingKeyDrawer
        open={isEndpointsDrawerOpen}
        keyName={endpointsDrawerData?.secret.secret_name ?? ''}
        endpoints={endpointsDrawerData?.endpoints ?? []}
        onClose={handleEndpointsDrawerClose}
      />

      {/* Bindings Using Key Drawer */}
      <BindingsUsingKeyDrawer
        open={isBindingsDrawerOpen}
        bindings={bindingsDrawerData?.bindings ?? []}
        endpoints={allEndpoints ?? []}
        onClose={handleBindingsDrawerClose}
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ApiKeysPage);
