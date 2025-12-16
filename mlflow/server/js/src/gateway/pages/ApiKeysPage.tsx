import { Button, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ApiKeysList } from '../components/api-keys/ApiKeysList';
import { CreateApiKeyModal } from '../components/api-keys/CreateApiKeyModal';
import { EditApiKeyModal } from '../components/api-keys/EditApiKeyModal';
import { DeleteApiKeyModal } from '../components/api-keys/DeleteApiKeyModal';
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
    isEditModalOpen,
    isDeleteModalOpen,
    isEndpointsDrawerOpen,
    isBindingsDrawerOpen,

    // Modal/drawer data
    selectedSecret,
    editingSecret,
    deleteModalData,
    endpointsDrawerData,
    bindingsDrawerData,

    // Handlers for ApiKeysList
    handleKeyClick,
    handleEditClick,
    handleDeleteClick,
    handleEndpointsClick,
    handleBindingsClick,

    // Handlers for Create modal
    handleCreateClick,
    handleCreateModalClose,
    handleCreateSuccess,

    // Handlers for Details drawer
    handleDrawerClose,
    handleDeleteFromDrawer,

    // Handlers for Edit modal
    handleEditModalClose,
    handleEditSuccess,

    // Handlers for Delete modal
    handleDeleteModalClose,
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
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="API Keys" description="API Keys page title" />
        </Typography.Title>
        <Button
          componentId="mlflow.gateway.api-keys.create-button"
          type="primary"
          icon={<PlusIcon />}
          onClick={handleCreateClick}
        >
          <FormattedMessage
            defaultMessage="Create API key"
            description="Gateway > API keys page > Create API key button"
          />
        </Button>
      </div>

      {/* Content */}
      <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
        <ApiKeysList
          onKeyClick={handleKeyClick}
          onEditClick={handleEditClick}
          onDeleteClick={handleDeleteClick}
          onEndpointsClick={handleEndpointsClick}
          onBindingsClick={handleBindingsClick}
        />
      </div>

      {/* Create Modal */}
      <CreateApiKeyModal open={isCreateModalOpen} onClose={handleCreateModalClose} onSuccess={handleCreateSuccess} />

      {/* Edit Modal */}
      <EditApiKeyModal
        open={isEditModalOpen}
        secret={editingSecret}
        onClose={handleEditModalClose}
        onSuccess={handleEditSuccess}
      />

      {/* Details Drawer */}
      <ApiKeyDetailsDrawer
        open={isDetailsDrawerOpen}
        secret={selectedSecret}
        onClose={handleDrawerClose}
        onEdit={handleEditClick}
        onDelete={handleDeleteFromDrawer}
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

      {/* Delete Confirmation Modal */}
      <DeleteApiKeyModal
        open={isDeleteModalOpen}
        secret={deleteModalData?.secret ?? null}
        modelDefinitions={deleteModalData?.modelDefinitions ?? []}
        onClose={handleDeleteModalClose}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ApiKeysPage);
