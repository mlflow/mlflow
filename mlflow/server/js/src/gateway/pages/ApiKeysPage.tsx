import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ApiKeysList } from '../components/api-keys/ApiKeysList';
import { CreateApiKeyModal } from '../components/api-keys/CreateApiKeyModal';
import { ApiKeyDetailsDrawer } from '../components/api-keys/ApiKeyDetailsDrawer';
import { EndpointsUsingKeyDrawer } from '../components/api-keys/EndpointsUsingKeyDrawer';
import { BindingsUsingKeyDrawer } from '../components/api-keys/BindingsUsingKeyDrawer';
import { useApiKeysPage } from '../hooks/useApiKeysPage';

export function ApiKeysPageInner() {
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
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        overflow: 'hidden',
        width: '100%',
      }}
    >
      <div css={{ flex: 1, overflow: 'auto' }}>
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
}

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ApiKeysPageInner);
