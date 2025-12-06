import { useState } from 'react';
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
import { useSecretsQuery } from '../hooks/useSecretsQuery';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { useBindingsQuery } from '../hooks/useBindingsQuery';
import type { Secret, Endpoint, EndpointBinding } from '../types';

const ApiKeysPage = () => {
  const { theme } = useDesignSystemTheme();
  const { refetch: refetchSecrets } = useSecretsQuery();
  const { data: allEndpoints, refetch: refetchEndpoints } = useEndpointsQuery();
  const { data: allBindings } = useBindingsQuery();

  // Helper to get endpoints using a secret
  const getEndpointsForSecret = (secretId: string): Endpoint[] => {
    if (!allEndpoints) return [];
    return allEndpoints.filter((endpoint) =>
      endpoint.model_mappings?.some((mapping) => mapping.model_definition?.secret_id === secretId),
    );
  };

  // Helper to get binding count for a secret
  const getBindingCountForSecret = (secretId: string): number => {
    if (!allBindings || !allEndpoints) return 0;
    const endpointIds = new Set(
      allEndpoints
        .filter((endpoint) =>
          endpoint.model_mappings?.some((mapping) => mapping.model_definition?.secret_id === secretId),
        )
        .map((endpoint) => endpoint.endpoint_id),
    );
    return allBindings.filter((binding) => endpointIds.has(binding.endpoint_id)).length;
  };
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [selectedSecret, setSelectedSecret] = useState<Secret | null>(null);
  const [editingSecret, setEditingSecret] = useState<Secret | null>(null);
  const [deleteModalData, setDeleteModalData] = useState<{
    secret: Secret;
    endpoints: Endpoint[];
    bindingCount: number;
  } | null>(null);
  const [endpointsDrawerData, setEndpointsDrawerData] = useState<{
    secret: Secret;
    endpoints: Endpoint[];
  } | null>(null);
  const [bindingsDrawerData, setBindingsDrawerData] = useState<{
    secret: Secret;
    bindings: EndpointBinding[];
  } | null>(null);

  const handleKeyClick = (secret: Secret) => {
    setSelectedSecret(secret);
  };

  const handleDrawerClose = () => {
    setSelectedSecret(null);
  };

  const handleEditClick = (secret: Secret) => {
    setEditingSecret(secret);
  };

  const handleEditModalClose = () => {
    setEditingSecret(null);
  };

  const handleEditSuccess = () => {
    refetchSecrets();
    // Update selectedSecret if it was the one being edited
    if (selectedSecret && editingSecret && selectedSecret.secret_id === editingSecret.secret_id) {
      // Close and reopen will refresh the data
      setSelectedSecret(null);
    }
  };

  const handleDeleteClick = (secret: Secret, endpoints: Endpoint[], bindingCount: number) => {
    setDeleteModalData({ secret, endpoints, bindingCount });
  };

  const handleDeleteFromDrawer = (secret: Secret) => {
    const endpoints = getEndpointsForSecret(secret.secret_id);
    const bindingCount = getBindingCountForSecret(secret.secret_id);
    setDeleteModalData({ secret, endpoints, bindingCount });
  };

  const handleDeleteModalClose = () => {
    setDeleteModalData(null);
  };

  const handleDeleteSuccess = () => {
    refetchSecrets();
    refetchEndpoints();
  };

  const handleEndpointsClick = (secret: Secret, endpoints: Endpoint[]) => {
    setEndpointsDrawerData({ secret, endpoints });
  };

  const handleEndpointsDrawerClose = () => {
    setEndpointsDrawerData(null);
  };

  const handleBindingsClick = (secret: Secret, bindings: EndpointBinding[]) => {
    setBindingsDrawerData({ secret, bindings });
  };

  const handleBindingsDrawerClose = () => {
    setBindingsDrawerData(null);
  };

  const handleCreateClick = () => {
    setIsCreateModalOpen(true);
  };

  const handleCreateModalClose = () => {
    setIsCreateModalOpen(false);
  };

  const handleCreateSuccess = () => {
    refetchSecrets();
  };

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
          <FormattedMessage defaultMessage="Create API key" description="Create API key button" />
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
        open={editingSecret !== null}
        secret={editingSecret}
        onClose={handleEditModalClose}
        onSuccess={handleEditSuccess}
      />

      {/* Details Drawer */}
      <ApiKeyDetailsDrawer
        open={selectedSecret !== null}
        secret={selectedSecret}
        onClose={handleDrawerClose}
        onEdit={handleEditClick}
        onDelete={handleDeleteFromDrawer}
      />

      {/* Endpoints Using Key Drawer */}
      <EndpointsUsingKeyDrawer
        open={endpointsDrawerData !== null}
        endpoints={endpointsDrawerData?.endpoints ?? []}
        onClose={handleEndpointsDrawerClose}
      />

      {/* Bindings Using Key Drawer */}
      <BindingsUsingKeyDrawer
        open={bindingsDrawerData !== null}
        bindings={bindingsDrawerData?.bindings ?? []}
        endpoints={allEndpoints ?? []}
        onClose={handleBindingsDrawerClose}
      />

      {/* Delete Confirmation Modal */}
      <DeleteApiKeyModal
        open={deleteModalData !== null}
        secret={deleteModalData?.secret ?? null}
        endpoints={deleteModalData?.endpoints ?? []}
        bindingCount={deleteModalData?.bindingCount ?? 0}
        onClose={handleDeleteModalClose}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ApiKeysPage);
