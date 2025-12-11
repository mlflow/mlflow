import { useState, useCallback } from 'react';
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
import { useModelDefinitionsQuery } from '../hooks/useModelDefinitionsQuery';
import type { SecretInfo, Endpoint, EndpointBinding, ModelDefinition } from '../types';

const ApiKeysPage = () => {
  const { theme } = useDesignSystemTheme();
  const { refetch: refetchSecrets } = useSecretsQuery();
  const { data: allEndpoints, refetch: refetchEndpoints } = useEndpointsQuery();
  const { data: allBindings } = useBindingsQuery();
  const { data: allModelDefinitions, refetch: refetchModelDefinitions } = useModelDefinitionsQuery();

  // Memoize helper to get model definitions using a secret
  const getModelDefinitionsForSecret = useCallback(
    (secretId: string): ModelDefinition[] => {
      if (!allModelDefinitions) return [];
      return allModelDefinitions.filter((modelDef) => modelDef.secret_id === secretId);
    },
    [allModelDefinitions],
  );

  // Memoize helper to get binding count for a secret (via endpoints that use model definitions with this secret)
  const getBindingCountForSecret = useCallback(
    (secretId: string): number => {
      if (!allBindings || !allEndpoints) return 0;
      const endpointIds = new Set(
        allEndpoints
          .filter((endpoint) =>
            endpoint.model_mappings?.some((mapping) => mapping.model_definition?.secret_id === secretId),
          )
          .map((endpoint) => endpoint.endpoint_id),
      );
      return allBindings.filter((binding) => endpointIds.has(binding.endpoint_id)).length;
    },
    [allBindings, allEndpoints],
  );

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [selectedSecret, setSelectedSecret] = useState<SecretInfo | null>(null);
  const [editingSecret, setEditingSecret] = useState<SecretInfo | null>(null);
  const [deleteModalData, setDeleteModalData] = useState<{
    secret: SecretInfo;
    modelDefinitions: ModelDefinition[];
    bindingCount: number;
  } | null>(null);
  const [endpointsDrawerData, setEndpointsDrawerData] = useState<{
    secret: SecretInfo;
    endpoints: Endpoint[];
  } | null>(null);
  const [bindingsDrawerData, setBindingsDrawerData] = useState<{
    secret: SecretInfo;
    bindings: EndpointBinding[];
  } | null>(null);

  const handleKeyClick = useCallback((secret: SecretInfo) => {
    setSelectedSecret(secret);
  }, []);

  const handleDrawerClose = useCallback(() => {
    setSelectedSecret(null);
  }, []);

  const handleEditClick = useCallback((secret: SecretInfo) => {
    setEditingSecret(secret);
  }, []);

  const handleEditModalClose = useCallback(() => {
    setEditingSecret(null);
  }, []);

  const handleEditSuccess = useCallback(() => {
    refetchSecrets();
    // Update selectedSecret if it was the one being edited
    if (selectedSecret && editingSecret && selectedSecret.secret_id === editingSecret.secret_id) {
      // Close and reopen will refresh the data
      setSelectedSecret(null);
    }
  }, [refetchSecrets, selectedSecret, editingSecret]);

  const handleDeleteClick = useCallback(
    (secret: SecretInfo, modelDefinitions: ModelDefinition[], bindingCount: number) => {
      setDeleteModalData({ secret, modelDefinitions, bindingCount });
    },
    [],
  );

  const handleDeleteFromDrawer = useCallback(
    (secret: SecretInfo) => {
      const modelDefinitions = getModelDefinitionsForSecret(secret.secret_id);
      const bindingCount = getBindingCountForSecret(secret.secret_id);
      setDeleteModalData({ secret, modelDefinitions, bindingCount });
    },
    [getModelDefinitionsForSecret, getBindingCountForSecret],
  );

  const handleDeleteModalClose = useCallback(() => {
    setDeleteModalData(null);
  }, []);

  const handleDeleteSuccess = useCallback(() => {
    refetchSecrets();
    refetchEndpoints();
    refetchModelDefinitions();
  }, [refetchSecrets, refetchEndpoints, refetchModelDefinitions]);

  const handleEndpointsClick = useCallback((secret: SecretInfo, endpoints: Endpoint[]) => {
    setEndpointsDrawerData({ secret, endpoints });
  }, []);

  const handleEndpointsDrawerClose = useCallback(() => {
    setEndpointsDrawerData(null);
  }, []);

  const handleBindingsClick = useCallback((secret: SecretInfo, bindings: EndpointBinding[]) => {
    setBindingsDrawerData({ secret, bindings });
  }, []);

  const handleBindingsDrawerClose = useCallback(() => {
    setBindingsDrawerData(null);
  }, []);

  const handleCreateClick = useCallback(() => {
    setIsCreateModalOpen(true);
  }, []);

  const handleCreateModalClose = useCallback(() => {
    setIsCreateModalOpen(false);
  }, []);

  const handleCreateSuccess = useCallback(() => {
    refetchSecrets();
  }, [refetchSecrets]);

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
        keyName={endpointsDrawerData?.secret.secret_name ?? ''}
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
        modelDefinitions={deleteModalData?.modelDefinitions ?? []}
        onClose={handleDeleteModalClose}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ApiKeysPage);
