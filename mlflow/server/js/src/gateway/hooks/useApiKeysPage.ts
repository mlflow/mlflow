import { useState, useCallback, useMemo } from 'react';
import { useSecretsQuery } from './useSecretsQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useBindingsQuery } from './useBindingsQuery';
import { useModelDefinitionsQuery } from './useModelDefinitionsQuery';
import type { SecretInfo, Endpoint, EndpointBinding, ModelDefinition } from '../types';

export interface DeleteModalData {
  secret: SecretInfo;
  modelDefinitions: ModelDefinition[];
  bindingCount: number;
}

export interface EndpointsDrawerData {
  secret: SecretInfo;
  endpoints: Endpoint[];
}

export interface BindingsDrawerData {
  secret: SecretInfo;
  bindings: EndpointBinding[];
}

/**
 * Hook containing all business logic for the API Keys page.
 * Handles state management, data fetching, and event handlers.
 */
export function useApiKeysPage() {
  const { refetch: refetchSecrets } = useSecretsQuery();
  const { data: allEndpoints, refetch: refetchEndpoints } = useEndpointsQuery();
  const { data: allBindings } = useBindingsQuery();
  const { data: allModelDefinitions, refetch: refetchModelDefinitions } = useModelDefinitionsQuery();

  // Modal/drawer state
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [selectedSecret, setSelectedSecret] = useState<SecretInfo | null>(null);
  const [editingSecret, setEditingSecret] = useState<SecretInfo | null>(null);
  const [deleteModalData, setDeleteModalData] = useState<DeleteModalData | null>(null);
  const [endpointsDrawerData, setEndpointsDrawerData] = useState<EndpointsDrawerData | null>(null);
  const [bindingsDrawerData, setBindingsDrawerData] = useState<BindingsDrawerData | null>(null);

  // Helper to get model definitions using a secret
  const getModelDefinitionsForSecret = useCallback(
    (secretId: string): ModelDefinition[] => {
      if (!allModelDefinitions) return [];
      return allModelDefinitions.filter((modelDef) => modelDef.secret_id === secretId);
    },
    [allModelDefinitions],
  );

  // Helper to get binding count for a secret (via endpoints that use model definitions with this secret)
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

  // Event handlers
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

  const handleDeleteSuccess = useCallback(async () => {
    // Refetch data after deletion - wrap in try-catch since backend may have
    // integrity issues if the deleted key was referenced by endpoints/model definitions
    await refetchSecrets();
    try {
      await refetchEndpoints();
    } catch {
      // Ignore errors - backend may fail if orphaned references exist
    }
    try {
      await refetchModelDefinitions();
    } catch {
      // Ignore errors - backend may fail if orphaned references exist
    }
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

  // Derived state for modals/drawers
  const isDetailsDrawerOpen = selectedSecret !== null;
  const isEditModalOpen = editingSecret !== null;
  const isDeleteModalOpen = deleteModalData !== null;
  const isEndpointsDrawerOpen = endpointsDrawerData !== null;
  const isBindingsDrawerOpen = bindingsDrawerData !== null;

  return {
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
  };
}
