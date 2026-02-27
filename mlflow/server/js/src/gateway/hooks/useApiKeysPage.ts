import { useState, useCallback } from 'react';
import { useSecretsQuery } from './useSecretsQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useBindingsQuery } from './useBindingsQuery';
import { useModelDefinitionsQuery } from './useModelDefinitionsQuery';
import type { SecretInfo, Endpoint, EndpointBinding, ModelDefinition } from '../types';

export interface DeleteModalData {
  secret: SecretInfo;
  modelDefinitions: ModelDefinition[];
  endpoints: Endpoint[];
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
 * Manages modal/drawer state and coordinates data fetching.
 */
export function useApiKeysPage() {
  const { refetch: refetchSecrets } = useSecretsQuery();
  const { data: allEndpoints, refetch: refetchEndpoints } = useEndpointsQuery();
  const { data: allBindings } = useBindingsQuery();
  const { data: allModelDefinitions, refetch: refetchModelDefinitions } = useModelDefinitionsQuery();

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [selectedSecret, setSelectedSecret] = useState<SecretInfo | null>(null);
  const [editingSecret, setEditingSecret] = useState<SecretInfo | null>(null);
  const [deleteModalData, setDeleteModalData] = useState<DeleteModalData | null>(null);
  const [endpointsDrawerData, setEndpointsDrawerData] = useState<EndpointsDrawerData | null>(null);
  const [bindingsDrawerData, setBindingsDrawerData] = useState<BindingsDrawerData | null>(null);

  const getModelDefinitionsForSecret = useCallback(
    (secretId: string): ModelDefinition[] => {
      if (!allModelDefinitions) return [];
      return allModelDefinitions.filter((modelDef) => modelDef.secret_id === secretId);
    },
    [allModelDefinitions],
  );

  const getEndpointsForSecret = useCallback(
    (secretId: string): Endpoint[] => {
      if (!allEndpoints) return [];
      return allEndpoints.filter((endpoint) =>
        endpoint.model_mappings?.some((mapping) => mapping.model_definition?.secret_id === secretId),
      );
    },
    [allEndpoints],
  );

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

  // Create modal handlers
  const handleCreateClick = useCallback(() => {
    setIsCreateModalOpen(true);
  }, []);

  const handleCreateModalClose = useCallback(() => {
    setIsCreateModalOpen(false);
  }, []);

  const handleCreateSuccess = useCallback(() => {
    refetchSecrets();
  }, [refetchSecrets]);

  // Details drawer handlers
  const handleKeyClick = useCallback((secret: SecretInfo) => {
    setSelectedSecret(secret);
  }, []);

  const handleDrawerClose = useCallback(() => {
    setSelectedSecret(null);
  }, []);

  // Edit modal handlers
  const handleEditClick = useCallback((secret: SecretInfo) => {
    setEditingSecret(secret);
  }, []);

  const handleEditModalClose = useCallback(() => {
    setEditingSecret(null);
  }, []);

  const handleEditSuccess = useCallback(() => {
    refetchSecrets();
    if (selectedSecret && editingSecret && selectedSecret.secret_id === editingSecret.secret_id) {
      setSelectedSecret(null);
    }
  }, [refetchSecrets, selectedSecret, editingSecret]);

  // Delete modal handlers
  const handleDeleteClick = useCallback(
    (secret: SecretInfo, modelDefinitions: ModelDefinition[], endpoints: Endpoint[], bindingCount: number) => {
      setDeleteModalData({ secret, modelDefinitions, endpoints, bindingCount });
    },
    [],
  );

  const handleDeleteFromDrawer = useCallback(
    (secret: SecretInfo) => {
      const modelDefinitions = getModelDefinitionsForSecret(secret.secret_id);
      const endpoints = getEndpointsForSecret(secret.secret_id);
      const bindingCount = getBindingCountForSecret(secret.secret_id);
      setDeleteModalData({ secret, modelDefinitions, endpoints, bindingCount });
    },
    [getModelDefinitionsForSecret, getEndpointsForSecret, getBindingCountForSecret],
  );

  const handleDeleteModalClose = useCallback(() => {
    setDeleteModalData(null);
  }, []);

  const handleDeleteSuccess = useCallback(async () => {
    await refetchSecrets();
    // Refetch related data - errors are ignored since backend may have
    // integrity issues if the deleted key was referenced elsewhere
    await Promise.allSettled([refetchEndpoints(), refetchModelDefinitions()]);
  }, [refetchSecrets, refetchEndpoints, refetchModelDefinitions]);

  // Endpoints drawer handlers
  const handleEndpointsClick = useCallback((secret: SecretInfo, endpoints: Endpoint[]) => {
    setEndpointsDrawerData({ secret, endpoints });
  }, []);

  const handleEndpointsDrawerClose = useCallback(() => {
    setEndpointsDrawerData(null);
  }, []);

  // Bindings drawer handlers
  const handleBindingsClick = useCallback((secret: SecretInfo, bindings: EndpointBinding[]) => {
    setBindingsDrawerData({ secret, bindings });
  }, []);

  const handleBindingsDrawerClose = useCallback(() => {
    setBindingsDrawerData(null);
  }, []);

  return {
    // Data
    allEndpoints,

    // Modal/drawer open state (derived from data)
    isCreateModalOpen,
    isDetailsDrawerOpen: selectedSecret !== null,
    isEditModalOpen: editingSecret !== null,
    isDeleteModalOpen: deleteModalData !== null,
    isEndpointsDrawerOpen: endpointsDrawerData !== null,
    isBindingsDrawerOpen: bindingsDrawerData !== null,

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
