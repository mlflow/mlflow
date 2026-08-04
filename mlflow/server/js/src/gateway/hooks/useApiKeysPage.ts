import { useState, useCallback, useMemo } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useSecretsQuery } from './useSecretsQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useModelDefinitionsQuery } from './useModelDefinitionsQuery';
import type { SecretInfo, Endpoint, EndpointBinding } from '../types';
import { useLogTelemetryEvent } from '@mlflow/mlflow/src/telemetry/hooks/useLogTelemetryEvent';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
} from '@databricks/design-system';

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
  const { refetch: refetchModelDefinitions } = useModelDefinitionsQuery();
  const logTelemetryEvent = useLogTelemetryEvent();
  const viewId = useMemo(() => uuidv4(), []);

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [selectedSecret, setSelectedSecret] = useState<SecretInfo | null>(null);
  const [endpointsDrawerData, setEndpointsDrawerData] = useState<EndpointsDrawerData | null>(null);
  const [bindingsDrawerData, setBindingsDrawerData] = useState<BindingsDrawerData | null>(null);

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
  const handleKeyClick = useCallback(
    (secret: SecretInfo) => {
      setSelectedSecret(secret);
      logTelemetryEvent({
        componentId: 'mlflow.gateway.api-keys.list.row',
        componentViewId: viewId,
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentSubType: null,
        eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
      });
    },
    [logTelemetryEvent, viewId],
  );

  const handleDrawerClose = useCallback(() => {
    setSelectedSecret(null);
  }, []);

  // Edit success handler (inline editing in drawer)
  const handleEditSuccess = useCallback(async () => {
    const result = await refetchSecrets();
    if (selectedSecret && result.data) {
      const updated = result.data.secrets.find((s) => s.secret_id === selectedSecret.secret_id);
      if (updated) {
        setSelectedSecret(updated);
      }
    }
  }, [refetchSecrets, selectedSecret]);

  // Delete success handler (shared by bulk delete)
  const handleDeleteSuccess = useCallback(async () => {
    await refetchSecrets();
    // Refetch related data - errors are ignored since backend may have
    // integrity issues if the deleted key was referenced elsewhere
    await Promise.allSettled([refetchEndpoints(), refetchModelDefinitions()]);
  }, [refetchSecrets, refetchEndpoints, refetchModelDefinitions]);

  // Endpoints drawer handlers
  const handleEndpointsClick = useCallback(
    (secret: SecretInfo, endpoints: Endpoint[]) => {
      setEndpointsDrawerData({ secret, endpoints });
      logTelemetryEvent({
        componentId: 'mlflow.gateway.api-keys.list.endpoints-link',
        componentViewId: viewId,
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentSubType: null,
        eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
      });
    },
    [logTelemetryEvent, viewId],
  );

  const handleEndpointsDrawerClose = useCallback(() => {
    setEndpointsDrawerData(null);
  }, []);

  // Bindings drawer handlers
  const handleBindingsClick = useCallback(
    (secret: SecretInfo, bindings: EndpointBinding[]) => {
      setBindingsDrawerData({ secret, bindings });
      logTelemetryEvent({
        componentId: 'mlflow.gateway.api-keys.list.used-by-link',
        componentViewId: viewId,
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentSubType: null,
        eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
      });
    },
    [logTelemetryEvent, viewId],
  );

  const handleBindingsDrawerClose = useCallback(() => {
    setBindingsDrawerData(null);
  }, []);

  return {
    // Data
    allEndpoints,

    // Modal/drawer open state (derived from data)
    isCreateModalOpen,
    isDetailsDrawerOpen: selectedSecret !== null,
    isEndpointsDrawerOpen: endpointsDrawerData !== null,
    isBindingsDrawerOpen: bindingsDrawerData !== null,

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
  };
}
