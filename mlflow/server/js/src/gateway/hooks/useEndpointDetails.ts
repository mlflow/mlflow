import { useMemo, useCallback, useState } from 'react';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useEndpointQuery } from './useEndpointQuery';
import { useModelsQuery } from './useModelsQuery';
import { useBindingsQuery } from './useBindingsQuery';
import GatewayRoutes from '../routes';
import type { EndpointBinding } from '../types';

/**
 * Hook containing all business logic for the Endpoint Details page.
 * Handles data fetching, derived state, and navigation.
 */
export function useEndpointDetails(endpointId: string) {
  const navigate = useNavigate();

  const { data, error, isLoading } = useEndpointQuery(endpointId);
  const endpoint = data?.endpoint;

  // Delete modal state
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  // Get the primary model mapping and its model definition (memoized)
  const primaryMapping = useMemo(() => endpoint?.model_mappings?.[0], [endpoint?.model_mappings]);
  const primaryModelDef = useMemo(() => primaryMapping?.model_definition, [primaryMapping?.model_definition]);

  // Fetch models metadata for the primary provider
  const { data: modelsData } = useModelsQuery({ provider: primaryModelDef?.provider });

  // Get bindings for this endpoint (memoized)
  const { data: allBindings } = useBindingsQuery();
  const endpointBindings: EndpointBinding[] = useMemo(
    () => allBindings?.filter((b) => b.endpoint_id === endpointId) ?? [],
    [allBindings, endpointId],
  );

  // Navigation handler
  const handleEdit = useCallback(() => {
    navigate(GatewayRoutes.getEditEndpointRoute(endpointId));
  }, [navigate, endpointId]);

  // Delete modal handlers
  const handleDeleteClick = useCallback(() => {
    setIsDeleteModalOpen(true);
  }, []);

  const handleDeleteModalClose = useCallback(() => {
    setIsDeleteModalOpen(false);
  }, []);

  const handleDeleteSuccess = useCallback(() => {
    navigate(GatewayRoutes.gatewayPageRoute);
  }, [navigate]);

  // Derived state
  const hasModels = Boolean(endpoint?.model_mappings && endpoint.model_mappings.length > 0);

  return {
    // Data
    endpoint,
    modelsData,
    endpointBindings,

    // Derived state
    hasModels,

    // Loading/error state
    isLoading,
    error,

    // Delete modal state
    isDeleteModalOpen,

    // Handlers
    handleEdit,
    handleDeleteClick,
    handleDeleteModalClose,
    handleDeleteSuccess,
  };
}
