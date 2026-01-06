import { useMemo, useCallback, useState } from 'react';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useEndpointQuery } from './useEndpointQuery';
import { useModelsQuery } from './useModelsQuery';
import { useBindingsQuery } from './useBindingsQuery';
import GatewayRoutes from '../routes';
import type { EndpointBinding } from '../types';

export function useEndpointDetails(endpointId: string) {
  const navigate = useNavigate();

  const { data, error, isLoading } = useEndpointQuery(endpointId);
  const endpoint = data?.endpoint;

  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const primaryMapping = useMemo(() => endpoint?.model_mappings?.[0], [endpoint?.model_mappings]);
  const primaryModelDef = useMemo(() => primaryMapping?.model_definition, [primaryMapping?.model_definition]);

  const { data: modelsData } = useModelsQuery({ provider: primaryModelDef?.provider });

  const { data: allBindings } = useBindingsQuery();
  const endpointBindings: EndpointBinding[] = useMemo(
    () => allBindings?.filter((b) => b.endpoint_id === endpointId) ?? [],
    [allBindings, endpointId],
  );

  const handleEdit = useCallback(() => {
    navigate(GatewayRoutes.getEditEndpointRoute(endpointId));
  }, [navigate, endpointId]);

  const handleDeleteClick = useCallback(() => {
    setIsDeleteModalOpen(true);
  }, []);

  const handleDeleteModalClose = useCallback(() => {
    setIsDeleteModalOpen(false);
  }, []);

  const handleDeleteSuccess = useCallback(() => {
    navigate(GatewayRoutes.gatewayPageRoute);
  }, [navigate]);

  const hasModels = Boolean(endpoint?.model_mappings && endpoint.model_mappings.length > 0);

  return {
    endpoint,
    modelsData,
    endpointBindings,
    hasModels,
    isLoading,
    error,
    isDeleteModalOpen,
    handleEdit,
    handleDeleteClick,
    handleDeleteModalClose,
    handleDeleteSuccess,
  };
}
