import { useState } from 'react';
import { Button, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ModelDefinitionsList } from '../components/model-definitions/ModelDefinitionsList';
import { CreateModelDefinitionModal } from '../components/model-definitions/CreateModelDefinitionModal';
import { DeleteModelDefinitionModal } from '../components/model-definitions/DeleteModelDefinitionModal';
import { EndpointsUsingModelDrawer } from '../components/model-definitions/EndpointsUsingModelDrawer';
import { useModelDefinitionsQuery } from '../hooks/useModelDefinitionsQuery';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import GatewayRoutes from '../routes';
import type { ModelDefinition, Endpoint } from '../types';

const ModelDefinitionsPage = () => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const { refetch: refetchModelDefinitions } = useModelDefinitionsQuery();
  const { refetch: refetchEndpoints } = useEndpointsQuery();

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [deleteModalData, setDeleteModalData] = useState<{
    modelDefinition: ModelDefinition;
    endpoints: Endpoint[];
  } | null>(null);
  const [endpointsDrawerData, setEndpointsDrawerData] = useState<{
    modelDefinition: ModelDefinition;
    endpoints: Endpoint[];
  } | null>(null);

  const handleModelDefinitionClick = (modelDefinition: ModelDefinition) => {
    navigate(GatewayRoutes.getModelDefinitionDetailsRoute(modelDefinition.model_definition_id));
  };

  const handleEditClick = (modelDefinition: ModelDefinition) => {
    navigate(GatewayRoutes.getEditModelDefinitionRoute(modelDefinition.model_definition_id));
  };

  const handleDeleteClick = (modelDefinition: ModelDefinition, endpoints: Endpoint[]) => {
    setDeleteModalData({ modelDefinition, endpoints });
  };

  const handleDeleteModalClose = () => {
    setDeleteModalData(null);
  };

  const handleDeleteSuccess = () => {
    refetchModelDefinitions();
    refetchEndpoints();
  };

  const handleEndpointsClick = (modelDefinition: ModelDefinition, endpoints: Endpoint[]) => {
    setEndpointsDrawerData({ modelDefinition, endpoints });
  };

  const handleEndpointsDrawerClose = () => {
    setEndpointsDrawerData(null);
  };

  const handleCreateClick = () => {
    setIsCreateModalOpen(true);
  };

  const handleCreateModalClose = () => {
    setIsCreateModalOpen(false);
  };

  const handleCreateSuccess = () => {
    refetchModelDefinitions();
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
          <FormattedMessage defaultMessage="Models" description="Models page title" />
        </Typography.Title>
        <Button
          componentId="mlflow.gateway.models.create-button"
          type="primary"
          icon={<PlusIcon />}
          onClick={handleCreateClick}
        >
          <FormattedMessage defaultMessage="Create Model" description="Create model button" />
        </Button>
      </div>

      {/* Content */}
      <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
        <ModelDefinitionsList
          onModelDefinitionClick={handleModelDefinitionClick}
          onEditClick={handleEditClick}
          onDeleteClick={handleDeleteClick}
          onEndpointsClick={handleEndpointsClick}
        />
      </div>

      {/* Delete Confirmation Modal */}
      <DeleteModelDefinitionModal
        open={deleteModalData !== null}
        modelDefinition={deleteModalData?.modelDefinition ?? null}
        endpoints={deleteModalData?.endpoints ?? []}
        onClose={handleDeleteModalClose}
        onSuccess={handleDeleteSuccess}
      />

      {/* Create Modal */}
      <CreateModelDefinitionModal
        open={isCreateModalOpen}
        onClose={handleCreateModalClose}
        onSuccess={handleCreateSuccess}
      />

      {/* Endpoints Using Model Drawer */}
      <EndpointsUsingModelDrawer
        open={endpointsDrawerData !== null}
        endpoints={endpointsDrawerData?.endpoints ?? []}
        onClose={handleEndpointsDrawerClose}
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ModelDefinitionsPage);
