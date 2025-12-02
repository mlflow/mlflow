import { useState } from 'react';
import { Alert, Button, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDeleteModelDefinitionMutation } from '../../hooks/useDeleteModelDefinitionMutation';
import { useDetachModelFromEndpointMutation } from '../../hooks/useDeleteEndpointModelMutation';
import type { ModelDefinition, Endpoint } from '../../types';

interface DeleteModelDefinitionModalProps {
  open: boolean;
  modelDefinition: ModelDefinition | null;
  endpoints: Endpoint[];
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteModelDefinitionModal = ({
  open,
  modelDefinition,
  endpoints,
  onClose,
  onSuccess,
}: DeleteModelDefinitionModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [confirmationText, setConfirmationText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const { mutateAsync: deleteModelDefinition, isLoading: isDeletingModelDefinition } =
    useDeleteModelDefinitionMutation();
  const { mutateAsync: detachModel, isLoading: isDetaching } = useDetachModelFromEndpointMutation();

  const endpointCount = endpoints.length;
  const hasEndpoints = endpointCount > 0;
  const isConfirmed = !hasEndpoints || confirmationText === modelDefinition?.name;
  const isDeleting = isDeletingModelDefinition || isDetaching;

  const handleDelete = async () => {
    if (!modelDefinition || !isConfirmed) return;

    setError(null);

    try {
      // First, detach this model definition from all endpoints
      if (hasEndpoints) {
        const detachRequests = endpoints.map((endpoint) => ({
          endpoint_id: endpoint.endpoint_id,
          model_definition_id: modelDefinition.model_definition_id,
        }));

        await Promise.all(detachRequests.map((req) => detachModel(req)));
      }

      // Then delete the model definition itself
      await deleteModelDefinition(modelDefinition.model_definition_id);

      handleClose();
      onSuccess?.();
    } catch (err) {
      setError(
        intl.formatMessage({
          defaultMessage: 'Failed to delete model. Please try again.',
          description: 'Error message when model deletion fails',
        }),
      );
    }
  };

  const handleClose = () => {
    setConfirmationText('');
    setError(null);
    onClose();
  };

  if (!modelDefinition) return null;

  return (
    <Modal
      componentId="mlflow.gateway.delete-model-definition-modal"
      title={intl.formatMessage({
        defaultMessage: 'Delete Model',
        description: 'Title for delete model modal',
      })}
      visible={open}
      onCancel={handleClose}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.gateway.delete-model-definition-modal.cancel"
            onClick={handleClose}
            disabled={isDeleting}
          >
            <FormattedMessage defaultMessage="Cancel" description="Cancel button text" />
          </Button>
          <Button
            componentId="mlflow.gateway.delete-model-definition-modal.delete"
            type="primary"
            danger={hasEndpoints}
            onClick={handleDelete}
            disabled={!isConfirmed || isDeleting}
            loading={isDeleting}
          >
            <FormattedMessage defaultMessage="Delete" description="Delete button text" />
          </Button>
        </div>
      }
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {error && (
          <Alert
            componentId="mlflow.gateway.delete-model-definition-modal.error"
            type="error"
            message={error}
            closable={false}
          />
        )}

        <Typography.Text>
          <FormattedMessage
            defaultMessage='Are you sure you want to delete the model "{modelName}"?'
            description="Delete confirmation message"
            values={{ modelName: <strong>{modelDefinition.name}</strong> }}
          />
        </Typography.Text>

        {hasEndpoints && (
          <Alert
            componentId="mlflow.gateway.delete-model-definition-modal.warning"
            type="warning"
            message={
              <FormattedMessage
                defaultMessage="This model is used by {count, plural, one {# endpoint} other {# endpoints}}. Deleting it will remove it from {count, plural, one {this endpoint} other {these endpoints}}."
                description="Warning about endpoints using this model"
                values={{ count: endpointCount }}
              />
            }
            closable={false}
          />
        )}

        {hasEndpoints && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text>
              <FormattedMessage
                defaultMessage="Type {modelName} to confirm deletion:"
                description="Type to confirm instruction"
                values={{ modelName: <strong>{modelDefinition.name}</strong> }}
              />
            </Typography.Text>
            <Input
              componentId="mlflow.gateway.delete-model-definition-modal.confirmation-input"
              value={confirmationText}
              onChange={(e) => setConfirmationText(e.target.value)}
              placeholder={modelDefinition.name}
              disabled={isDeleting}
            />
          </div>
        )}
      </div>
    </Modal>
  );
};
