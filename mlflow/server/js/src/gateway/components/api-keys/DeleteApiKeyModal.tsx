import { useState } from 'react';
import { Alert, Button, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDeleteSecretMutation } from '../../hooks/useDeleteSecretMutation';
import { useDetachModelFromEndpointMutation } from '../../hooks/useDeleteEndpointModelMutation';
import type { Secret, Endpoint } from '../../types';

interface DeleteApiKeyModalProps {
  open: boolean;
  secret: Secret | null;
  endpoints: Endpoint[];
  bindingCount: number;
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteApiKeyModal = ({
  open,
  secret,
  endpoints,
  bindingCount,
  onClose,
  onSuccess,
}: DeleteApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [confirmationText, setConfirmationText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const { mutateAsync: deleteSecret, isLoading: isDeletingSecret } = useDeleteSecretMutation();
  const { mutateAsync: detachModel, isLoading: isDetaching } = useDetachModelFromEndpointMutation();

  const endpointCount = endpoints.length;
  const hasEndpoints = endpointCount > 0;
  const isConfirmed = !hasEndpoints || confirmationText === secret?.secret_name;
  const isDeleting = isDeletingSecret || isDetaching;

  const handleDelete = async () => {
    if (!secret || !isConfirmed) return;

    setError(null);

    try {
      // First, detach all model definitions that use this secret from their endpoints
      if (hasEndpoints) {
        const detachRequests: { endpoint_id: string; model_definition_id: string }[] = [];
        endpoints.forEach((endpoint) => {
          endpoint.model_mappings?.forEach((mapping) => {
            if (mapping.model_definition?.secret_id === secret.secret_id) {
              detachRequests.push({
                endpoint_id: endpoint.endpoint_id,
                model_definition_id: mapping.model_definition.model_definition_id,
              });
            }
          });
        });

        await Promise.all(detachRequests.map((req) => detachModel(req)));
      }

      // Then delete the secret itself
      await deleteSecret(secret.secret_id);

      handleClose();
      onSuccess?.();
    } catch (err) {
      setError(
        intl.formatMessage({
          defaultMessage: 'Failed to delete API key. Please try again.',
          description: 'Error message when API key deletion fails',
        }),
      );
    }
  };

  const handleClose = () => {
    setConfirmationText('');
    setError(null);
    onClose();
  };

  if (!secret) return null;

  return (
    <Modal
      componentId="mlflow.gateway.delete-api-key-modal"
      title={intl.formatMessage({
        defaultMessage: 'Delete API Key',
        description: 'Title for delete API key modal',
      })}
      visible={open}
      onCancel={handleClose}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.gateway.delete-api-key-modal.cancel" onClick={handleClose} disabled={isDeleting}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button text" />
          </Button>
          <Button
            componentId="mlflow.gateway.delete-api-key-modal.delete"
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
            componentId="mlflow.gateway.delete-api-key-modal.error"
            type="error"
            message={error}
            closable={false}
          />
        )}

        <Typography.Text>
          <FormattedMessage
            defaultMessage='Are you sure you want to delete the API key "{keyName}"?'
            description="Delete confirmation message"
            values={{ keyName: <strong>{secret.secret_name}</strong> }}
          />
        </Typography.Text>

        {hasEndpoints && (
          <Alert
            componentId="mlflow.gateway.delete-api-key-modal.warning"
            type="warning"
            message={
              bindingCount > 0 ? (
                <FormattedMessage
                  defaultMessage="This key is used by {endpointCount, plural, one {# endpoint} other {# endpoints}} and {bindingCount, plural, one {# resource} other {# resources}}. Deleting it will remove {endpointCount, plural, one {this endpoint} other {these endpoints}}."
                  description="Warning about endpoints and bindings using this key"
                  values={{ endpointCount, bindingCount }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="This key is used by {count, plural, one {# endpoint} other {# endpoints}}. Deleting it will remove {count, plural, one {this endpoint} other {these endpoints}}."
                  description="Warning about endpoints using this key"
                  values={{ count: endpointCount }}
                />
              )
            }
            closable={false}
          />
        )}

        {hasEndpoints && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text>
              <FormattedMessage
                defaultMessage="Type {keyName} to confirm deletion:"
                description="Type to confirm instruction"
                values={{ keyName: <strong>{secret.secret_name}</strong> }}
              />
            </Typography.Text>
            <Input
              componentId="mlflow.gateway.delete-api-key-modal.confirmation-input"
              value={confirmationText}
              onChange={(e) => setConfirmationText(e.target.value)}
              placeholder={secret.secret_name}
              disabled={isDeleting}
            />
          </div>
        )}
      </div>
    </Modal>
  );
};
