import { useState } from 'react';
import { Alert, Button, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDeleteSecretMutation } from '../../hooks/useDeleteSecretMutation';
import type { SecretInfo, ModelDefinition } from '../../types';

interface DeleteApiKeyModalProps {
  open: boolean;
  secret: SecretInfo | null;
  modelDefinitions: ModelDefinition[];
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteApiKeyModal = ({ open, secret, modelDefinitions, onClose, onSuccess }: DeleteApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [confirmationText, setConfirmationText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const { mutateAsync: deleteSecret, isLoading: isDeleting } = useDeleteSecretMutation();

  const modelCount = modelDefinitions.length;
  const hasModels = modelCount > 0;
  const isConfirmed = !hasModels || confirmationText === secret?.secret_name;

  const handleDelete = async () => {
    if (!secret || !isConfirmed) return;

    setError(null);

    try {
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
            danger
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

        {hasModels && (
          <Alert
            componentId="mlflow.gateway.delete-api-key-modal.warning"
            type="warning"
            message={
              <FormattedMessage
                defaultMessage="This key is currently used by {modelCount, plural, one {# model definition} other {# model definitions}}. After deletion, you will need to attach a different API key to {modelCount, plural, one {this model} other {these models}} via the Edit Endpoint page."
                description="Warning about models using this key"
                values={{ modelCount }}
              />
            }
            closable={false}
          />
        )}

        {hasModels && (
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
