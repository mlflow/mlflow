import { useState } from 'react';
import { Alert, Button, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDeleteSecretMutation } from '../../hooks/useDeleteSecretMutation';
import type { Secret, ModelDefinition } from '../../types';

interface DeleteApiKeyModalProps {
  open: boolean;
  secret: Secret | null;
  modelDefinitions: ModelDefinition[];
  bindingCount: number;
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteApiKeyModal = ({
  open,
  secret,
  modelDefinitions,
  bindingCount,
  onClose,
  onSuccess,
}: DeleteApiKeyModalProps) => {
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
            danger={hasModels}
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
              bindingCount > 0 ? (
                <FormattedMessage
                  defaultMessage="This key is used by {modelCount, plural, one {# model} other {# models}} and {bindingCount, plural, one {# resource} other {# resources}}. Deleting it will also delete {modelCount, plural, one {this model} other {these models}}."
                  description="Warning about models and bindings using this key"
                  values={{ modelCount, bindingCount }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="This key is used by {count, plural, one {# model} other {# models}}. Deleting it will also delete {count, plural, one {this model} other {these models}}."
                  description="Warning about models using this key"
                  values={{ count: modelCount }}
                />
              )
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
