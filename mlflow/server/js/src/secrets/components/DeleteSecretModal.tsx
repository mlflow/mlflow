import {
  DangerModal,
  FormUI,
  Input,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCallback, useState } from 'react';
import type { Secret } from '../types';
import { useDeleteSecretMutation } from '../hooks/useDeleteSecretMutation';
import { BindingsTable } from './BindingsTable';

export interface DeleteSecretModalProps {
  secret: Secret | null;
  visible: boolean;
  onCancel: () => void;
}

export const DeleteSecretModal = ({ secret, visible, onCancel }: DeleteSecretModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [confirmationText, setConfirmationText] = useState('');

  const { deleteSecret, isLoading } = useDeleteSecretMutation({
    onSuccess: () => {
      setConfirmationText('');
      onCancel();
    },
    onError: (error: Error) => {
      // Error handling is done in the mutation hook
      console.error('Failed to delete secret:', error);
    },
  });

  const handleDelete = useCallback(() => {
    if (!secret) return;
    deleteSecret({ secret_id: secret.secret_id });
  }, [secret, deleteSecret]);

  const handleCancel = useCallback(() => {
    setConfirmationText('');
    onCancel();
  }, [onCancel]);

  const isDeleteDisabled = !secret || confirmationText !== secret.secret_name;

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isDeleteDisabled && !isLoading) {
      e.preventDefault();
      handleDelete();
    }
  }, [isDeleteDisabled, isLoading, handleDelete]);

  if (!secret) return null;

  return (
    <DangerModal
      componentId="mlflow.secrets.delete_secret_modal"
      visible={visible}
      onCancel={handleCancel}
      okText={intl.formatMessage({
        defaultMessage: 'Delete',
        description: 'Delete secret modal > delete button text',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Delete secret modal > cancel button text',
      })}
      onOk={handleDelete}
      okButtonProps={{ loading: isLoading, disabled: isDeleteDisabled }}
      title={
        <FormattedMessage
          defaultMessage="Delete Secret"
          description="Delete secret modal > modal title"
        />
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Are you sure you want to delete the secret '{secretName}'? This action cannot be undone."
            description="Delete secret modal > confirmation message"
            values={{ secretName: secret.secret_name }}
          />
        </Typography.Text>

        <BindingsTable secretId={secret.secret_id} variant="warning" isSharedSecret={secret.is_shared} />

        <div>
          <FormUI.Label htmlFor="delete-secret-confirmation-input">
            <FormattedMessage
              defaultMessage="To confirm, type the secret name:"
              description="Delete secret modal > confirmation input label"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.secrets.delete_secret_modal.confirmation_input"
            id="delete-secret-confirmation-input"
            placeholder={secret.secret_name}
            value={confirmationText}
            onChange={(e) => setConfirmationText(e.target.value)}
            autoComplete="off"
            onKeyDown={handleKeyDown}
          />
        </div>
      </div>
    </DangerModal>
  );
};
